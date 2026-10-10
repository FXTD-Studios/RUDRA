// rudra-native: the headless tool over librudra.
//
//   rudra-native version
//   rudra-native info <package>
//   rudra-native diff <package> [--runtime libtorch|onnxruntime|all] [--device cpu|cuda|mps|directml|coreml|rocm|openvino|tensorrt]
//                     [--precision fp32|fp16]
//   rudra-native bench <package> [--runtime ...] [--device ...] [--precision fp32|fp16] [--size 1920x1080,3840x2160] [--iters 5]
//                      [--json out.json] [--budget native/bench/latency_budgets.json --machine <label>]
//   rudra-native master <package> <image> --out <file.exr> [--runtime ...] [--device ...] [--params JSON]
//   rudra-native master-check <package> <golden-dir> [--runtime ...] [--device ...]
//   rudra-native master-compare <package> <workflow-report.json> [--runtime ...] [--device ...]
//   rudra-native deliver <frames> --output <stem> [rudra deliver options]
//   rudra-native video <package> <input> --output <file> [rudra/video.py options] [--runtime ...] [--device ...]
//   rudra-native bench-scopes [--iters 7]
//
// `diff` is the native half of Gate A (NATIVE_ARCHITECTURE.md 12): it runs the
// package's golden frames through each compiled runtime, untiled and tiled, and
// compares against what eager PyTorch produced when the package was exported.
// Exit code 0 only if every runtime asked for passes.
//
// `bench` times inference for the budget table (NATIVE_ARCHITECTURE.md 6.6):
// one warm-up, then the median of --iters runs, untiled and tiled 512/64, on
// a synthetic frame, at each --size (1080p and 4K by default). Wall time,
// fields back in host memory, so a GPU run is timed to completion. Each result
// is also printed as a BENCH line for scripts and, with --json, written to a
// file. With --budget, every result that has a budget for this --machine (a
// label the budget file names, e.g. "rtx4080-win") must be within it, or the
// command exits 1: a slower build fails the gate instead of shipping.
//
// `bench-scopes` times what the viewer does on the CPU after a slider move
// (core/scopes.cpp: the 768-side sample, computeStats, buildScopes and the
// vectorscope) at 1080p and 4K, median of --iters, one thread.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rudra/core/model_manifest.hpp"
#include "rudra/core/scopes.hpp"
#include "rudra/core/view.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/infer/self_test.hpp"
#include "rudra/infer/tiler.hpp"
#include "rudra/platform/npy.hpp"
#include "rudra/platform/tools.hpp"

#ifdef RUDRA_HAVE_STILL_DECODE
#include "batch.hpp"
#include "master.hpp"
#include "video.hpp"
#endif
#include "deliver.hpp"
#include "ffmpeg_check.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

int fail(const Error& e) {
    std::fprintf(stderr, "error [%s]: %s\n  %s\n", to_string(e.code), e.message.c_str(), e.detail.c_str());
    return 2;
}

Result<Device> parse_device(const std::string& s) {
    if (auto d = device_from_string(s)) return *d;
    return make_error(ErrorCode::InvalidArgument, "Unknown device.", s);
}

Result<Precision> parse_precision(const std::string& s) {
    if (auto p = precision_from_string(s)) return *p;
    return make_error(ErrorCode::InvalidArgument, "Unknown precision (fp32 or fp16).", s);
}

Result<std::unique_ptr<InferenceBackend>> make_backend(Runtime rt, const ModelManifest& m, Device d, Precision p) {
    return rt == Runtime::LibTorch ? make_libtorch_backend(m, d, p) : make_onnxruntime_backend(m, d, p);
}

int cmd_info(const fs::path& pkg) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    std::printf("%s  (contract %s)\n", m->name.c_str(), m->contract.c_str());
    std::printf("  source     %s  sha256 %s\n", m->source_file.c_str(), m->source_sha256.c_str());
    std::printf("  heads      residual_gate=%d shadow_gate=%d curve=%d (curve params %d)\n", m->has_residual_gate,
                m->has_shadow_gate, m->has_curve, m->curve_params);
    std::printf("  baseline   corpus_ev %+.1f  log_scale %.1f  max_hdr %.1f\n", m->corpus_ev, m->log_scale, m->max_hdr);
    std::printf("  tiling     %d px, overlap %d\n", m->tile_size, m->overlap);
    auto v = verify_package_files(*m);
    std::printf("  files      %s\n", v ? "sha256 verified" : v.error().detail.c_str());
    std::printf("  runtimes   ");
    for (auto r : compiled_runtimes()) std::printf("%s ", to_string(r));
    std::printf("\n");
    return v ? 0 : 3;
}

int cmd_diff(const fs::path& pkg, const std::string& which, Device device, Precision precision) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    if (auto v = verify_package_files(*m); !v) return fail(v.error());

    std::vector<Runtime> runtimes;
    for (auto r : compiled_runtimes())
        if (which == "all" || which == to_string(r)) runtimes.push_back(r);
    if (runtimes.empty())
        return fail(make_error(ErrorCode::Unsupported, "No requested runtime is compiled into this build.", which));

    bool all_pass = true;
    for (auto rt : runtimes) {
        auto backend = make_backend(rt, *m, device, precision);
        if (!backend && precision == Precision::Fp16 && rt == Runtime::LibTorch && which == "all") {
            std::printf("libtorch: skipped, fp16 runs through ONNX Runtime\n\n");
            continue;
        }
        if (!backend) return fail(backend.error());
        auto r = self_test(*m, **backend);   // infer/self_test, shared with the app's first load
        if (!r) return fail(r.error());
        std::printf("%s %s on %s (%s), %s: %d golden frames + stitch, atol %.0e rtol %.0e (%s)\n",
                    to_string(r->backend.runtime), r->backend.version.c_str(), to_string(r->backend.device),
                    r->backend.detail.c_str(), to_string(r->backend.precision), r->frames, r->tolerance.atol,
                    r->tolerance.rtol, r->tolerance_key.c_str());
        for (const auto& [key, st] : r->outputs)
            std::printf("  %-20s max |d| %.3e  %s\n", key.c_str(), st.max_abs, st.pass() ? "pass" : "FAIL");
        std::printf("  => %s\n\n", r->pass() ? "PASS" : "FAIL");
        all_pass = all_pass && r->pass();
    }
    return all_pass ? 0 : 1;
}

struct BenchSize {
    int w = 0, h = 0;
    std::string label() const { return std::to_string(w) + "x" + std::to_string(h); }
};

// "1920x1080,3840x2160"
std::optional<std::vector<BenchSize>> parse_sizes(const std::string& s) {
    std::vector<BenchSize> out;
    std::size_t at = 0;
    while (at <= s.size()) {
        const std::size_t comma = s.find(',', at);
        const std::string one = s.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        BenchSize b;
        if (std::sscanf(one.c_str(), "%dx%d", &b.w, &b.h) != 2 || b.w <= 0 || b.h <= 0) return std::nullopt;
        out.push_back(b);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return out.empty() ? std::nullopt : std::optional(out);
}

// Deterministic synthetic frame: a lit gradient with a clipped patch and fine
// texture, so every head has something to do.
SdrImage bench_frame(int w, int h) {
    PlanarBuffer b(3, h, w);
    std::uint32_t seed = 20260923u;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            seed = seed * 1664525u + 1013904223u;
            const float n = float(seed >> 8) / float(1u << 24) * 0.02f;
            const float g = float(x) / float(w) * 0.8f + float(y) / float(h) * 0.2f;
            const bool clip = std::abs(x - w * 3 / 4) < w / 10 && std::abs(y - h / 3) < h / 8;
            for (int c = 0; c < 3; ++c) b.at(c, y, x) = clip ? 1.0f : std::clamp(g * (1.0f - 0.15f * float(c)) + n, 0.0f, 1.0f);
        }
    return SdrImage(std::move(b));
}

int cmd_bench(const fs::path& pkg, const std::string& which, Device device, Precision precision,
              const std::vector<BenchSize>& sizes, int iters, const std::string& json_out,
              const std::string& budget_file, const std::string& machine) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());

    nlohmann::json budgets = nlohmann::json::array();
    if (!budget_file.empty()) {
        std::ifstream in(budget_file);
        if (!in) return fail(make_error(ErrorCode::InvalidArgument, "Cannot read the budget file.", budget_file));
        try {
            budgets = nlohmann::json::parse(in).at("budgets");
        } catch (const std::exception& e) {
            return fail(make_error(ErrorCode::InvalidArgument, "The budget file is not valid.", e.what()));
        }
        if (machine.empty())
            return fail(make_error(ErrorCode::InvalidArgument, "--budget needs --machine: budgets are per machine.",
                                   budget_file));
    }
    // A budget row without "precision" is fp32, as every row was before fp16.
    auto budget_for = [&](const char* rt, const char* dev, const std::string& size, const char* mode) -> double {
        for (const auto& b : budgets)
            if (b.value("machine", "") == machine && b.value("runtime", "") == rt && b.value("device", "") == dev &&
                b.value("size", "") == size && b.value("mode", "") == mode &&
                b.value("precision", "fp32") == to_string(precision))
                return b.at("max_ms").get<double>();
        return 0.0;
    };

    std::vector<Runtime> runtimes;
    for (auto r : compiled_runtimes())
        if (which == "all" || which == to_string(r)) runtimes.push_back(r);
    if (runtimes.empty())
        return fail(make_error(ErrorCode::Unsupported, "No requested runtime is compiled into this build.", which));

    nlohmann::json results = nlohmann::json::array();
    int over = 0, checked = 0;
    using clock = std::chrono::steady_clock;
    for (auto rt : runtimes) {
        auto backend = make_backend(rt, *m, device, precision);
        if (!backend && precision == Precision::Fp16 && rt == Runtime::LibTorch && which == "all") {
            std::printf("libtorch: skipped, fp16 runs through ONNX Runtime\n");
            continue;
        }
        if (!backend) return fail(backend.error());
        const auto info = (*backend)->info();
        for (const auto& size : sizes) {
            const SdrImage frame = bench_frame(size.w, size.h);
            std::printf("%s %s on %s (%s), %s, %s, median of %d after one warm-up\n", to_string(info.runtime),
                        info.version.c_str(), to_string(info.device), info.detail.c_str(), size.label().c_str(),
                        to_string(precision), iters);
            const std::pair<const char*, TileConfig> modes[] = {{"untiled", TileConfig{0, 0}},
                                                                {"tiled", TileConfig{m->tile_size, m->overlap}}};
            for (const auto& [name, cfg] : modes) {
                nlohmann::json row{{"runtime", to_string(info.runtime)}, {"device", to_string(info.device)},
                                   {"precision", to_string(precision)},
                                   {"version", info.version}, {"detail", info.detail}, {"size", size.label()},
                                   {"mode", name}, {"iters", iters}};
                std::vector<double> ms;
                std::string error;
                for (int i = 0; i <= iters; ++i) {
                    const auto t0 = clock::now();
                    auto r = infer_frame(**backend, frame, cfg);
                    const auto t1 = clock::now();
                    if (!r) {
                        error = r.error().message;
                        ms.clear();
                        break;
                    }
                    if (i > 0) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
                const double limit = budget_for(to_string(info.runtime), to_string(info.device), size.label(), name);
                if (ms.empty()) {
                    std::printf("  %-8s %s\n", name, error.c_str());
                    row["error"] = error;
                    if (limit > 0.0) {   // a budgeted path that cannot run is over its budget
                        ++checked;
                        ++over;
                        row["budget_ms"] = limit;
                        row["within_budget"] = false;
                    }
                    results.push_back(row);
                    continue;
                }
                std::sort(ms.begin(), ms.end());
                const double med = ms[ms.size() / 2];
                row["median_ms"] = med;
                row["min_ms"] = ms.front();
                std::string verdict;
                if (limit > 0.0) {
                    ++checked;
                    const bool ok = med <= limit;
                    over += !ok;
                    row["budget_ms"] = limit;
                    row["within_budget"] = ok;
                    char buf[64];
                    std::snprintf(buf, sizeof buf, "  budget %8.1f ms  %s", limit, ok ? "ok" : "OVER");
                    verdict = buf;
                }
                std::printf("  %-8s median %8.1f ms  min %8.1f ms%s\n", name, med, ms.front(), verdict.c_str());
                std::printf("BENCH %s %s %s %s %.2f %s\n", to_string(info.runtime), to_string(info.device),
                            size.label().c_str(), name, med, to_string(precision));
                results.push_back(row);
            }
        }
    }
    if (!budget_file.empty())
        std::printf("budgets (%s, machine %s): %d checked, %d over => %s\n", budget_file.c_str(), machine.c_str(),
                    checked, over, over ? "FAIL" : "PASS");
    if (!json_out.empty()) {
        nlohmann::json doc{{"package", m->name}, {"source_sha256", m->source_sha256}, {"machine", machine},
                           {"precision", to_string(precision)}, {"results", results}};
        if (!budget_file.empty())
            doc["budget"] = {{"file", budget_file}, {"checked", checked}, {"over", over}, {"pass", over == 0}};
        std::ofstream out(json_out);
        out << doc.dump(2) << "\n";
        if (!out) return fail(make_error(ErrorCode::IoError, "Cannot write the bench JSON.", json_out));
    }
    return over ? 1 : 0;
}

#ifdef RUDRA_HAVE_STILL_DECODE
Result<std::unique_ptr<InferenceBackend>> backend_for(const ModelManifest& m, const std::string& runtime, Device device) {
    if (runtime == "onnxruntime") return make_onnxruntime_backend(m, device);
    if (runtime == "libtorch") return make_libtorch_backend(m, device);
    // Default: the reference runtime when it is compiled in.
    for (auto r : compiled_runtimes())
        if (r == Runtime::LibTorch) return make_libtorch_backend(m, device);
    return make_onnxruntime_backend(m, device);
}

int cmd_master(const fs::path& pkg, const fs::path& image, const fs::path& out, const std::string& runtime,
               Device device, const std::string& params) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    auto b = backend_for(*m, runtime, device);
    if (!b) return fail(b.error());
    auto q = master_request_from_json(params.empty() ? "{}" : params);
    if (!q) return fail(q.error());
    if (q->checkpoint.empty()) q->checkpoint = m->source_file;
    auto r = render_master(*m, **b, image, *q, out);
    if (!r) return fail(r.error());
    std::printf("%s  %dx%d, %d-bit source, MaxCLL %d, MaxFALL %d, peak %.1f nits\n  sidecar %s\n",
                r->exr.string().c_str(), r->width, r->height, r->source_bits, r->maxcll, r->maxfall, r->peak_nits,
                r->sidecar.string().c_str());
    return 0;
}

int half_ulp(std::uint16_t a, std::uint16_t b) {
    auto key = [](std::uint16_t h) { return (h & 0x8000) ? -int(h & 0x7fff) : int(h); };
    return std::abs(key(a) - key(b));
}

std::string read_text(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

// Every golden master from tools/emit_master_golden.py, rendered here and
// compared: EXR pixels within 1 half-float ulp, every EXR header attribute
// equal, every sidecar field equal (peak within its 0.1-nit rounding).
// A UTF-8 path from a JSON string (Windows paths may hold any character).
fs::path utf8_path(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }

struct MasterDiff {
    bool ok = false, bytes_equal = false;
    int worst = 0;
    std::size_t off = 0, total = 0;
    std::string header_diff, side_diff;
};

// A master against a reference: pixels within 1 half ulp, the header's
// attributes equal, every sidecar key equal (peak_nits within 0.1).
Result<MasterDiff> diff_master(const fs::path& got_exr, const fs::path& got_side, const fs::path& want_exr,
                               const fs::path& want_side) {
    auto got = read_exr(got_exr), want = read_exr(want_exr);
    if (!got) return got.error();
    if (!want) return want.error();
    MasterDiff d;
    for (std::size_t i = 0; i < want->half_bits.size() && i < got->half_bits.size(); ++i) {
        const int u = half_ulp(got->half_bits[i], want->half_bits[i]);
        d.worst = std::max(d.worst, u);
        d.off += u != 0;
    }
    d.total = want->half_bits.size();
    const bool same_shape = got->half_bits.size() == want->half_bits.size() && !want->half_bits.empty();
    if (got->attributes.size() != want->attributes.size()) d.header_diff = "attribute count";
    for (std::size_t i = 0; d.header_diff.empty() && i < want->attributes.size(); ++i)
        if (got->attributes[i] != want->attributes[i] || got->attribute_types[i] != want->attribute_types[i])
            d.header_diff = want->attributes[i].first;
    try {
        const auto gs = nlohmann::json::parse(read_text(got_side));
        const auto ws = nlohmann::json::parse(read_text(want_side));
        for (const auto& [k, v] : ws.items()) {
            if (!gs.contains(k)) { d.side_diff = k; break; }
            if (k == "peak_nits") {
                if (std::abs(gs[k].get<double>() - v.get<double>()) > 0.1 + 1e-9) d.side_diff = k;
            } else if (gs[k] != v) {
                d.side_diff = k;
            }
            if (!d.side_diff.empty()) break;
        }
    } catch (const std::exception& e) {
        d.side_diff = std::string("unreadable: ") + e.what();
    }
    d.bytes_equal = read_text(got_side) == read_text(want_side);
    d.ok = same_shape && d.worst <= 1 && d.header_diff.empty() && d.side_diff.empty();
    return d;
}

void print_diff(const std::string& name, int w, int h, const MasterDiff& d) {
    std::printf("  %-18s %dx%d  pixels %s (worst %d half ulp, %zu of %zu off)  header %s  sidecar %s%s  => %s\n",
                name.c_str(), w, h, d.worst <= 1 ? "ok" : "FAIL", d.worst, d.off, d.total,
                d.header_diff.empty() ? "equal" : ("differs at " + d.header_diff).c_str(),
                d.side_diff.empty() ? "equal" : ("differs at " + d.side_diff).c_str(),
                d.bytes_equal ? ", byte-identical" : "", d.ok ? "PASS" : "FAIL");
}

int cmd_master_check(const fs::path& pkg, const fs::path& dir, const std::string& runtime, Device device) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    auto b = backend_for(*m, runtime, device);
    if (!b) return fail(b.error());
    std::ifstream in(dir / "index.json");
    if (!in) return fail(make_error(ErrorCode::NotFound, "No master goldens there.", dir.string()));
    const auto idx = nlohmann::json::parse(in);
    const fs::path tmp = fs::temp_directory_path() / "rudra_master_check";
    fs::create_directories(tmp);
    const auto info = (*b)->info();
    std::printf("master parity: %s %s on %s, %zu cases (oracle %s)\n", to_string(info.runtime), info.version.c_str(),
                to_string(info.device), idx.at("cases").size(), idx.value("oracle", "").c_str());
    bool all = true;
    for (const auto& c : idx.at("cases")) {
        const std::string name = c.at("name");
        auto q = master_request_from_json(c.at("params").dump());
        if (!q) return fail(q.error());
        const fs::path out = tmp / c.at("exr").get<std::string>();
        auto r = render_master(*m, **b, dir / c.at("image").get<std::string>(), *q, out);
        if (!r) return fail(r.error());
        auto d = diff_master(out, r->sidecar, dir / c.at("exr").get<std::string>(), dir / c.at("sidecar").get<std::string>());
        if (!d) return fail(d.error());
        print_diff(name, r->width, r->height, *d);
        all = all && d->ok;
    }
    std::printf("  => %s\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}

// The app's masters from RUDRA --workflow-check, against this CLI's own
// master of the same frame with the same parameters (the path master-check
// holds to the Studio).
int cmd_master_compare(const fs::path& pkg, const fs::path& report, const std::string& runtime, Device device) {
    auto m = read_manifest(pkg);
    if (!m) return fail(m.error());
    auto b = backend_for(*m, runtime, device);
    if (!b) return fail(b.error());
    nlohmann::json rep;
    try {
        rep = nlohmann::json::parse(read_text(report));
    } catch (const std::exception& e) {
        return fail(make_error(ErrorCode::ParseError, "The workflow report could not be read.", e.what()));
    }
    if (!rep.contains("master") || !rep["master"].contains("masters"))
        return fail(make_error(ErrorCode::NotFound, "The workflow report has no masters.", report.string()));
    const fs::path tmp = fs::temp_directory_path() / "rudra_master_compare";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    const auto info = (*b)->info();
    std::printf("app masters against rudra-native master: %s %s on %s, %zu frames\n", to_string(info.runtime),
                info.version.c_str(), to_string(info.device), rep["master"]["masters"].size());
    bool all = !rep["master"]["masters"].empty();
    int i = 0;
    for (const auto& c : rep["master"]["masters"]) {
        auto q = master_request_from_json(c.at("params").get<std::string>());
        if (!q) return fail(q.error());
        const fs::path out = tmp / ("cli_" + std::to_string(i++) + ".exr");
        auto r = render_master(*m, **b, utf8_path(c.at("frame").get<std::string>()), *q, out);
        if (!r) return fail(r.error());
        auto d = diff_master(utf8_path(c.at("exr").get<std::string>()), utf8_path(c.at("sidecar").get<std::string>()),
                             out, r->sidecar);
        if (!d) return fail(d.error());
        print_diff(utf8_path(c.at("frame").get<std::string>()).filename().string(), r->width, r->height, *d);
        all = all && d->ok;
    }
    std::printf("  => %s\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}
#endif

int cmd_bench_scopes(int iters) {
    std::printf("Viewer measurements and scopes on the CPU (sample, computeStats, buildScopes, vectorscope), median of %d\n",
                iters);
    for (auto [w, h] : {std::pair{1920, 1080}, std::pair{3840, 2160}}) {
        PlanarBuffer m(3, h, w), b(3, h, w);
        std::uint32_t x = 2463534242u;   // xorshift: a busy, noise-like frame
        for (auto* buf : {&m, &b})
            for (float& v : buf->span()) {
                x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                v = float(x % 100000u) / 100000.0f * 0.3f;
            }
        const std::vector<float> hi(std::size_t(w) * h, 0.7f), sh(std::size_t(w) * h, 0.2f);
        const Reductions rm = reduce_ladder(m), rb = reduce_ladder(b);
        std::vector<double> t_measure, t_vector;
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            const SampleGrid g = sample_grid(w, h);
            const PlanarBuffer ms = take_sample(m, g), bs = take_sample(b, g);
            const Measured r = measure_view(ms, bs, g, hi, sh, MaskCoverage{}, rm, rb, std::size_t(w) * h);
            const auto t1 = std::chrono::steady_clock::now();
            const auto img = vectorscope(ms);
            const auto t2 = std::chrono::steady_clock::now();
            if (r.scopes.mid.empty() || img.empty()) return 1;
            t_measure.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            t_vector.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        std::sort(t_measure.begin(), t_measure.end());
        std::sort(t_vector.begin(), t_vector.end());
        const SampleGrid g = sample_grid(w, h);
        const double tm = t_measure[t_measure.size() / 2], tv = t_vector[t_vector.size() / 2];
        std::printf("  %dx%d (sample %dx%d): measurements and scopes %.2f ms, vectorscope %.2f ms\n", w, h, g.width,
                    g.height, tm, tv);
        std::printf("BENCH scopes %dx%d %.3f %.3f\n", w, h, tm, tv);
    }
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: rudra-native version\n"
                 "       rudra-native info <package>\n"
                 "       rudra-native diff <package> [--runtime libtorch|onnxruntime|all] [--device cpu|cuda|mps|"
                 "directml|coreml|rocm|openvino|tensorrt] [--precision fp32|fp16]\n"
                 "       rudra-native bench <package> [--runtime ...] [--device ...] [--precision fp32|fp16] [--size WxH[,WxH]] [--iters N]\n"
                 "                          [--json FILE] [--budget FILE --machine LABEL]\n"
                 "       rudra-native master <package> <image> --out <file.exr> [--runtime ...] [--device ...] [--params JSON]\n"
                 "       rudra-native master-check <package> <golden-dir> [--runtime ...] [--device ...]\n"
                 "       rudra-native master-compare <package> <workflow-report.json> [--runtime ...] [--device ...]\n"
                 "       rudra-native video <package> <input> --output <file> [rudra/video.py options] [--runtime ...] [--device ...]\n"
                 "       rudra-native batch run <queue.json> [--retry-failed] [--package DIR] [--runtime ...] [--device ...]\n"
                 "       rudra-native batch status <queue.json>\n"
                 "       rudra-native deliver <frames> --output <stem> [--target hdr10|hlg|prores422hq|prores4444] [--fps N]\n"
                 "                            [--peak-nits N] [--min-nits N] [--source-space rec709|rec2020|p3d65] [--nits-scale N] [--no-verify-tags]\n"
                 "       rudra-native ffmpeg-check [--ffmpeg PATH] [--ffprobe PATH] [--force] [--no-self-test]\n"
                 "       rudra-native bench-scopes [--iters N]\n");
}

}  // namespace

int main(int argc, char** argv) {
    // A package's own ffmpeg and ffprobe go first on the PATH (platform/tools.hpp);
    // RUDRA_FFMPEG_DIR=path keeps the user's.
    rudra::use_bundled_tools(rudra::executable_dir());
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { usage(); return 64; }
    if (args[0] == "version") {
        std::printf("rudra-native 0.1.0 (model contract %d.x)\n", kSupportedContractMajor);
        return 0;
    }
    if (args[0] == "ffmpeg-check") return cmd_ffmpeg_check(std::vector<std::string>(args.begin() + 1, args.end()));
    if (args[0] == "bench-scopes") {
        int iters = 7;
        if (args.size() == 3 && args[1] == "--iters") iters = std::max(1, std::atoi(args[2].c_str()));
        return cmd_bench_scopes(iters);
    }
    if (args.size() < 2) { usage(); return 64; }
    if (args[0] == "deliver") return cmd_deliver(std::vector<std::string>(args.begin() + 1, args.end()));
#ifdef RUDRA_HAVE_STILL_DECODE
    if (args[0] == "batch") return cmd_batch(std::vector<std::string>(args.begin() + 1, args.end()));
    if (args[0] == "video") return cmd_video(std::vector<std::string>(args.begin() + 1, args.end()));
#endif
    const fs::path pkg = args[1];
    if (args[0] == "info") return cmd_info(pkg);
    if (args[0] == "diff") {
        std::string runtime = "all", device = "cpu", precision = "fp32";
        for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
            if (args[i] == "--runtime") runtime = args[i + 1];
            else if (args[i] == "--device") device = args[i + 1];
            else if (args[i] == "--precision") precision = args[i + 1];
            else { usage(); return 64; }
        }
        auto d = parse_device(device);
        if (!d) return fail(d.error());
        auto p = parse_precision(precision);
        if (!p) return fail(p.error());
        return cmd_diff(pkg, runtime, *d, *p);
    }
#ifdef RUDRA_HAVE_STILL_DECODE
    if (args[0] == "master" || args[0] == "master-check" || args[0] == "master-compare") {
        if (args.size() < 3) { usage(); return 64; }
        std::string runtime = "auto", device = "cpu", out, params;
        for (std::size_t i = 3; i + 1 < args.size(); i += 2) {
            if (args[i] == "--runtime") runtime = args[i + 1];
            else if (args[i] == "--device") device = args[i + 1];
            else if (args[i] == "--out") out = args[i + 1];
            else if (args[i] == "--params") params = args[i + 1];
            else { usage(); return 64; }
        }
        auto d = parse_device(device);
        if (!d) return fail(d.error());
        if (args[0] == "master-check") return cmd_master_check(pkg, args[2], runtime, *d);
        if (args[0] == "master-compare") return cmd_master_compare(pkg, args[2], runtime, *d);
        if (out.empty()) { usage(); return 64; }
        return cmd_master(pkg, args[2], out, runtime, *d, params);
    }
#endif
    if (args[0] == "bench") {
        std::string runtime = "all", device = "cpu", precision = "fp32", json_out, budget, machine;
        std::vector<BenchSize> sizes{{1920, 1080}, {3840, 2160}};
        int iters = 5;
        for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
            if (args[i] == "--runtime") runtime = args[i + 1];
            else if (args[i] == "--device") device = args[i + 1];
            else if (args[i] == "--size") {
                auto s = parse_sizes(args[i + 1]);
                if (!s) { usage(); return 64; }
                sizes = *s;
            } else if (args[i] == "--iters") iters = std::max(1, std::atoi(args[i + 1].c_str()));
            else if (args[i] == "--json") json_out = args[i + 1];
            else if (args[i] == "--budget") budget = args[i + 1];
            else if (args[i] == "--machine") machine = args[i + 1];
            else if (args[i] == "--precision") precision = args[i + 1];
            else { usage(); return 64; }
        }
        auto d = parse_device(device);
        if (!d) return fail(d.error());
        auto p = parse_precision(precision);
        if (!p) return fail(p.error());
        return cmd_bench(pkg, runtime, *d, *p, sizes, iters, json_out, budget, machine);
    }
    usage();
    return 64;
}
