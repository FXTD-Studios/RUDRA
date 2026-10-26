// rudra-native aces-v2-serve: the ACES v2 staged master (core/aces_v2.hpp) as a
// long-running engine for the app (26 Oct 2026). The view and the network load
// once; requests then come as one JSON object per line on stdin and every
// answer is one JSON object per line on stdout, so the app can show a frame
// through the model at any exposure, compare it with the plain ACES inverse,
// and master with nothing loaded twice. The masters are the ones
// `rudra-native aces-v2` writes (same stages, same EXR, same provenance).
//
//   rudra-native aces-v2-serve <package> [--runtime libtorch|onnxruntime] [--device cpu|cuda|directml]
//                              [--ocio-threads N] [--no-smoothing]
// On start:  {"event":"ready", ...}  or  {"event":"error", ...} and exit 2.
// Requests (id is echoed):
//   {"id":1,"op":"preview","source":"<still or video>","frame":0,"guard":true}
//       runs the frame, keeps it, answers its size and statistics
//   {"id":2,"op":"view","ev":-2,"mode":"model|plain|source|mask|split","max_side":1600,"out":"<file.ppm>"}
//       the kept frame through the ACES 2.0 SDR view at that exposure, 8-bit PPM
//   {"id":3,"op":"master","sources":["<still or video>",...],"out":"<new dir>","guard":true}
//       a video is mastered frame by frame as <stem>_<NNNNNN>.exr
//   {"id":4,"op":"probe","source":"<still or video>"}  -> frames, width, height, fps
//       {"id":3,"progress":{...}} per frame, then {"id":3,"ok":true,"done":true,...}
//   {"op":"cancel"}  stops a master after the frame in hand
//   {"op":"quit"}

#include "aces_v2_serve.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

#include <nlohmann/json.hpp>

#include "rudra/core/aces_v2.hpp"
#include "rudra/core/aces_v2_manifest.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
#include "rudra/media/still.hpp"
#include "aces_v2_inputs.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace rudra;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

std::mutex g_out;
void send(const json& j) {
    std::lock_guard l(g_out);
    std::fputs((j.dump() + "\n").c_str(), stdout);
    std::fflush(stdout);
}
json err(const json& id, const std::string& message, const std::string& detail = {}) {
    return {{"id", id}, {"ok", false}, {"error", message}, {"detail", detail}};
}
json err(const json& id, const Error& e) { return err(id, e.message, e.detail); }

// The kept frame for "view".
struct Kept {
    fs::path source;
    PlanarBuffer codes, baseline, master, mask;
};

std::size_t count_positive(const PlanarBuffer& m) {
    std::size_t n = 0;
    for (float v : m.span()) n += v > 0.0f;
    return n;
}

double peak_luma(const PlanarBuffer& rgb) {
    double peak = 0;
    for (std::size_t i = 0; i < rgb.plane_size(); ++i)
        peak = std::max(peak, aces_v2::kLumaAp1[0] * rgb.plane(0)[i] + aces_v2::kLumaAp1[1] * rgb.plane(1)[i] +
                                  aces_v2::kLumaAp1[2] * rgb.plane(2)[i]);
    return peak;
}

// The display image of `mode` at `ev`, box-averaged down to max_side, as 8-bit RGB.
std::vector<std::uint8_t> render_view(const Kept& k, const aces_v2::AcesView& view, const std::string& mode, double ev,
                                      int max_side, int& out_w, int& out_h) {
    const int h = k.codes.height(), w = k.codes.width();
    const float gain = static_cast<float>(std::exp2(ev));
    auto shown = [&](const PlanarBuffer& lin) {
        PlanarBuffer s = lin;
        for (float& v : s.span()) v *= gain;
        return aces_v2::view_forward(view, s);
    };
    PlanarBuffer disp;
    if (mode == "source") disp = k.codes;
    else if (mode == "plain") disp = shown(k.baseline);
    else {
        disp = shown(k.master);
        if (mode == "mask") {
            for (std::size_t i = 0; i < disp.plane_size(); ++i) {
                const float m = k.mask.plane(0)[i] > 0.0f ? std::min(1.0f, 0.35f + 0.65f * k.mask.plane(0)[i]) : 0.0f;
                const float magenta[3] = {1.0f, 0.0f, 1.0f};
                for (int c = 0; c < 3; ++c) disp.plane(c)[i] = disp.plane(c)[i] * 0.35f * (1 - m) + magenta[c] * m;
            }
        } else if (mode == "split") {
            const PlanarBuffer plain = shown(k.baseline);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w / 2; ++x)
                    for (int c = 0; c < 3; ++c) disp.at(c, y, x) = plain.at(c, y, x);
            for (int y = 0; y < h; ++y)
                for (int c = 0; c < 3; ++c) disp.at(c, y, w / 2) = 1.0f;
        }
    }
    const int f = std::max(1, (std::max(h, w) + std::max(1, max_side) - 1) / std::max(1, max_side));
    out_w = w / f;
    out_h = h / f;
    std::vector<std::uint8_t> rgb(std::size_t(out_w) * out_h * 3);
    for (int y = 0; y < out_h; ++y)
        for (int x = 0; x < out_w; ++x)
            for (int c = 0; c < 3; ++c) {
                double s = 0;
                for (int dy = 0; dy < f; ++dy)
                    for (int dx = 0; dx < f; ++dx) s += disp.at(c, y * f + dy, x * f + dx);
                const double v = std::clamp(s / (f * f), 0.0, 1.0);
                rgb[(std::size_t(y) * out_w + x) * 3 + c] = static_cast<std::uint8_t>(std::lround(v * 255.0));
            }
    return rgb;
}

bool write_ppm(const fs::path& p, const std::vector<std::uint8_t>& rgb, int w, int h) {
    std::ofstream o(p, std::ios::binary);
    o << "P6\n" << w << " " << h << "\n255\n";
    o.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
    return bool(o);
}

// Requests from stdin on their own thread, so "cancel" reaches a running master.
struct Inbox {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<json> q;
    bool closed = false;
    std::atomic<bool> cancel{false};
};

}  // namespace

int cmd_aces_v2_serve(const std::vector<std::string>& args) {
    if (args.empty() || args[0] == "--help") {
        std::fprintf(stderr, "usage: rudra-native aces-v2-serve <package> [--runtime libtorch|onnxruntime] [--device cpu|cuda|directml] [--ocio-threads N] [--no-smoothing]\n");
        return 64;
    }
    const auto t0 = Clock::now();
    const fs::path pkg = args[0];
    std::string device = "cpu", runtime = "libtorch";
    int threads = 8;
    bool smoothing = true;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--device" && i + 1 < args.size()) device = args[++i];
        else if (args[i] == "--runtime" && i + 1 < args.size()) runtime = args[++i];
        else if (args[i] == "--ocio-threads" && i + 1 < args.size()) threads = std::atoi(args[++i].c_str());
        else if (args[i] == "--no-smoothing") smoothing = false;
        else {
            send({{"event", "error"}, {"error", "Unknown option."}, {"detail", args[i]}});
            return 2;
        }
    }
    auto fail_start = [](const Error& e) {
        send({{"event", "error"}, {"error", e.message}, {"detail", e.detail}});
        return 2;
    };
    auto m = aces_v2::read_manifest(pkg);
    if (!m) return fail_start(m.error());
    if (auto v = aces_v2::verify_files(*m); !v) return fail_start(v.error());
    if (smoothing && m->smoothing != "hybrid64")
        return fail_start(make_error(ErrorCode::ContractMismatch, "This package declares no clipped smoothing."));
    auto view = aces_v2::make_ocio_view(threads);
    if (!view) return fail_start(view.error());
    Result<aces_v2::G1Fn> g1 = make_error(ErrorCode::Unsupported, "Unknown runtime (libtorch or onnxruntime).", runtime);
    if (runtime == "libtorch") {
#ifdef RUDRA_HAVE_LIBTORCH_ACES
        g1 = aces_v2::load_g1_libtorch(m->g1, device);
#else
        g1 = make_error(ErrorCode::Unsupported, "This build has no LibTorch.");
#endif
    } else if (runtime == "onnxruntime") {
        if (m->g1_onnx.empty()) return fail_start(make_error(ErrorCode::Unsupported, "This package has no ONNX model."));
#ifdef RUDRA_HAVE_ORT_ACES
        g1 = aces_v2::load_g1_onnxruntime(m->g1_onnx, device);
#else
        g1 = make_error(ErrorCode::Unsupported, "This build has no ONNX Runtime.");
#endif
    }
    if (!g1) return fail_start(g1.error());
    // Warm-up on a small grey frame: the first CUDA call is the slow one.
    {
        PlanarBuffer warm(3, 64, 64, 0.5f);
        (void)aces_v2::run_frame(warm, **view, *g1, smoothing, true, m->forbid_new_negative_y);
    }
    send({{"event", "ready"}, {"package", pkg.string()}, {"g1_sha256", m->g1_sha256}, {"checkpoint_sha256", m->checkpoint_sha256},
          {"runtime", runtime}, {"device", device}, {"view", (*view)->identity()}, {"luminance_guard_default", m->forbid_new_negative_y},
          {"release_authorized", m->release_authorized}, {"load_ms", ms_since(t0)}});

    Inbox in;
    std::thread reader([&in] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            json j = json::parse(line, nullptr, false);
            if (j.is_discarded() || !j.is_object()) {
                send(err(nullptr, "A request must be one JSON object per line.", line.substr(0, 200)));
                continue;
            }
            const std::string op = j.value("op", std::string());
            if (op == "cancel") {
                in.cancel = true;
                continue;
            }
            std::lock_guard l(in.mu);
            in.q.push_back(std::move(j));
            in.cv.notify_one();
        }
        std::lock_guard l(in.mu);
        in.closed = true;
        in.cv.notify_one();
    });

    std::optional<Kept> kept;
    for (;;) {
        json req;
        {
            std::unique_lock l(in.mu);
            in.cv.wait(l, [&] { return in.closed || !in.q.empty(); });
            if (in.q.empty()) break;
            req = std::move(in.q.front());
            in.q.pop_front();
        }
        const json id = req.value("id", json());
        const std::string op = req.value("op", std::string());
        try {
            if (op == "quit") break;
            const bool guard = req.value("guard", m->forbid_new_negative_y);
            if (op == "preview") {
                const fs::path src = req.at("source").get<std::string>();
                const int frame_index = req.value("frame", 0);
                const auto t = Clock::now();
                auto still = aces_read_one(src, frame_index, fs::temp_directory_path() / "rudra_aces_preview_decode.log");
                if (!still) {
                    send(err(id, still.error()));
                    continue;
                }
                const PlanarBuffer codes = still->codes;
                auto r = aces_v2::run_frame(codes, **view, *g1, smoothing, true, guard);
                if (!r) {
                    send(err(id, r.error()));
                    continue;
                }
                Kept k{src, codes, aces_v2::view_inverse(**view, codes), std::move(r->acescg), std::move(r->mask)};
                const double ms = ms_since(t);
                const std::size_t recon = count_positive(k.mask);
                send({{"id", id}, {"ok", true}, {"source", src.string()}, {"frame", frame_index}, {"label", still->source},
                      {"width", codes.width()}, {"height", codes.height()},
                      {"bits", still->bits}, {"ms", ms}, {"reconstructed_fraction", double(recon) / double(k.mask.plane_size())},
                      {"peak", peak_luma(k.master)}, {"peak_plain", peak_luma(k.baseline)},
                      {"fully_clipped_pixels", r->smoothing.full_pixels}, {"remaining_display_errors", r->constraint.remaining_display_errors},
                      {"new_negative_luminance", r->constraint.new_negative_luminance}, {"luminance_guard", guard}});
                kept = std::move(k);
            } else if (op == "probe") {
                const fs::path src = req.at("source").get<std::string>();
                if (aces_is_video(src)) {
                    auto vs = aces_open_video(src);
                    if (!vs) {
                        send(err(id, vs.error()));
                        continue;
                    }
                    send({{"id", id}, {"ok", true}, {"video", true}, {"frames", vs->clock.frames}, {"width", vs->stream.width},
                          {"height", vs->stream.height}, {"fps", vs->clock.fps}, {"bits", aces_video_bits(vs->stream.pix_fmt)}});
                } else {
                    send({{"id", id}, {"ok", true}, {"video", false}, {"frames", aces_is_still(src) ? 1 : 0}});
                }
            } else if (op == "view") {
                if (!kept) {
                    send(err(id, "Preview a frame first."));
                    continue;
                }
                const auto t = Clock::now();
                int w = 0, h = 0;
                const auto rgb = render_view(*kept, **view, req.value("mode", std::string("model")), req.value("ev", 0.0),
                                             req.value("max_side", 1600), w, h);
                const fs::path out = req.at("out").get<std::string>();
                if (!write_ppm(out, rgb, w, h)) {
                    send(err(id, "The preview image could not be written.", out.string()));
                    continue;
                }
                send({{"id", id}, {"ok", true}, {"path", out.string()}, {"width", w}, {"height", h}, {"ms", ms_since(t)}});
            } else if (op == "master") {
                in.cancel = false;
                std::vector<fs::path> sources;
                for (const auto& s : req.at("sources")) sources.emplace_back(s.get<std::string>());
                const fs::path out = req.at("out").get<std::string>();
                if (sources.empty()) {
                    send(err(id, "No stills or videos to master."));
                    continue;
                }
                auto frames = AcesInputs::open(sources, fs::temp_directory_path() / "rudra_aces_serve_decode.log");
                if (!frames) {
                    send(err(id, frames.error()));
                    continue;
                }
                if (fs::exists(out)) {
                    send(err(id, "Choose a new output folder.", out.string()));
                    continue;
                }
                fs::create_directories(out);
                json run = {{"contract", aces_v2::kContractId}, {"package", pkg.string()}, {"g1_sha256", m->g1_sha256},
                            {"checkpoint_sha256", m->checkpoint_sha256}, {"model_contract", m->model_contract_id},
                            {"view", (*view)->identity()}, {"device", device}, {"runtime", runtime},
                            {"g1_file", runtime == "onnxruntime" ? m->g1_onnx.filename().string() : m->g1.filename().string()},
                            {"clipped_smoothing", smoothing ? m->smoothing : "none"},
                            {"appearance_constraint", guard ? "1 code, include fully clipped, no new negative luminance"
                                                            : "1 code, include fully clipped"},
                            {"luminance_guard", guard}, {"luminance_guard_package_default", m->forbid_new_negative_y},
                            {"master", "ACEScg scene-linear, signed, unclamped, float32"},
                            {"mask", "rudra.reconstructed = highlight reconstruction weight; 0 = from source (incl. analytic shadows)"},
                            {"not", "not camera RAW; reconstructed highlights are estimates"},
                            {"release_authorized", m->release_authorized}, {"engine", "aces-v2-serve"}, {"frames", json::array()}};
                bool all_ok = true, stopped = false;
                std::optional<Error> failure;
                const std::size_t total = (*frames)->total();
                for (std::size_t i = 0; i < total; ++i) {
                    if (in.cancel) {
                        stopped = true;
                        break;
                    }
                    auto f = (*frames)->next();
                    if (!f) {
                        failure = f.error();
                        break;
                    }
                    const auto t = Clock::now();
                    auto r = aces_v2::run_frame(f->codes, **view, *g1, smoothing, true, guard);
                    if (!r) {
                        failure = r.error();
                        break;
                    }
                    const double ms = ms_since(t);
                    const fs::path exr = out / f->exr;
                    const ExrAttributes prov = {{"contract", aces_v2::kContractId}, {"g1_sha256", m->g1_sha256},
                                                {"source", f->source}, {"view", (*view)->identity()}};
                    if (auto w = write_acescg_exr(exr, r->acescg, Primaries::Ap1, 1.0, prov, /*half=*/false, &r->mask); !w) {
                        failure = w.error();
                        break;
                    }
                    const std::size_t recon = count_positive(r->mask);
                    const bool ok = r->constraint.remaining_display_errors == 0 && (!guard || r->constraint.new_negative_luminance == 0);
                    all_ok = all_ok && ok;
                    const json frame = {{"source", f->source_path}, {"frame", f->source}, {"exr", exr.filename().string()}, {"bits", f->bits}, {"ms", ms},
                                        {"reconstructed_fraction", double(recon) / double(r->mask.plane_size())},
                                        {"fully_clipped_pixels", r->smoothing.full_pixels},
                                        {"smoothed_components", r->smoothing.components},
                                        {"constraint_support", r->constraint.support_pixels},
                                        {"constraint_changed", r->constraint.changed_pixels},
                                        {"constraint_baseline_fallback", r->constraint.baseline_fallback_pixels},
                                        {"remaining_display_errors", r->constraint.remaining_display_errors},
                                        {"new_negative_luminance", r->constraint.new_negative_luminance}};
                    run["frames"].push_back(frame);
                    send({{"id", id}, {"progress", {{"index", i + 1}, {"total", total}, {"source", f->source},
                                                    {"exr", exr.filename().string()}, {"ms", ms},
                                                    {"remaining_display_errors", r->constraint.remaining_display_errors}}}});
                }
                run["appearance_constraint_pass"] = all_ok;
                run["cancelled"] = stopped;
                std::ofstream(out / "provenance.json") << run.dump(1);
                if (failure) send(err(id, *failure));
                else
                    send({{"id", id}, {"ok", true}, {"done", true}, {"cancelled", stopped}, {"frames", run["frames"].size()},
                          {"appearance_pass", all_ok}, {"provenance", (out / "provenance.json").string()}});
            } else {
                send(err(id, "Unknown op.", op));
            }
        } catch (const std::exception& e) {
            send(err(id, "Bad request.", e.what()));
        }
    }
    // stdin stays open in the reader until the app closes it or exits.
    if (reader.joinable()) reader.detach();
    // Leave at once: the reader may still be blocked on stdin, and runtime
    // thread pools can hold up a normal exit (the app waits for this process).
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(0);
}
