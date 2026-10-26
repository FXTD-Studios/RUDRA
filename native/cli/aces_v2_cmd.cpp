// rudra-native aces-v2: the ACES v2 staged master (core/aces_v2.hpp) for
// stills: 32-bit float ACEScg EXR with the rudra.reconstructed mask, plus a
// JSON sidecar per frame and a run provenance file. Same contract as
// tools/rudra_aces_export.py --clipped-smoothing hybrid64 --colour-backtrack-all.

#include "aces_v2_cmd.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

#include "rudra/core/aces_v2.hpp"
#include "rudra/core/aces_v2_manifest.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
#include "rudra/media/still.hpp"

namespace fs = std::filesystem;
using namespace rudra;

namespace {

int fail_v2(const Error& e) {
    std::fprintf(stderr, "error [%s]: %s\n  %s\n", to_string(e.code), e.message.c_str(), e.detail.c_str());
    return 2;
}

bool is_still(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e == ".png" || e == ".tif" || e == ".tiff" || e == ".jpg" || e == ".jpeg" || e == ".bmp" || e == ".webp";
}

}  // namespace

int cmd_aces_v2(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::fprintf(stderr, "usage: rudra-native aces-v2 <package> <image|folder> --out <dir> [--runtime libtorch|onnxruntime] [--device cpu|cuda|directml] [--ocio-threads N] [--no-smoothing]\n");
        return 64;
    }
    const fs::path pkg = args[0], input = args[1];
    fs::path out;
    std::string device = "cpu", runtime = "libtorch";
    int threads = 4;
    bool smoothing = true;
    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--out" && i + 1 < args.size()) out = args[++i];
        else if (args[i] == "--device" && i + 1 < args.size()) device = args[++i];
        else if (args[i] == "--runtime" && i + 1 < args.size()) runtime = args[++i];
        else if (args[i] == "--ocio-threads" && i + 1 < args.size()) threads = std::atoi(args[++i].c_str());
        else if (args[i] == "--no-smoothing") smoothing = false;
        else return fail_v2(make_error(ErrorCode::InvalidArgument, "Unknown option.", args[i]));
    }
    if (out.empty()) return fail_v2(make_error(ErrorCode::InvalidArgument, "--out <dir> is required."));
    if (fs::exists(out)) return fail_v2(make_error(ErrorCode::InvalidArgument, "Choose a new output folder.", out.string()));

    // Validate the complete output mapping before loading a model or writing a
    // master. Different input extensions can otherwise overwrite the same EXR.
    std::vector<fs::path> frames;
    if (fs::is_directory(input)) {
        for (const auto& e : fs::directory_iterator(input))
            if (e.is_regular_file() && is_still(e.path())) frames.push_back(e.path());
        std::sort(frames.begin(), frames.end());
    } else if (fs::is_regular_file(input) && is_still(input)) {
        frames.push_back(input);
    } else {
        return fail_v2(make_error(ErrorCode::InvalidArgument, "Expected an existing supported still or folder.", input.string()));
    }
    if (frames.empty()) return fail_v2(make_error(ErrorCode::NotFound, "No stills found.", input.string()));
    std::set<std::string> output_names;
    for (const auto& f : frames) {
        auto name = f.stem().string() + ".exr";
        // Refuse ASCII case aliases on every platform so a sequence is portable
        // between case-sensitive storage and the Windows/macOS application.
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (!output_names.insert(name).second)
            return fail_v2(make_error(ErrorCode::InvalidArgument, "Inputs map to the same EXR name; rename them before export.", f.string()));
    }

    auto m = aces_v2::read_manifest(pkg);
    if (!m) return fail_v2(m.error());
    if (auto v = aces_v2::verify_files(*m); !v) return fail_v2(v.error());
    if (smoothing && m->smoothing != "hybrid64")
        return fail_v2(make_error(ErrorCode::ContractMismatch, "This package declares no clipped smoothing; pass --no-smoothing."));
    auto view = aces_v2::make_ocio_view(threads);
    if (!view) return fail_v2(view.error());
    Result<aces_v2::G1Fn> g1 = make_error(ErrorCode::Unsupported, "Unknown runtime (libtorch or onnxruntime).", runtime);
    if (runtime == "libtorch") {
#ifdef RUDRA_HAVE_LIBTORCH_ACES
        g1 = aces_v2::load_g1_libtorch(m->g1, device);
#else
        g1 = make_error(ErrorCode::Unsupported, "This build has no LibTorch.");
#endif
    } else if (runtime == "onnxruntime") {
        if (m->g1_onnx.empty()) return fail_v2(make_error(ErrorCode::Unsupported, "This package has no ONNX model; re-export it."));
#ifdef RUDRA_HAVE_ORT_ACES
        g1 = aces_v2::load_g1_onnxruntime(m->g1_onnx, device);
#else
        g1 = make_error(ErrorCode::Unsupported, "This build has no ONNX Runtime.");
#endif
    }
    if (!g1) return fail_v2(g1.error());

    fs::create_directories(out);

    nlohmann::json run = {{"contract", aces_v2::kContractId}, {"package", pkg.string()}, {"g1_sha256", m->g1_sha256},
                          {"checkpoint_sha256", m->checkpoint_sha256}, {"model_contract", m->model_contract_id},
                          {"view", (*view)->identity()}, {"device", device}, {"runtime", runtime},
                          {"g1_file", runtime == "onnxruntime" ? m->g1_onnx.filename().string() : m->g1.filename().string()}, {"clipped_smoothing", smoothing ? m->smoothing : "none"},
                          {"appearance_constraint", m->forbid_new_negative_y ? "1 code, include fully clipped, no new negative luminance"
                                                                           : "1 code, include fully clipped"},
                          {"master", "ACEScg scene-linear, signed, unclamped, float32"},
                          {"mask", "rudra.reconstructed = highlight reconstruction weight; 0 = from source (incl. analytic shadows)"},
                          {"not", "not camera RAW; reconstructed highlights are estimates"},
                          {"release_authorized", m->release_authorized}, {"frames", nlohmann::json::array()}};
    bool all_ok = true;
    for (const auto& f : frames) {
        auto still = decode_sdr_file(f);
        if (!still) return fail_v2(still.error());
        const auto t0 = std::chrono::steady_clock::now();
        auto r = aces_v2::run_frame(still->rgb.buffer(), **view, *g1, smoothing, true, m->forbid_new_negative_y);
        if (!r) return fail_v2(r.error());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const fs::path exr = out / (f.stem().string() + ".exr");
        const ExrAttributes prov = {{"contract", aces_v2::kContractId}, {"g1_sha256", m->g1_sha256},
                                    {"source", f.filename().string()}, {"view", (*view)->identity()}};
        if (auto w = write_acescg_exr(exr, r->acescg, Primaries::Ap1, 1.0, prov, /*half=*/false, &r->mask); !w)
            return fail_v2(w.error());
        std::size_t reconstructed = 0;
        for (float v : r->mask.span()) reconstructed += v > 0;
        const bool ok = r->constraint.remaining_display_errors == 0 &&
                        (!m->forbid_new_negative_y || r->constraint.new_negative_luminance == 0);
        all_ok = all_ok && ok;
        run["frames"].push_back({{"source", f.string()}, {"exr", exr.filename().string()}, {"bits", still->bits},
                                 {"ms", ms}, {"reconstructed_fraction", double(reconstructed) / double(r->mask.plane_size())},
                                 {"fully_clipped_pixels", r->smoothing.full_pixels},
                                 {"smoothed_components", r->smoothing.components},
                                 {"constraint_support", r->constraint.support_pixels},
                                 {"constraint_changed", r->constraint.changed_pixels},
                                 {"constraint_baseline_fallback", r->constraint.baseline_fallback_pixels},
                                 {"remaining_display_errors", r->constraint.remaining_display_errors},
                                 {"new_negative_luminance", r->constraint.new_negative_luminance}});
        std::printf("%s -> %s  %.0f ms  remaining_display_errors=%zu\n", f.filename().string().c_str(),
                    exr.filename().string().c_str(), ms, r->constraint.remaining_display_errors);
    }
    run["appearance_constraint_pass"] = all_ok;
    std::ofstream(out / "provenance.json") << run.dump(1);
    return all_ok ? 0 : 3;
}
