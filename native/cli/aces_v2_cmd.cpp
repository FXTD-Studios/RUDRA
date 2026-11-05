// rudra-native aces-v2: the ACES v2 staged master (core/aces_v2.hpp) for
// stills, folders of stills and videos (every frame, <stem>_<NNNNNN>.exr): 32-bit float ACEScg EXR with the rudra.reconstructed mask, plus a
// JSON sidecar per frame and a run provenance file. Same contract as
// tools/rudra_aces_export.py --clipped-smoothing hybrid64 --colour-backtrack-all.

#include "aces_v2_g1.hpp"
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
#include "aces_v2_inputs.hpp"
#include "aces_v2_deliver.hpp"
#include "aces_v2_log.hpp"

namespace fs = std::filesystem;
using namespace rudra;

namespace {

int fail_v2(const Error& e) {
    std::fprintf(stderr, "error [%s]: %s\n  %s\n", to_string(e.code), e.message.c_str(), e.detail.c_str());
    return 2;
}

}  // namespace

int cmd_aces_v2(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::fprintf(stderr, "usage: rudra-native aces-v2 <package> <image|folder|video> --out <dir> [--runtime libtorch|onnxruntime] [--device cpu|cuda|directml|coreml] [--ocio-threads N] [--no-smoothing] [--luminance-guard|--no-luminance-guard] [--exr float|half|none] [--exr-gamut acescg|ap0] [--frame N]... [--fps F] [--name NAME [--start N]] [--padding N] [--into-existing [--no-create-dirs]] [--range FIRST:LAST] [--timecode HH:MM:SS:FF] [--reel NAME] [--source-display bt1886|srgb|gamma22] [--strength S] [--saturation S] [--peak-nits N] [--exposure EV] [--contrast C] [--log logc4|acescct|pq1000|pq2000|pq4000|hlg1000|sdr709] [--log-format tiff16,dpx10,prores4444xq|prores4444|prores422hq|mov16|hevc10]\n");
        return 64;
    }
    const fs::path pkg = args[0], input = args[1];
    fs::path out;
    std::string device = "cpu", runtime = "libtorch";
    int threads = 4;
    bool smoothing = true;
    int guard_override = -1;   // -1: the package's appearance_constraint.forbid_new_negative_y
    std::string log_curve, log_formats;
    nlohmann::json deliver = nlohmann::json::object();   // aces_v2_deliver.hpp, as aces-v2-serve takes it
    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--out" && i + 1 < args.size()) out = args[++i];
        else if (args[i] == "--device" && i + 1 < args.size()) device = args[++i];
        else if (args[i] == "--runtime" && i + 1 < args.size()) runtime = args[++i];
        else if (args[i] == "--ocio-threads" && i + 1 < args.size()) threads = std::atoi(args[++i].c_str());
        else if (args[i] == "--no-smoothing") smoothing = false;
        else if (args[i] == "--luminance-guard") guard_override = 1;
        else if (args[i] == "--no-luminance-guard") guard_override = 0;
        else if (args[i] == "--log" && i + 1 < args.size()) log_curve = args[++i];
        else if (args[i] == "--log-format" && i + 1 < args.size()) log_formats = args[++i];
        else if (args[i] == "--exr" && i + 1 < args.size()) deliver["exr"] = args[++i];
        else if (args[i] == "--exr-gamut" && i + 1 < args.size()) deliver["exr_gamut"] = args[++i];
        else if (args[i] == "--frame" && i + 1 < args.size()) deliver["frames"].push_back(std::atoi(args[++i].c_str()));
        else if (args[i] == "--fps" && i + 1 < args.size()) deliver["fps"] = std::atof(args[++i].c_str());
        else if (args[i] == "--name" && i + 1 < args.size()) deliver["name"] = args[++i];
        else if (args[i] == "--start" && i + 1 < args.size()) deliver["start"] = std::atoi(args[++i].c_str());
        else if (args[i] == "--range" && i + 1 < args.size()) {
            const std::string r = args[++i];
            const auto colon = r.find(':');
            deliver["range"] = {std::atoi(r.substr(0, colon).c_str()), colon == std::string::npos ? -1 : std::atoi(r.substr(colon + 1).c_str())};
        }
        else if (args[i] == "--timecode" && i + 1 < args.size()) deliver["timecode"] = args[++i];
        else if (args[i] == "--padding" && i + 1 < args.size()) deliver["padding"] = std::atoi(args[++i].c_str());
        else if (args[i] == "--into-existing") deliver["into_existing"] = true;
        else if (args[i] == "--no-create-dirs") deliver["create_dirs"] = false;
        else if (args[i] == "--reel" && i + 1 < args.size()) deliver["reel"] = args[++i];
        else if (args[i] == "--source-display" && i + 1 < args.size()) deliver["source_display"] = args[++i];
        else if (args[i] == "--strength" && i + 1 < args.size()) deliver["trims"]["strength"] = std::atof(args[++i].c_str());
        else if (args[i] == "--saturation" && i + 1 < args.size()) deliver["trims"]["saturation"] = std::atof(args[++i].c_str());
        else if (args[i] == "--peak-nits" && i + 1 < args.size()) deliver["trims"]["peak_nits"] = std::atof(args[++i].c_str());
        else if (args[i] == "--exposure" && i + 1 < args.size()) deliver["trims"]["exposure_ev"] = std::atof(args[++i].c_str());
        else if (args[i] == "--contrast" && i + 1 < args.size()) deliver["trims"]["contrast"] = std::atof(args[++i].c_str());
        else return fail_v2(make_error(ErrorCode::InvalidArgument, "Unknown option.", args[i]));
    }
    if (out.empty()) return fail_v2(make_error(ErrorCode::InvalidArgument, "--out <dir> is required."));

    // Validate the complete output mapping (and probe any video) before loading
    // a model or writing a master. Different inputs can otherwise overwrite the same EXR.
    const auto sources = aces_expand_input(input);
    if (sources.empty()) {
        if (fs::is_directory(input)) return fail_v2(make_error(ErrorCode::NotFound, "No stills found.", input.string()));
        return fail_v2(make_error(ErrorCode::InvalidArgument, "Expected an existing supported still, folder or video.", input.string()));
    }
    deliver["log"] = log_curve;
    deliver["log_formats"] = log_formats;
    auto opts = aces_deliver_from_json(deliver);
    if (!opts) return fail_v2(opts.error());
    auto frames = AcesFrameFeed::open(sources, opts->frames, fs::temp_directory_path() / "rudra_aces_decode.log", opts->range_first, opts->range_last);
    if (!frames) return fail_v2(frames.error());

    auto m = aces_v2::read_manifest(pkg);
    if (!m) return fail_v2(m.error());
    if (auto v = aces_v2::verify_files(*m); !v) return fail_v2(v.error());
    if (smoothing && m->smoothing != "hybrid64")
        return fail_v2(make_error(ErrorCode::ContractMismatch, "This package declares no clipped smoothing; pass --no-smoothing."));
    auto view = aces_v2::make_ocio_view(threads);
    if (!view) return fail_v2(view.error());
    std::unique_ptr<aces_v2::AcesView> hdr_view;   // the ACES 2.0 HDR output: pq1000 copies and the HDR10 light level
    {
        auto hv = aces_v2::make_ocio_display_view(aces_v2::kOcioHdrDisplay, aces_v2::kOcioHdrView, threads);
        if (!hv) return fail_v2(hv.error());
        hdr_view = std::move(*hv);
    }
    // Another display target (pq2000, pq4000, hlg1000, sdr709): its own view for the copies.
    std::unique_ptr<aces_v2::AcesView> target_view;
    if (opts->copies.pq && opts->copies.target != "pq1000") {
        auto tv = aces_target_view(opts->copies.display_target(), threads);
        if (!tv) return fail_v2(tv.error());
        target_view = std::move(*tv);
    }
    const aces_v2::AcesView* copy_view = target_view ? target_view.get() : hdr_view.get();
    const aces_v2::AcesView* light_view = opts->copies.hdr10() && target_view ? target_view.get() : hdr_view.get();
    // The runtime asked for, or the one this build has (cli/aces_v2_g1.hpp).
    auto chosen = aces_v2::load_g1_choice(*m, runtime, device);
    if (!chosen) return fail_v2(chosen.error());
    if (chosen->runtime != runtime || chosen->device != device)
        std::fprintf(stderr, "this build runs the model on %s (%s)\n", chosen->device.c_str(), chosen->runtime.c_str());
    runtime = chosen->runtime;
    device = chosen->device;
    Result<aces_v2::G1Fn> g1 = std::move(chosen->fn);

    const bool guard = guard_override < 0 ? m->forbid_new_negative_y : guard_override == 1;
    const std::string movie_stem = (fs::is_directory(input) ? input.filename() : input.stem()).string();
    if (auto ready = aces_prepare_out(*opts, out, (*frames)->total(),
                                      opts->copies.any() && opts->copies.movie() ? aces_movie_file(*opts, movie_stem) : "");
        !ready)
        return fail_v2(ready.error());

    nlohmann::json run = {{"contract", aces_v2::kContractId}, {"package", pkg.string()}, {"g1_sha256", m->g1_sha256},
                          {"checkpoint_sha256", m->checkpoint_sha256}, {"model_contract", m->model_contract_id},
                          {"view", (*view)->identity()}, {"device", device}, {"runtime", runtime},
                          {"g1_file", runtime == "onnxruntime" ? m->g1_onnx.filename().string() : m->g1.filename().string()}, {"clipped_smoothing", smoothing ? m->smoothing : "none"},
                          {"appearance_constraint", guard ? "1 code, include fully clipped, no new negative luminance"
                                                           : "1 code, include fully clipped"},
                          {"luminance_guard", guard}, {"luminance_guard_package_default", m->forbid_new_negative_y},
                          {"master", aces_exr_label(*opts)}, {"trims", aces_trims_json(opts->trims)},
                          {"source_display", aces_source_display_label(opts->source_display)},
                          {"mask", "rudra.reconstructed = highlight reconstruction weight; 0 = from source (incl. analytic shadows)"},
                          {"not", "not camera RAW; reconstructed highlights are estimates"},
                          {"release_authorized", m->release_authorized}, {"frames", nlohmann::json::array()}};
    bool all_ok = true;
    std::unique_ptr<AcesLogWriter> logw;
    if (!opts->name.empty()) run["name"] = opts->name, run["start"] = opts->start;
    if (opts->copies.any()) {
        auto w = AcesLogWriter::open(opts->copies, out, out / aces_movie_file(*opts, movie_stem),
                                     opts->fps > 0 ? opts->fps : (*frames)->fps(), copy_view);
        if (!w) return fail_v2(w.error());
        logw = std::move(*w);
    }
    const double run_fps = opts->fps > 0 ? opts->fps : (*frames)->fps();
    double run_maxcll = 0, run_maxfall = 0, run_scene_peak = 0;
    std::vector<FrameStats> run_stats;   // the PQ output's, for Dolby Vision
    int run_w = 0, run_h = 0;
    for (std::size_t n = 0; n < (*frames)->total(); ++n) {
        auto f = (*frames)->next();
        if (!f) return fail_v2(f.error());
        aces_adapt_source(f->codes, opts->source_display);   // an sRGB or 2.2 source, as BT.1886 codes
        const auto t0 = std::chrono::steady_clock::now();
        auto r = aces_v2::run_frame(f->codes, **view, *g1, smoothing, true, guard);
        if (!r) return fail_v2(r.error());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // The colorist's trims, then the frame's timecode and reel (28 Oct 2026).
        const PlanarBuffer master = aces_trimmed(*opts, r->acescg, r->mask, **view, f->codes);
        const AcesFrameStamp stamp = aces_frame_stamp(*opts, n, *f, run_fps);
        const AcesLightLevel light = aces_light_level(master, light_view, int(n));
        if (opts->copies.hdr10()) run_stats.push_back(light.stats);
        run_w = master.width(), run_h = master.height();
        run_maxcll = std::max(run_maxcll, light.maxcll);
        run_maxfall = std::max(run_maxfall, light.maxfall);
        run_scene_peak = std::max(run_scene_peak, light.scene_peak);
        if (logw && n == 0) logw->set_movie_stamp(stamp.timecode, stamp.reel);
        const std::string stem = aces_frame_stem(*opts, n, f->exr);
        const fs::path exr = out / (stem + ".exr");
        const ExrAttributes prov = {{"contract", aces_v2::kContractId}, {"g1_sha256", m->g1_sha256},
                                    {"source", f->source}, {"view", (*view)->identity()}};
        if (opts->writes_exr())
            if (auto w = write_aces_master_exr(exr, master, r->mask, *opts, prov,
                                                          exr_timecode_attributes(stamp.timecode, stamp.fps, stamp.reel)); !w) return fail_v2(w.error());
        if (logw)
            if (auto lw = logw->write(master, stem + ".exr", stamp.timecode); !lw) return fail_v2(lw.error());
        std::size_t reconstructed = 0;
        for (float v : r->mask.span()) reconstructed += v > 0;
        const bool ok = r->constraint.remaining_display_errors == 0 &&
                        (!guard || r->constraint.new_negative_luminance == 0);
        all_ok = all_ok && ok;
        run["frames"].push_back({{"source", f->source_path}, {"frame", f->source},
                                 {"exr", opts->writes_exr() ? exr.filename().string() : std::string()}, {"name", stem}, {"timecode", stamp.timecode}, {"reel", stamp.reel}, {"maxcll", light.maxcll}, {"maxfall", light.maxfall}, {"scene_peak_nits", light.scene_peak}, {"bits", f->bits},
                                 {"ms", ms}, {"reconstructed_fraction", double(reconstructed) / double(r->mask.plane_size())},
                                 {"fully_clipped_pixels", r->smoothing.full_pixels},
                                 {"smoothed_components", r->smoothing.components},
                                 {"constraint_support", r->constraint.support_pixels},
                                 {"constraint_changed", r->constraint.changed_pixels},
                                 {"constraint_baseline_fallback", r->constraint.baseline_fallback_pixels},
                                 {"remaining_display_errors", r->constraint.remaining_display_errors},
                                 {"new_negative_luminance", r->constraint.new_negative_luminance}});
        std::printf("%s -> %s  %.0f ms  remaining_display_errors=%zu\n", f->source.c_str(),
                    (opts->writes_exr() ? exr.filename().string() : stem).c_str(), ms, r->constraint.remaining_display_errors);
        std::fflush(stdout);   // one line per frame reaches a pipe (the app's progress) as it happens
    }
    if (logw) {
        if (auto fin = logw->finish(run_maxcll, run_maxfall); !fin) return fail_v2(fin.error());
        run["log_delivery"] = logw->provenance();
    }
    if (opts->copies.any() && opts->copies.hdr10()) {   // the HDR10 numbers beside the PQ files
        run["hdr10"] = aces_hdr10_json(run_maxcll, run_maxfall, opts->copies.display_target().peak_nits);
        if (auto hw = aces_write_whole(aces_hdr10_file(*opts, out), run["hdr10"].dump(1)); !hw) return fail_v2(hw.error());
        run["hdr10_file"] = aces_hdr10_file(*opts, out).filename().string();
        if (!run_stats.empty()) {   // Dolby Vision CM v4.0 of the same PQ output
            auto dv = aces_dolby_vision(run_stats, opts->copies.display_target().peak_nits,
                                        opts->name.empty() ? out.filename().string() : opts->name, run_fps, run_w, run_h);
            if (!dv) return fail_v2(dv.error());
            if (auto a = aces_write_whole(aces_dovi_xml_file(*opts, out), dv->xml); !a) return fail_v2(a.error());
            if (auto b = aces_write_whole(aces_dovi_generate_file(*opts, out), dv->generate_json); !b) return fail_v2(b.error());
            run["dolby_vision"] = {{"xml", aces_dovi_xml_file(*opts, out).filename().string()}, {"shots", dv->shots}};
        }
    }
    run["maxcll"] = run_maxcll;
    run["maxfall"] = run_maxfall;
    run["scene_peak_nits"] = run_scene_peak;
    run["light_level"] = kAcesLightLevelNote;
    run["appearance_constraint_pass"] = all_ok;
    if (auto pw = aces_write_whole(aces_provenance_file(*opts, out), run.dump(1)); !pw) return fail_v2(pw.error());
    return all_ok ? 0 : 3;
}
