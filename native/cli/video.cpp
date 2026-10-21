#include "video.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>

#include <fstream>
#include <iterator>

#include "rudra/core/model_manifest.hpp"
#include "rudra/deliver/conform.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/video/convert.hpp"

namespace rudra {
namespace fs = std::filesystem;

namespace {

int usage_error(const std::string& message) {
    std::fprintf(stderr, "rudra-native video: error: %s\n", message.c_str());
    return 64;
}

bool one_of(const std::string& v, const std::set<std::string>& choices) { return choices.count(v) > 0; }

}  // namespace

int cmd_video(const std::vector<std::string>& args) {
    // video.py add_arguments, with the package in the checkpoint's place.
    // --grade <settings.json> (native only, 11 Oct 2026): the app's movie export
    // from the command line, the master's chain with those settings; an
    // analytic one needs no package ("-" in its place).
    if (args.size() < 2)
        return usage_error("usage: rudra-native video <package|-> <input> --output <file> [--grade <settings.json>] [--shots <shots.json>] [options]");
    VideoConvertArgs a;
    a.package = args[0];
    a.input = args[1];
    std::string runtime = "auto", device = "cpu";
    bool have_output = false;
    for (std::size_t i = 2; i < args.size(); ++i) {
        const std::string& k = args[i];
        if (i + 1 >= args.size()) return usage_error("argument " + k + ": expected one argument");
        const std::string v = args[++i];
        auto number = [&](double& out) {
            char* end = nullptr;
            out = std::strtod(v.c_str(), &end);
            return end && *end == '\0' && !v.empty();
        };
        auto integer = [&](int& out) {
            char* end = nullptr;
            const long l = std::strtol(v.c_str(), &end, 10);
            out = int(l);
            return end && *end == '\0' && !v.empty();
        };
        double d = 0;
        if (k == "--output") a.output = v, have_output = true;
        else if (k == "--format") {
            if (!one_of(v, {"hdr10", "hlg", "prores422", "prores422hq", "prores4444"})) return usage_error("invalid --format: " + v);
            a.format = v;
        } else if (k == "--alpha-mode") {
            if (v != "straight") return usage_error("invalid --alpha-mode: " + v);
            a.alpha_mode = v;
        } else if (k == "--input-transfer") {
            if (!one_of(v, {"auto", "srgb", "rec709", "gamma22", "gamma24"})) return usage_error("invalid --input-transfer: " + v);
            a.input_transfer = v;
        } else if (k == "--input-primaries") {
            if (!one_of(v, {"auto", "rec709", "rec2020"})) return usage_error("invalid --input-primaries: " + v);
            a.input_primaries = v;
        } else if (k == "--input-matrix") {
            if (!one_of(v, {"auto", "bt709", "bt2020nc", "gbr"})) return usage_error("invalid --input-matrix: " + v);
            a.input_matrix = v;
        } else if (k == "--input-range") {
            if (!one_of(v, {"auto", "full", "limited"})) return usage_error("invalid --input-range: " + v);
            a.input_range = v;
        } else if (k == "--audio") {
            if (!one_of(v, {"copy", "aac", "none"})) return usage_error("invalid --audio: " + v);
            a.audio = v;
        } else if (k == "--peak-nits") { if (!number(a.peak_nits)) return usage_error("invalid --peak-nits: " + v); }
        else if (k == "--min-nits") { if (!number(a.min_nits)) return usage_error("invalid --min-nits: " + v); }
        else if (k == "--knee-nits") { if (!number(d)) return usage_error("invalid --knee-nits: " + v); a.knee_nits = d; }
        else if (k == "--crf") { if (!integer(a.crf)) return usage_error("invalid --crf: " + v); }
        else if (k == "--preset") {
            if (!one_of(v, {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}))
                return usage_error("invalid --preset: " + v);
            a.preset = v;
        } else if (k == "--tile-size") { if (!integer(a.tile_size)) return usage_error("invalid --tile-size: " + v); }
        else if (k == "--tile-overlap") { if (!integer(a.tile_overlap)) return usage_error("invalid --tile-overlap: " + v); }
        else if (k == "--shadow-smoothing") { if (!number(a.shadow_smoothing)) return usage_error("invalid --shadow-smoothing: " + v); }
        else if (k == "--cut-threshold") { if (!number(a.cut_threshold)) return usage_error("invalid --cut-threshold: " + v); }
        else if (k == "--temporal-smoothing") { if (!number(a.temporal_smoothing)) return usage_error("invalid --temporal-smoothing: " + v); }
        else if (k == "--dynamic-metadata") {
            if (!one_of(v, {"on", "off"})) return usage_error("invalid --dynamic-metadata: " + v);
            a.dynamic_metadata = v;
        } else if (k == "--static-hold") {
            if (!one_of(v, {"on", "off"})) return usage_error("invalid --static-hold: " + v);
            a.static_hold = v;
        }
        else if (k == "--work-dir") a.work_dir = fs::path(v);
        else if (k == "--grade") a.grade_file = fs::path(v);
        else if (k == "--shots") a.shots_file = fs::path(v);   // shot keyframes (roadmap 4.3)
        else if (k == "--runtime") runtime = v;
        else if (k == "--device") device = v;
        else return usage_error("unrecognized argument: " + k);
    }
    if (!have_output) return usage_error("the following arguments are required: --output");

    auto fail = [](const Error& e) {
        std::fprintf(stderr, "error [%s]: %s\n", to_string(e.code), e.message.c_str());
        if (!e.detail.empty()) std::fprintf(stderr, "  %s\n", e.detail.c_str());
        return 2;
    };
    VideoConvertHooks hooks;
    hooks.print = [](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
    };
    if (a.shots_file) {
        auto s = load_shot_grades(*a.shots_file);
        if (!s) return fail(s.error());
        a.shots = std::move(*s);
        if (a.shots->base.analytic) {   // every grade in it is analytic (convert_video checks): no model
            auto r = convert_video(a, nullptr, nullptr, hooks);
            if (!r) return fail(r.error());
            return 0;
        }
    } else if (a.grade_file) {
        auto g = read_grade_file(*a.grade_file);
        if (!g) return fail(g.error());
        a.grade = *g;
        if (g->analytic) {   // no model runs, none is opened
            auto r = convert_video(a, nullptr, nullptr, hooks);
            if (!r) return fail(r.error());
            return 0;
        }
    }
    if (a.package == "-") return usage_error("this export runs the model: give its package");
    auto m = read_manifest(a.package);
    if (!m) return fail(m.error());
    const auto dev = device_from_string(device);
    if (!dev) return usage_error("invalid --device: " + device);
    Result<std::unique_ptr<InferenceBackend>> b = make_error(ErrorCode::Unsupported, "No runtime");
    if (runtime == "libtorch") b = make_libtorch_backend(*m, *dev);
    else if (runtime == "onnxruntime") b = make_onnxruntime_backend(*m, *dev);
    else {
        for (auto r : compiled_runtimes())
            if (r == Runtime::LibTorch) b = make_libtorch_backend(*m, *dev);
        if (!b) b = make_onnxruntime_backend(*m, *dev);
    }
    if (!b) return fail(b.error());
    auto r = convert_video(a, *m, **b, hooks);
    if (!r) return fail(r.error());
    return 0;
}

int cmd_conform(const std::vector<std::string>& args) {
    auto err = [](const std::string& m) {
        std::fprintf(stderr, "rudra-native conform: error: %s\n", m.c_str());
        return 64;
    };
    auto fail = [](const Error& e) {
        std::fprintf(stderr, "error [%s]: %s\n", to_string(e.code), e.message.c_str());
        if (!e.detail.empty()) std::fprintf(stderr, "  %s\n", e.detail.c_str());
        return 2;
    };
    if (args.empty())
        return err("usage: rudra-native conform <movie> (--edl FILE | --fcpxml FILE | --detect) --output <shots.json> "
                   "[--base <settings.json>] [--record-start TC] [--fps RATE] [--threshold X] [--key-each-shot]");
    const fs::path movie = args[0];
    std::optional<fs::path> edl, fcpxml, output, base_file;
    std::optional<std::string> record_start, fps_text;
    bool detect = false, key_each = false;
    double threshold = 0.15;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& k = args[i];
        if (k == "--detect") { detect = true; continue; }
        if (k == "--key-each-shot") { key_each = true; continue; }
        if (i + 1 >= args.size()) return err("argument " + k + ": expected one argument");
        const std::string& v = args[++i];
        if (k == "--edl") edl = fs::path(v);
        else if (k == "--fcpxml") fcpxml = fs::path(v);
        else if (k == "--output") output = fs::path(v);
        else if (k == "--base") base_file = fs::path(v);
        else if (k == "--record-start") record_start = v;
        else if (k == "--fps") fps_text = v;
        else if (k == "--threshold") threshold = std::strtod(v.c_str(), nullptr);
        else return err("unrecognized argument: " + k);
    }
    if (int(bool(edl)) + int(bool(fcpxml)) + int(detect) != 1) return err("give one of --edl, --fcpxml or --detect");
    if (!output) return err("the following arguments are required: --output");
    std::error_code ec;
    if (fs::exists(*output, ec)) return fail(make_error(ErrorCode::InvalidArgument, "Refusing to overwrite the shots file", output->string()));
    MasterRequest base;
    if (base_file) {
        auto g = read_grade_file(*base_file);
        if (!g) return fail(g.error());
        base = *g;
    }
    ShotGrades shots;
    std::vector<std::string> notes;
    if (detect) {
        auto cuts = detect_cuts(movie, threshold);
        if (!cuts) return fail(cuts.error());
        shots = shot_grades_from_cuts(cuts->cuts, cuts->frames, base);   // named sh010, sh020 ..., source "detected"
    } else {
        auto probe = probe_video(movie);
        if (!probe) return fail(probe.error());
        auto video = open_video_source(*probe, VideoArgs{});
        if (!video) return fail(video.error());
        auto rate = parse_frame_rate(fps_text ? *fps_text : video->clock.fps);
        if (!rate) return fail(rate.error());
        const fs::path file = edl ? *edl : *fcpxml;
        std::ifstream in(file, std::ios::binary);
        if (!in) return fail(make_error(ErrorCode::NotFound, "The edit file is missing", file.string()));
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto edit = edl ? parse_edl(text, *rate) : parse_fcpxml(text, *rate);
        if (!edit) return fail(edit.error());
        std::optional<long long> start;
        if (record_start) {
            auto f = timecode_frames(*record_start, *rate, edit->drop_frame);   // in the edit's timecode
            if (!f) return fail(f.error());
            start = *f;
        }
        auto r = conform_shots(*edit, video->clock.frames, base, start);
        if (!r) return fail(r.error());
        shots = std::move(r->shots);
        notes = std::move(r->notes);
    }
    if (key_each)   // each shot carries its own copy of the base, to be graded on its own
        for (auto& g : shots.shots) g.keys = {GradeKey{g.first, base}};
    if (auto w = save_shot_grades(*output, shots); !w) return fail(w.error());
    for (const auto& n : notes) std::printf("note: %s\n", n.c_str());
    std::printf("%zu shots -> %s\n", shots.shots.size(), output->string().c_str());
    return 0;
}

}  // namespace rudra
