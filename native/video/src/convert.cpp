#include "rudra/video/convert.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <mutex>
#include <map>

#include "rudra/core/hdr10.hpp"
#include "rudra/core/log_encode.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/deliver/sidecars.hpp"
#include "rudra/core/metadata.hpp"
#include "rudra/deliver/video_encode.hpp"
#include "rudra/deliver/video_master.hpp"
#include "rudra/media/png16.hpp"
#include "rudra/media/video_decode.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/video/predictor.hpp"
#include "rudra/video/qc.hpp"

namespace rudra {
namespace fs = std::filesystem;

fs::path video_sidecar_path(const fs::path& output) {
    fs::path s = output;
    s += ".json";
    return s;
}

Result<MasterRequest> read_grade_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The export's settings file is missing", path.string());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return master_request_from_json(text);
}


#ifdef RUDRA_HAVE_STILL_DECODE

namespace {

Error refuse(std::string message) { return make_error(ErrorCode::InvalidArgument, std::move(message)); }

std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

fs::path resolved(const fs::path& p) {
    std::error_code ec;
    fs::path r = fs::weakly_canonical(fs::absolute(p, ec), ec);
    return ec ? fs::absolute(p) : r;
}

}  // namespace

std::vector<std::pair<std::string, fs::path>> dynamic_metadata_paths(const fs::path& output) {
    // Named after the whole output name, as the report is: clip.mp4 and clip.mov never share them.
    const std::string base = output.filename().string();
    const fs::path dir = output.parent_path();
    return {{"hdr10plus", dir / (base + "_hdr10plus.json")},
            {"dovi_generate", dir / (base + "_dovi_generate.json")},
            {"analysis", dir / (base + "_hdr_analysis.json")},
            {"hdr10plus_scenes", dir / (base + "_hdr10plus_scenes.json")}};
}

bool hdr10plus_capable(const fs::path& ffmpeg) {
    static std::mutex mu;
    static std::map<std::string, bool> cache;
    std::lock_guard lock(mu);
    if (auto it = cache.find(ffmpeg.string()); it != cache.end()) return it->second;
    bool capable = false;
    auto work = StagingDir::create(fs::temp_directory_path(), "rudra-hdr10plus-");
    if (!work) return false;   // not cached: the next export asks again
    {
        const fs::path probe = work->path() / "probe.json";
        std::ofstream(probe, std::ios::binary)
            << R"({"JSONInfo": {"HDR10plusProfile": "A", "Version": "1.0"}, "SceneInfo": [{"LuminanceParameters": )"
               R"({"AverageRGB": 1000, "LuminanceDistributions": {"DistributionIndex": [1, 5, 10, 25, 50, 75, 90, 95, 99], )"
               R"("DistributionValues": [0, 10, 20, 50, 100, 500, 1000, 2000, 5000]}, "MaxScl": [5000, 5000, 5000]}, )"
               R"("NumberOfWindows": 1, "TargetedSystemDisplayMaximumLuminance": 0, "SceneFrameIndex": 0, "SceneId": 0, )"
               R"("SequenceFrameIndex": 0}]})";
        auto r = run_process({ffmpeg.string(), "-hide_banner", "-v", "info", "-nostdin", "-f", "lavfi", "-i",
                              "color=c=black:s=64x64:r=24:d=0.05", "-frames:v", "1", "-c:v", "libx265", "-pix_fmt",
                              "yuv420p10le", "-x265-params", "log-level=warning:dhdr10-info=" + x265_param_path(probe),
                              "-f", "null", "-"});
        capable = r && r->exit_code == 0 && r->err.find("dhdr10-info disabled") == std::string::npos;
    }
    cache[ffmpeg.string()] = capable;
    return capable;
}

namespace {
// Whether the master's first frame carries HDR10+ (ST 2094-40) metadata (video.py has_hdr10plus).
Result<bool> has_hdr10plus(const fs::path& path) {
    auto ffprobe = require_executable("ffprobe");
    if (!ffprobe) return ffprobe.error();
    auto out = run_tool({ffprobe->string(), "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
                         "-show_frames", "-show_entries", "frame=side_data_list", "-of", "json", path.string()});
    if (!out) return out.error();
    return out->find("2094-40") != std::string::npos;
}
}  // namespace

Result<CutDetection> detect_cuts(const fs::path& movie, double threshold, const std::atomic<bool>* cancel) {
    if (!(0 < threshold && threshold <= 1)) return refuse("Invalid cut threshold");
    const fs::path source = resolved(movie);
    std::error_code ec;
    if (!fs::is_regular_file(source, ec)) return make_error(ErrorCode::NotFound, source.string());
    auto probe = probe_video(source);
    if (!probe) return probe.error();
    auto video = open_video_source(*probe, VideoArgs{});
    if (!video) return video.error();
    auto work = StagingDir::create(fs::temp_directory_path(), "rudra-cuts-");
    if (!work) return work.error();
    auto decoder = VideoDecoder::open(source, *video, work->path() / "decode.log");
    if (!decoder) return decoder.error();
    CutDetection out;
    out.frames = video->clock.frames;
    out.fps = video->clock.fps;
    ShadowSmoother smoother(0.0, threshold, false);
    const auto& c = video->contract;
    for (int index = 0; index < out.frames; ++index) {
        if (cancel && cancel->load()) return make_error(ErrorCode::Cancelled, "Cancelled");
        auto frame = (*decoder)->next();
        if (!frame) return frame.error();
        // The thumbnail VideoPredictor::prepare makes: canonical code, Rec.2020, 36x64.
        auto canon = canonicalize_sdr(rgb_from_frame(*frame).buffer(), c.transfer, "full");
        if (!canon) return canon.error();
        PlanarBuffer x = std::move(*canon);
        if (c.primaries != "rec2020") x = srgb_code_to_rec2020(x, Primaries::Rec709);
        const auto [w, cut] = smoother.update(area_resize(x, kCutThumbHeight, kCutThumbWidth), 1.0);
        (void)w;
        if (cut && index > 0) out.cuts.push_back(index);
    }
    if (auto done = (*decoder)->finish(); !done) return done.error();
    return out;
}

Result<VideoConvertResult> convert_video(const VideoConvertArgs& a, const ModelManifest& package,
                                         InferenceBackend& backend, const VideoConvertHooks& hooks) {
    return convert_video(a, &package, &backend, hooks);
}

Result<VideoConvertResult> convert_video(const VideoConvertArgs& a, const ModelManifest* package_ptr,
                                         InferenceBackend* backend, const VideoConvertHooks& hooks) {
    // Shot keyframes: every grade in the file runs the same reconstruction.
    std::optional<bool> shots_analytic;
    if (a.shots) {
        shots_analytic = a.shots->base.analytic;
        for (const auto& g : a.shots->shots)
            for (const auto& k : g.keys)
                if (k.grade.analytic != *shots_analytic)
                    return make_error(ErrorCode::InvalidArgument,
                                      "Every grade in the shots file must use the model, or none may",
                                      g.name + " at frame " + std::to_string(k.frame));
    }
    const bool graded = a.grade.has_value() || a.shots.has_value();
    const bool analytic = shots_analytic ? *shots_analytic : (a.grade && a.grade->analytic);
    if (!analytic && (!package_ptr || !backend))
        return make_error(ErrorCode::InvalidArgument, "This export runs the model, and no model package is open");
    static const ModelManifest kNoPackage{};
    const ModelManifest& package = package_ptr ? *package_ptr : kNoPackage;
    const fs::path source = resolved(a.input), output = resolved(a.output);
    const fs::path sidecar = video_sidecar_path(output);
    const std::string ext = lower(output.extension().string());
    if (ext != ".mp4" && ext != ".mov" && ext != ".mkv") return refuse("Output must be MP4, MOV or MKV");
    const DeliveryProfile* profile = find_delivery_profile(a.format);
    if (!profile) return refuse("Unknown delivery format: " + a.format);
    if (profile->codec == "prores" && ext != ".mov") return refuse("ProRes delivery requires a .mov output");
    std::error_code ec;
    if (!fs::is_regular_file(source, ec)) return make_error(ErrorCode::NotFound, source.string());
    if (a.dynamic_metadata != "on" && a.dynamic_metadata != "off") return refuse("Invalid dynamic metadata setting");
    // PQ masters carry dynamic HDR metadata (10 Oct 2026); HLG and log masters have none to carry.
    const bool dynamic = a.dynamic_metadata == "on" && profile->transfer == "smpte2084" && profile->log.empty();
    const auto dynamic_paths = dynamic ? dynamic_metadata_paths(output) : decltype(dynamic_metadata_paths(output)){};
    const auto any_dynamic_exists = [&] {
        std::error_code e;
        for (const auto& [k, p] : dynamic_paths)
            if (fs::exists(p, e)) return true;
        return false;
    };
    if (source == output || fs::exists(output, ec) || fs::exists(sidecar, ec) || any_dynamic_exists())
        return refuse("Refusing to overwrite source/output/sidecar");
    auto ffmpeg = require_executable("ffmpeg");
    if (!ffmpeg) return ffmpeg.error();
    auto encoders = run_tool({ffmpeg->string(), "-hide_banner", "-encoders"});
    if (!encoders) return encoders.error();
    auto filters = run_tool({ffmpeg->string(), "-hide_banner", "-filters"});
    if (!filters) return filters.error();
    if (encoders->find(profile->encoder) == std::string::npos || filters->find("zscale") == std::string::npos)
        return make_error(ErrorCode::Unsupported, "This FFmpeg build needs " + profile->encoder + " and zscale support");
    if (!(0 <= a.shadow_smoothing && a.shadow_smoothing < 1) || !(0 < a.cut_threshold && a.cut_threshold <= 1) ||
        !(0 <= a.temporal_smoothing && a.temporal_smoothing <= 10) || (a.static_hold != "on" && a.static_hold != "off"))
        return refuse("Invalid smoothing/cut threshold");
    if (a.tile_size < 0 || a.tile_overlap < 0 || (a.tile_size && a.tile_overlap >= a.tile_size))
        return refuse("Invalid tile size/overlap");
    if (!(0 <= a.crf && a.crf <= 51) || !(0 <= a.min_nits && a.min_nits < a.peak_nits))
        return refuse("Invalid mastering/encoding settings");
    if (auto m = master_to_pq(PlanarBuffer(3, 1, 1, 0.0f), a.peak_nits, a.knee_nits); !m) return m.error();

    VideoArgs va;
    va.format = a.format;
    va.alpha_mode = a.alpha_mode;
    va.input_transfer = a.input_transfer, va.input_primaries = a.input_primaries;
    va.input_matrix = a.input_matrix, va.input_range = a.input_range;
    auto probe = probe_video(source);
    if (!probe) return probe.error();
    auto video = open_video_source(*probe, va);
    if (!video) return video.error();
    const bool alpha = video->alpha;
    const int width = video->stream.width, height = video->stream.height, total = video->clock.frames;
    if (a.shots)
        if (auto v = validate_shot_grades(*a.shots, total); !v) return v.error();

    fs::create_directories(output.parent_path(), ec);
    const fs::path work_parent = resolved(a.work_dir ? *a.work_dir : output.parent_path());
    fs::create_directories(work_parent, ec);
    const int channels = alpha ? 4 : 3;
    const std::uint64_t frame_bytes = std::uint64_t(width) * height * channels * 2;
    if (spool_space_low(work_parent, frame_bytes)) return make_error(ErrorCode::IoError, "Insufficient working disk space");
    VideoPredictor predictor(backend, ModelConstants{package.log_scale, package.max_hdr, package.corpus_ev}, a.tile_size,
                             a.tile_overlap);
    auto predict = [&](const SdrImage& rgb, const std::string& transfer, const std::string& primaries,
                       ShadowSmoother& sm, int index) {
        if (a.shots) return predictor.predict_graded(rgb, transfer, primaries, sm, grade_at(*a.shots, index));
        return a.grade ? predictor.predict_graded(rgb, transfer, primaries, sm, *a.grade)
                       : predictor.predict(rgb, transfer, primaries, sm);
    };
    const double retention =
        a.shadow_smoothing != 0.0 ? a.shadow_smoothing : retention_for(video->clock.fps, a.temporal_smoothing);
    ShadowSmoother smoother(retention, a.cut_threshold, a.static_hold == "on");
    const auto started = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };

    const auto& c = video->contract;
    const auto& clk = video->clock;
    pyjson::Dict report = {
        {"source", source.string()},
        {"output", output.string()},
        {"input_contract", pyjson::Dict{{"transfer", c.transfer}, {"primaries", c.primaries}, {"matrix", c.matrix},
                                        {"range", c.range}}},
        {"timing", pyjson::Dict{{"fps", clk.fps}, {"frames", clk.frames}, {"start", clk.start},
                                {"duration", clk.duration}, {"tolerance", clk.tolerance}}},
        {"delivery_format", a.format},
        {"alpha_mode", alpha && a.alpha_mode ? pyjson::Value(*a.alpha_mode) : pyjson::Value()},
        {"encoder_input_precision_bits", 10},
        {"alpha_preserved", alpha},
        {"checkpoint", analytic ? pyjson::Value() : pyjson::Value(resolved(package.root).string())},
        {"checkpoint_sha256", analytic ? pyjson::Value() : pyjson::Value(package.source_sha256)},
        {"mastering_peak_nits", a.peak_nits},
        {"mastering_min_nits", a.format == "hdr10" ? pyjson::Value(a.min_nits) : pyjson::Value()},
        {"audio_mode", a.audio},
        {"shadow_smoothing", a.shadow_smoothing},
        {"cut_threshold", a.cut_threshold},
        {"temporal_smoothing", a.temporal_smoothing},
        {"static_hold", a.static_hold},
        {"dynamic_metadata", a.dynamic_metadata},
        {"preserve_outside", a.shots ? a.shots->base.preserve_outside : a.grade ? a.grade->preserve_outside : true},
        {"recovery_mode", a.shots ? a.shots->base.recovery_mode : a.grade ? a.grade->recovery_mode : std::string("all")},
        {"tile_size", a.tile_size},
        {"tile_overlap", a.tile_overlap},
        {"timestamp_policy", "Video begins at zero; audio retains relative offset, clipped to video interval"},
        {"scene_cuts", pyjson::List{}},
        {"frames", pyjson::List{}},
        {"warnings", a.shadow_smoothing != 0.0 ? pyjson::List{"Scalar shadow smoothing is not validated temporal reconstruction"}
                                        : pyjson::List{}},
    };
    if (graded) {
        // What made the picture: the app's settings, the file they came from.
        const MasterRequest& g = a.shots ? a.shots->base : *a.grade;
        pyjson::set(report, "reconstruction", analytic ? "analytic (no model)" : "model");
        pyjson::set(report, "grade_file", a.grade_file ? pyjson::Value(resolved(*a.grade_file).string()) : pyjson::Value());
        pyjson::set(report, "residual_strength", g.strength);
        pyjson::set(report, "source_curve", g.source_curve);
        pyjson::set(report, "anchored", g.anchor);
    }
    if (a.shots)
        pyjson::set(report, "shots_file", a.shots_file ? pyjson::Value(resolved(*a.shots_file).string()) : pyjson::Value());
    // Per shot: the frames scored for flicker and their sum and peak.
    struct ShotFlicker { int scored = 0; double sum = 0.0, peak = 0.0; };
    std::vector<ShotFlicker> shot_flicker(a.shots ? a.shots->shots.size() : 0);
    int current_shot = -2;
    pyjson::List cuts, frames;
    std::vector<VideoFrameLight> lights;
    std::optional<std::pair<std::vector<double>, std::vector<double>>> previous_exposure;   // (source, HDR) cells, EV
    std::vector<std::pair<int, double>> scored;
    std::vector<FrameStats> stats;   // the dynamic HDR metadata's, per frame
    std::vector<double> local_scored;

    auto spool_dir = StagingDir::create(work_parent, "rudra-video-");
    if (!spool_dir) return spool_dir.error();
    const fs::path spool = spool_dir->path();
    {
        auto decoder = VideoDecoder::open(source, *video, spool / "decode.log");
        if (!decoder) return decoder.error();
        for (int index = 0; index < total; ++index) {
            if (hooks.cancel && hooks.cancel->load()) return make_error(ErrorCode::Cancelled, "Cancelled");
            auto frame = (*decoder)->next();
            if (!frame) return frame.error();
            const SdrImage rgb = rgb_from_frame(*frame);
            // A shot boundary resets the holds as a detected cut does, and is not scored.
            bool shot_start = false;
            if (a.shots) {
                const int shot = shot_index(*a.shots, index);
                if (shot != current_shot) {
                    shot_start = current_shot != -2;
                    if (shot_start) smoother = ShadowSmoother(retention, a.cut_threshold, a.static_hold == "on");
                    current_shot = shot;
                }
            }
            auto pred = predict(rgb, c.transfer, c.primaries, smoother, index);
            if (!pred) return pred.error();
            // The flicker measure: how far the HDR's exposure moved beyond the source's.
            // The static hold (ShadowSmoother::hold_pixels), then the measure.
            const std::vector<double> linear = source_luminance(rgb.buffer(), c.transfer, c.primaries);
            if (smoother.static_hold()) {
                auto [luma, peak] = source_code_luma(rgb.buffer(), c.primaries);
                pred->hdr = NetworkLinearImage(
                    smoother.hold_pixels(std::move(luma), std::move(peak), linear, pred->hdr.buffer()));
            }
            std::pair<std::vector<double>, std::vector<double>> exposure{
                exposure_cells(linear, height, width), exposure_cells(hdr_luminance(pred->hdr.buffer()), height, width)};
            pyjson::Value flicker, local;
            if (!pred->cut && !shot_start && previous_exposure) {
                auto mean = [](const std::vector<double>& v) {
                    double s = 0.0;
                    for (double x : v) s += x;
                    return s / double(v.size());
                };
                const double f = std::fabs((mean(exposure.second) - mean(previous_exposure->second)) -
                                           (mean(exposure.first) - mean(previous_exposure->first)));
                std::vector<double> d(exposure.first.size());
                for (std::size_t k = 0; k < d.size(); ++k)
                    d[k] = std::fabs((exposure.second[k] - previous_exposure->second[k]) -
                                     (exposure.first[k] - previous_exposure->first[k]));
                flicker = f;
                local = mean(d);
                scored.emplace_back(index, f);
                local_scored.push_back(mean(d));
                if (current_shot >= 0) {
                    auto& sf = shot_flicker[std::size_t(current_shot)];
                    ++sf.scored, sf.sum += f, sf.peak = std::max(sf.peak, f);
                }
            }
            previous_exposure = std::move(exposure);
            auto sf = master_video_frame(pred->hdr, index, a.format, a.peak_nits, a.knee_nits,
                                         alpha ? std::span<const std::uint16_t>(frame->data) : std::span<const std::uint16_t>(),
                                         dynamic);
            if (!sf) return sf.error();
            if (sf->stats) stats.push_back(*sf->stats);
            frames.push_back(pyjson::Dict{{"index", index}, {"shadow_weight", pred->shadow_weight},
                                          {"max_cll", sf->light.max_cll}, {"frame_average", sf->light.frame_average},
                                          {"flicker_ev", flicker}, {"local_flicker_ev", local}});
            lights.push_back(sf->light);
            if (pred->cut) cuts.push_back(index);
            if (spool_space_low(spool, frame_bytes)) return make_error(ErrorCode::IoError, "Working disk is full");
            if (auto w = write_png16(spool / spool_frame_name(index), sf->packed, width, height, sf->channels); !w)
                return w.error();
            if (index % 10 == 0 || index + 1 == total) {
                if (hooks.progress) hooks.progress({"inference", index + 1, total});
                if (hooks.print) {
                    char line[96];
                    std::snprintf(line, sizeof line, "Frame %d/%d, %.2f fps", index + 1, total,
                                  (index + 1) / std::max(elapsed(), 0.001));
                    hooks.print(line);
                }
            }
        }
        if (auto done = (*decoder)->finish(); !done) return done.error();
    }
    pyjson::set(report, "scene_cuts", cuts);
    pyjson::set(report, "frames", frames);
    if (a.shots) {
        pyjson::List shots;
        for (std::size_t i = 0; i < a.shots->shots.size(); ++i) {
            const auto& g = a.shots->shots[i];
            const auto& sf = shot_flicker[i];
            shots.push_back(pyjson::Dict{
                {"name", g.name}, {"first", g.first}, {"last", g.last}, {"source", g.source},
                {"keys", int(g.keys.size())}, {"frames_scored", sf.scored},
                {"flicker_ev_mean", sf.scored ? pyjson::Value(sf.sum / sf.scored) : pyjson::Value()},
                {"flicker_ev_max", sf.scored ? pyjson::Value(sf.peak) : pyjson::Value()}});
        }
        pyjson::set(report, "shots", shots);
    }
    {
        pyjson::Value mean, peak, peak_frame, local_mean, local_peak;
        if (!scored.empty()) {
            double sum = 0.0, top = -1.0;
            int at = 0;
            for (const auto& [i, f] : scored) {
                sum += f;
                if (f > top) top = f, at = i;
            }
            mean = sum / double(scored.size()), peak = top, peak_frame = at;
            double lsum = 0.0, ltop = -1.0;
            for (double f : local_scored) lsum += f, ltop = std::max(ltop, f);
            local_mean = lsum / double(local_scored.size()), local_peak = ltop;
        }
        pyjson::set(report, "temporal",
                    pyjson::Dict{{"retention", retention}, {"static_hold", a.static_hold},
                                 {"frames_scored", int(scored.size())}, {"flicker_ev_mean", mean},
                                 {"flicker_ev_max", peak}, {"flicker_ev_max_frame", peak_frame},
                                 {"local_flicker_ev_mean", local_mean}, {"local_flicker_ev_max", local_peak}});
    }
    const LightCeilings ceil = light_ceilings(lights);

    auto staging = StagingDir::create(output.parent_path(), "rudra-master-");
    if (!staging) return staging.error();
    const fs::path staged = staging->path() / output.filename();
    VideoEncodeRequest er;
    er.format = a.format, er.preset = a.preset, er.crf = a.crf, er.audio = a.audio;
    er.peak_nits = a.peak_nits, er.min_nits = a.min_nits, er.alpha = alpha;
    // The dynamic HDR metadata, per shot of the export's own cuts (so it changes
    // where the hold restarts), written beside the staged master.
    std::vector<std::pair<fs::path, fs::path>> staged_dynamic;   // (staged, published)
    bool embed = false;
    if (dynamic) {
        std::vector<int> cut_frames;
        for (const auto& cut : cuts) cut_frames.push_back(int(std::get<std::int64_t>(cut.v)));
        if (a.shots)   // the keyframed shots' boundaries are cuts too (roadmap 4.3)
            for (const auto& g : a.shots->shots) cut_frames.push_back(g.first);
        std::sort(cut_frames.begin(), cut_frames.end());
        cut_frames.erase(std::unique(cut_frames.begin(), cut_frames.end()), cut_frames.end());
        const std::vector<Shot> shots = shots_from_cuts(cut_frames, int(stats.size()));
        auto written = write_all_sidecars(stats, staged, a.peak_nits, 0.35, shots, a.format == "hdr10" ? a.min_nits : 0.0001);
        if (!written) return written.error();
        const fs::path h10 = staging->path() / dynamic_paths[0].second.filename();
        if (auto w = write_text_file(h10, pyjson::dumps(to_hdr10plus_x265_json(stats, shots))); !w) return w.error();
        for (const auto& [k, p] : dynamic_paths) staged_dynamic.emplace_back(staging->path() / p.filename(), p);
        embed = a.format == "hdr10" && hdr10plus_capable(*ffmpeg);
        if (embed) er.hdr10plus = h10;
        pyjson::Value reason;
        if (!embed)
            reason = a.format != "hdr10" ? "HDR10+ is carried in HEVC only"
                                         : "This ffmpeg's x265 was built without HDR10+; inject the file with hdr10plus_tool";
        pyjson::set(report, "hdr_metadata",
                    pyjson::Dict{{"shots", int(shots.size())},
                                 {"hdr10plus", pyjson::Dict{{"file", dynamic_paths[0].second.filename().string()},
                                                            {"embedded", embed}, {"reason", reason}}},
                                 {"dovi_generate", dynamic_paths[1].second.filename().string()},
                                 {"analysis", dynamic_paths[2].second.filename().string()},
                                 {"hdr10plus_scenes", dynamic_paths[3].second.filename().string()}});
    } else {
        pyjson::set(report, "hdr_metadata", pyjson::Value());
    }
    auto command = encode_command(ffmpeg->string(), er, clk, spool, source, staged, ceil.max_cll, ceil.max_fall);
    if (!command) return command.error();
    pyjson::List cmd;
    for (const auto& s : *command) cmd.push_back(s);
    pyjson::set(report, "encode_command", cmd);
    if (hooks.progress) hooks.progress({"encoding", total, total});
    if (auto ran = run_tool(*command); !ran) return ran.error();
    if (hooks.progress) hooks.progress({"quality_check", total, total});
    VideoQcRequest qr;
    qr.audio_mode = a.audio, qr.format = a.format, qr.alpha = alpha;
    qr.expected = ExpectedLight{double(ceil.max_cll), double(ceil.max_fall), a.peak_nits, a.min_nits};
    auto qc = run_quality_check(video->streams_json, clk, staged, qr);
    if (!qc) return qc.error();
    if (alpha) {
        auto al = run_alpha_check(staged, spool, total, width, height);
        if (!al) return al.error();
        pyjson::set(*std::get<std::shared_ptr<pyjson::Dict>>(qc->v), "alpha", *al);
    }
    if (embed) {
        auto present = has_hdr10plus(staged);
        if (!present) return present.error();
        if (!*present)
            return make_error(ErrorCode::IoError, "HDR10+ QC failed: the master carries no ST 2094-40 metadata");
        pyjson::set(*std::get<std::shared_ptr<pyjson::Dict>>(qc->v), "hdr10plus", pyjson::Dict{{"passed", true}});
    }
    pyjson::set(report, "qc", *qc);
    if (auto curve = log_curve_from_name(profile->log))
        pyjson::set(report, "log_encoding",
                    pyjson::Dict{{"curve", log_curve_label(*curve)}, {"gamut", log_gamut_label(*curve)},
                                 {"scene_referred", true},
                                 {"exposure_convention", "0.18 = 18% grey (scene), diffuse white = 1.0"},
                                 {"note", "Peak and knee are not applied. Tagged unknown primaries and transfer: "
                                          "set the clip's input to this curve and gamut."}});
    if (a.format == "hlg")
        pyjson::set(report, "hlg_reference",
                    pyjson::Dict{{"display_peak_nits", a.peak_nits}, {"black_nits", 0},
                                 {"system_gamma", 1.2 + 0.42 * std::log10(a.peak_nits / 1000)}});
    pyjson::set(report, "max_cll", ceil.max_cll);
    pyjson::set(report, "max_fall", ceil.max_fall);
    pyjson::set(report, "elapsed_seconds", elapsed());
    const pyjson::Value rep(report);
    const fs::path staged_json = staging->path() / sidecar.filename();
    {
        std::ofstream out(staged_json, std::ios::binary);
        out << pyjson::dumps(rep, 2);
        if (!out) return make_error(ErrorCode::IoError, "Could not write the report", staged_json.string());
    }
    if (staged_dynamic.empty()) {
        if (auto p = publish_video(staged, staged_json, output, sidecar); !p) return p.error();
    } else {
        // Every name checked before anything is published; the master first, so
        // a failure after it leaves an output a retry refuses, never metadata
        // without a master; the report last (video.py's order).
        std::error_code e;
        if (fs::exists(output, e) || fs::exists(sidecar, e) || any_dynamic_exists())
            return make_error(ErrorCode::InvalidArgument, "Output appeared during processing; refusing overwrite");
        fs::rename(staged, output, e);
        if (e) return make_error(ErrorCode::IoError, "Could not publish the master", e.message());
        for (const auto& [from, to] : staged_dynamic) {
            fs::rename(from, to, e);
            if (e) return make_error(ErrorCode::IoError, "Could not publish the HDR metadata", e.message());
        }
        fs::rename(staged_json, sidecar, e);
        if (e) return make_error(ErrorCode::IoError, "Could not publish the sidecar", e.message());
    }
    if (hooks.print) hooks.print("QC passed: " + output.string() + "\nReport: " + sidecar.string());
    return VideoConvertResult{output, sidecar, rep};
}

#endif

}  // namespace rudra
