#pragma once
// rudra/video.py convert_video in C++ (Phase 4, step 7): an SDR clip to an
// HDR10, HLG or ProRes master with its audio, checked before it is published
// and never replacing anything. The same checks in the same order with the
// same words, the same decode, predictor, mastering, spool, encoder command,
// QC and sidecar report, key for key.

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <utility>
#include <vector>
#include <string>

#include "rudra/core/model_manifest.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/deliver/shot_grades.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/platform/pyjson.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// video.py's arguments, with argparse's defaults. `package` stands where the
// Python takes --checkpoint: the model package exported from that checkpoint.
struct VideoConvertArgs {
    std::filesystem::path input, output, package;
    std::string format = "hdr10";
    std::optional<std::string> alpha_mode;
    std::string input_transfer = "auto", input_primaries = "auto", input_matrix = "auto", input_range = "auto";
    std::string audio = "copy";
    double peak_nits = 1000.0, min_nits = 0.005;
    std::optional<double> knee_nits;
    int crf = 12;
    std::string preset = "medium";
    int tile_size = 512, tile_overlap = 64;
    double shadow_smoothing = 0.0, cut_threshold = 0.15;
    // Seconds: the per-frame values (shadow weight, residual scale, curve, the
    // anchor's hold) held steady within a shot with this time constant
    // (10 Oct 2026); 0 turns it off. shadow_smoothing, when set, is the
    // retention instead, as before.
    double temporal_smoothing = 0.25;
    // "on": where the source pixel is still, the HDR's gain over it held the
    // same way (ShadowSmoother::hold_pixels); "off": per frame.
    std::string static_hold = "on";
    // "on": a PQ master gets its dynamic HDR metadata per shot beside it
    // (<name>_hdr10plus.json, _dovi_generate.json, _hdr_analysis.json,
    // _hdr10plus_scenes.json), and HDR10 carries HDR10+ when this ffmpeg's x265
    // can write it (video.py --dynamic-metadata, 10 Oct 2026).
    std::string dynamic_metadata = "on";
    std::optional<std::filesystem::path> work_dir;
    // The app's movie export (10 Oct 2026): the settings of the view and of
    // Master EXR (a master-params JSON, deliver/master.hpp) applied to every
    // frame through grade_master, instead of `rudra video`'s recovery "all"
    // and no grade. `grade.analytic`: no model runs, and none is needed.
    // Unset: the Python's path, unchanged.
    std::optional<std::filesystem::path> grade_file;
    std::optional<MasterRequest> grade;
    // Shot keyframes (roadmap 4.3, 20 Oct 2026): a grade per frame from its
    // shot's keys (deliver/shot_grades.hpp), with the holds reset at every
    // shot boundary as at a detected cut. Replaces `grade` when set; every
    // grade in it must agree on `analytic`. The report lists the shots, each
    // with its own flicker.
    std::optional<std::filesystem::path> shots_file;
    std::optional<ShotGrades> shots;
};

// progress(dict(phase=..., frames_done=..., frames_total=...)).
struct VideoProgress {
    std::string phase;   // inference, encoding, quality_check
    int frames_done = 0, frames_total = 0;
};

struct VideoConvertHooks {
    std::function<void(const VideoProgress&)> progress;
    std::function<void(const std::string&)> print;   // the lines the Python prints
    const std::atomic<bool>* cancel = nullptr;        // checked once per frame
};

struct VideoConvertResult {
    std::filesystem::path output, sidecar;
    pyjson::Value report;
};

// sidecar = output + ".json" (output.with_suffix(output.suffix + '.json')).
std::filesystem::path video_sidecar_path(const std::filesystem::path& output);

// The dynamic HDR metadata beside a PQ master, in the Python's order:
// hdr10plus, dovi_generate, analysis, hdr10plus_scenes (video.py dynamic_metadata_paths).
std::vector<std::pair<std::string, std::filesystem::path>> dynamic_metadata_paths(const std::filesystem::path& output);

// Whether this ffmpeg's x265 writes HDR10+ (built with ENABLE_HDR10_PLUS):
// one frame encoded with --dhdr10-info, x265's warning looked for. Cached per
// executable (video.py hdr10plus_capable).
bool hdr10plus_capable(const std::filesystem::path& ffmpeg);

#ifdef RUDRA_HAVE_STILL_DECODE
// Conform (roadmap 5.4): the hard cuts in a movie, by the export's own cut
// measure (ShadowSmoother::update on the 36x64 thumbnail, `threshold` the
// mean change that makes a cut), each the first frame of a new shot.
struct CutDetection {
    int frames = 0;
    std::string fps;          // ffprobe's, "24000/1001"
    std::vector<int> cuts;    // frames > 0, in order
};
Result<CutDetection> detect_cuts(const std::filesystem::path& movie, double threshold = 0.15,
                                 const std::atomic<bool>* cancel = nullptr);

Result<VideoConvertResult> convert_video(const VideoConvertArgs& args, const ModelManifest& package,
                                         InferenceBackend& backend, const VideoConvertHooks& hooks = {});
// The same with the package and backend optional: null for an analytic grade.
Result<VideoConvertResult> convert_video(const VideoConvertArgs& args, const ModelManifest* package,
                                         InferenceBackend* backend, const VideoConvertHooks& hooks = {});
#endif

// A master-params JSON file as a grade (its masks_file read too).
Result<MasterRequest> read_grade_file(const std::filesystem::path& path);

}  // namespace rudra
