#pragma once
// Transactional movie export for the experimental headroom policy (E2, 17 Oct 2026).
//
// One prepared, protected source decoded once by a single HeadroomStream owner;
// every frame graded by render_headroom_frame (the preview's renderer) and
// written as a float32 linear Rec.2020 EXR (1.0 = 203 nits) into a private
// staging folder beside the output. Only after the decoder's completion
// receipt and the source identity both validate are the frames encoded
// (deliver_linear_frames: Rec.2020, roll-off into the peak, measured
// MaxCLL/MaxFALL, colour tags verified). The staged movie is then probed (frame
// count, size) and its provenance sidecar written; only then are the two files
// renamed into place, never replacing an existing file. Any failure, including
// cancellation, leaves nothing at the output paths and removes the staging.
//
// The grade must pass validate_headroom_grade (anchor, chroma, grain, clip
// fill, calibration, reference and model all off). No temporal smoothing, no
// audio, no SDR conversion and no fallback to the legacy path. Windows only
// while source leases are (the source preparation refuses elsewhere).

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>

#include "rudra/core/measure.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct HeadroomExportRequest {
    std::filesystem::path source;    // the movie (v1 contract: limited-range Rec.709 YUV 4:4:4)
    std::filesystem::path output;    // the movie to write; its suffix becomes the target's
    std::string target = "hdr10";    // hdr10, hlg, prores422hq, prores4444
    double fps = 0.0;                // 0: the source's own rate
    double peak_nits = 1000.0;
    MasterRequest grade;             // the session's grade (validate_headroom_grade)
    std::filesystem::path decoder;   // FFmpeg; empty: resolved on PATH
    std::string app_version;         // recorded in the sidecar
};

struct HeadroomExportProgress {
    std::string phase;   // prepare, render, encode, qc, publish
    int done = 0, total = 0;
};

struct HeadroomExportResult {
    std::filesystem::path file, sidecar;
    int frames = 0, width = 0, height = 0;
    StaticMetadata metadata;
    std::string source_sha256;
};

// The sidecar beside the movie: <output>.json.
std::filesystem::path headroom_export_sidecar(const std::filesystem::path& file);

Result<HeadroomExportResult> export_headroom_movie(const HeadroomExportRequest& request,
    const std::function<void(const HeadroomExportProgress&)>& progress = {},
    const std::atomic<bool>* cancel = nullptr);

}  // namespace rudra
