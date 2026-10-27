#pragma once
// Camera-log copies of an ACES v2 master (27 Oct 2026): ARRI LogC4 in ARRI Wide
// Gamut 4 (or ACEScct in AP1) from the same ACEScg pixels the EXR carries
// (scene-linear, 18% grey = 0.18), no peak or knee, no exposure change.
//   tiff16        16-bit TIFF per frame (lossless), log_<curve>/<frame>.tif
//   dpx10         10-bit DPX per frame (RGB, 10 bits a channel), log_<curve>_dpx/<frame>.dpx
//   prores4444xq  ProRes 4444 XQ (12-bit 4:4:4), one .mov for the run
//   prores4444    ProRes 4444 (12-bit 4:4:4), one .mov for the run
//   prores422hq   ProRes 422 HQ (10-bit 4:2:2), one .mov for the run
// ProRes is tagged with unknown primaries and transfer (there are no codes for
// LogC4 / AWG4), BT.709 YCbCr matrix, limited range, as the app's log ProRes.
// Codes outside [0, 1] are clamped (LogC4 holds up to 469.8 x diffuse white).
// The rudra.reconstructed mask stays in the EXR.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rudra/core/image.hpp"
#include "rudra/core/log_encode.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

class Process;

struct AcesLogOptions {
    std::optional<LogCurve> curve;   // none: EXR only
    bool tiff16 = false, dpx10 = false;
    std::string prores;              // "", "4444xq", "4444" or "422hq"
    bool any() const { return curve && (tiff16 || dpx10 || !prores.empty()); }
};

// "logc4" or "acescct"; formats comma-separated ("tiff16,dpx10,prores4444xq").
// An empty curve is no log copy at all.
Result<AcesLogOptions> parse_aces_log(const std::string& curve, const std::string& formats);

// ACEScg (AP1 linear) to the curve's code values in its own gamut, clamped to [0, 1].
PlanarBuffer aces_log_codes(const PlanarBuffer& acescg, LogCurve curve);

class AcesLogWriter {
public:
    // `movie` is the .mov path (used when ProRes is asked for); fps its rate.
    static Result<std::unique_ptr<AcesLogWriter>> open(const AcesLogOptions& o, const std::filesystem::path& out_dir,
                                                       const std::filesystem::path& movie, double fps);
    ~AcesLogWriter();
    // One frame in order; `name` is the frame's EXR file name.
    Result<void> write(const PlanarBuffer& acescg, const std::string& name);
    // Closes the movie and waits for ffmpeg; call once after the last frame.
    Result<void> finish();
    nlohmann::json provenance() const;

private:
    AcesLogWriter() = default;
    AcesLogOptions o_;
    std::filesystem::path dir_, dpx_dir_, movie_, log_;
    double fps_ = 24.0;
    int width_ = 0, height_ = 0, frames_ = 0;
    std::unique_ptr<Process> ffmpeg_;
};

}  // namespace rudra
