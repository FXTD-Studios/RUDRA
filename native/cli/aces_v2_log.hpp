#pragma once
// Encoded copies of an ACES v2 master (27 Oct 2026), from the same ACEScg
// pixels the EXR carries (scene-linear, 18% grey = 0.18), no exposure change:
//   logc4     ARRI LogC4 in ARRI Wide Gamut 4 (scene-referred, no peak or knee)
//   acescct   ACEScct in AP1 (scene-referred, no peak or knee)
//   pq1000    the ACES 2.0 output transform, HDR 1000 nits (P3 D65) on a
//             Rec.2100 PQ display (display-referred; needs the OCIO view)
//   pq2000, pq4000  the same at 2000 / 4000 nits (ACES studio config)
//   hlg1000   ACES 2.0 HDR 1000 nits (P3 D65) on a Rec.2100-HLG display (studio config)
//   sdr709    ACES 2.0 SDR 100 nits (Rec.709) on BT.1886: the SDR trim pass of the same master
// (the display-referred targets, 28 Oct 2026 review)
// in these files:
//   tiff16         16-bit TIFF per frame (lossless), <dir>/<frame>.tif
//   dpx10          10-bit DPX per frame (RGB, 10 bits a channel), <dir>_dpx/<frame>.dpx
//   prores4444xq   ProRes 4444 XQ (12-bit 4:4:4), one .mov for the run
//   prores4444     ProRes 4444 (12-bit 4:4:4), one .mov for the run
//   prores422hq    ProRes 422 HQ (10-bit 4:2:2), one .mov for the run
//   mov16          uncompressed 16-bit RGB QuickTime ('b48r', lossless), one .mov
//   hevc10         HDR10 HEVC Main 10 (pq1000 only), one .mp4: a ProRes 4444 XQ
//                  intermediate, then x265 with the run's measured MaxCLL / MaxFALL
//                  and the P3 D65 1000-nit mastering display in its SEI (28 Oct 2026)
// Movies are written as <name>.part.<ext> and renamed when whole.
// <dir> is log_<curve> (or pq1000). Log ProRes is tagged with unknown primaries
// and transfer (there are no codes for LogC4 / AWG4), BT.709 YCbCr matrix,
// limited range, as the app's log ProRes; PQ is tagged BT.2020 / SMPTE 2084.
// The 16-bit movie is RGB, full range, tagged the same way. Codes outside
// [0, 1] are clamped (LogC4 holds up to 469.8 x diffuse white). The
// rudra.reconstructed mask stays in the EXR.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rudra/core/aces_v2.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/log_encode.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

class Process;

// A display-referred output of the ACES 2.0 transforms.
struct AcesDisplayTarget {
    const char* id;          // "pq1000", "pq2000", "pq4000", "hlg1000", "sdr709"
    const char* config;      // the OCIO built-in config
    const char* display;
    const char* view;
    const char* transfer;    // "pq", "hlg" or "sdr"
    double peak_nits;
    const char* label;       // "PQ (SMPTE ST 2084), 1000 nits"
};
// nullptr for an unknown id.
const AcesDisplayTarget* aces_display_target(const std::string& id);

struct AcesLogOptions {
    std::optional<LogCurve> curve;   // a camera log, or
    bool pq = false;                 // a display-referred ACES 2.0 output (`target`); neither: EXR only
    std::string target = "pq1000";   // which (aces_display_target), when pq
    const AcesDisplayTarget& display_target() const { return *aces_display_target(target); }
    bool hdr10() const { return pq && std::string(display_target().transfer) == "pq"; }
    bool tiff16 = false, dpx10 = false;
    std::string prores;              // "", "4444xq", "4444" or "422hq"
    bool mov16 = false;              // uncompressed 16-bit RGB QuickTime
    bool hevc10 = false;             // HDR10 HEVC Main 10 (.mp4), PQ only
    bool flat = false;               // TIFF / DPX beside the EXRs in the output folder, not in sub-folders
    bool encoded() const { return curve.has_value() || pq; }
    bool movie() const { return !prores.empty() || mov16 || hevc10; }
    std::string movie_extension() const { return hevc10 ? ".mp4" : ".mov"; }
    bool any() const { return encoded() && (tiff16 || dpx10 || movie()); }
    // "logc4", "acescct" or the target (what parse_aces_log takes); "" for none.
    std::string name() const;
};

// "logc4", "acescct" or "pq1000"; formats comma-separated
// ("tiff16,dpx10,prores4444xq", "mov16"; one movie at most). An empty curve is no copy at all.
Result<AcesLogOptions> parse_aces_log(const std::string& curve, const std::string& formats);

// ACEScg (AP1 linear) to the curve's code values in its own gamut, clamped to [0, 1].
PlanarBuffer aces_log_codes(const PlanarBuffer& acescg, LogCurve curve);

// A written DPX's header filled in for the conform (28 Oct 2026 review): the SMPTE
// timecode (TV header, BCD), the frame rate (film and TV headers), the reel (as the
// input device name, which Resolve and Baselight show), and transfer / colorimetric
// (6 = BT.709 for the SDR pass; 0 = user-defined for a log, PQ or HLG, named in
// the provenance). Either byte order, as the file's magic says.
Result<void> stamp_dpx_header(const std::filesystem::path& dpx, const std::string& timecode, double fps,
                              const std::string& reel, int transfer, int colorimetric);

class AcesLogWriter {
public:
    // `movie` is the .mov path (used when a movie is asked for); fps its rate.
    // `display` is the ACES 2.0 HDR view (ACEScg -> PQ codes), needed for pq1000 and kept, not owned.
    static Result<std::unique_ptr<AcesLogWriter>> open(const AcesLogOptions& o, const std::filesystem::path& out_dir,
                                                       const std::filesystem::path& movie, double fps,
                                                       const aces_v2::AcesView* display = nullptr);
    ~AcesLogWriter();
    // One frame in order; `name` is the frame's file name (its stem names the copies).
    // `timecode` (the frame's own, "" for none) goes into the DPX's TV header with the rate and the reel.
    Result<void> write(const PlanarBuffer& acescg, const std::string& name, const std::string& timecode = {});
    // The movie's start timecode and reel (before the first frame; "" for none).
    void set_movie_stamp(const std::string& timecode, const std::string& reel) {
        timecode_ = timecode;
        reel_ = reel;
    }
    // Closes the movie and waits for ffmpeg; call once after the last frame.
    // The run's HDR10 light level goes into an HEVC's SEI (0: none measured).
    Result<void> finish(double maxcll = 0, double maxfall = 0);
    nlohmann::json provenance() const;

    // The code values each file holds, in [0, 1]: the log curve in its gamut, or PQ.
    PlanarBuffer codes(const PlanarBuffer& acescg) const;
    // The ffmpeg arguments of the movie (from "-c:v" on), for these options.
    static std::vector<std::string> movie_arguments(const AcesLogOptions& o);
    // The x265 HDR10 arguments of the HEVC pass (P3 D65 1000-nit mastering display).
    static std::vector<std::string> hevc_arguments(double maxcll, double maxfall, double peak_nits = 1000);

private:
    AcesLogWriter() = default;
    AcesLogOptions o_;
    const aces_v2::AcesView* display_ = nullptr;
    std::filesystem::path dir_, dpx_dir_, movie_, log_;
    std::filesystem::path part_;     // what ffmpeg writes: renamed to movie_ when whole
    double maxcll_ = 0, maxfall_ = 0;
    std::string timecode_, reel_;
    double fps_ = 24.0;
    int width_ = 0, height_ = 0, frames_ = 0;
    std::unique_ptr<Process> ffmpeg_;
};

}  // namespace rudra
