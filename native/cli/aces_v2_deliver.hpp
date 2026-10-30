#pragma once
// What an ACES v2 master writes (27 Oct 2026), shared by `aces-v2` and
// `aces-v2-serve` so the app's Deliver tab and the command line agree:
//   exr        "float" (32-bit, the default), "half" (16-bit) or "none"
//   exr_gamut  "acescg" (AP1, the default) or "ap0" (ACES 2065-1)
//   log / log_formats   the encoded copies (aces_v2_log.hpp): DPX, TIFF, ProRes, 16-bit movie
//   frames     only these frames of one video (a still from a movie); none: every frame
//   range      [first, last]: only these frames of one video, in order (a shot's in and out)
//   fps        the movie's rate; 0: the video's own, else 24
//   name       "": every file keeps its source's name; else <name>.<NNNNNN> from `start`
//   timecode   the start timecode ("01:00:00:00"); without it a movie carries
//              its own and stills have none (28 Oct 2026)
//   reel       the reel name (default: the movie's reel_name tag, else its file name)
//   trims      {"strength", "saturation", "peak_nits"} (core/aces_v2_trims.hpp)
//   padding    digits of the frame number with a name (default 6; "####" is 4)
//   separator  between the name and the number: "." (default), "_" or ""
//   into_existing  write into `out` as it is (Nuke's Write): it may exist already,
//              every file is checked first and nothing is replaced; with
//              create_dirs (default true) a missing folder is made, else refused.
//              The copies (DPX, TIFF) go beside the EXRs ("flat") and the run's
//              provenance is <name>_provenance.json (29 Oct 2026)

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "aces_v2_inputs.hpp"
#include "aces_v2_log.hpp"
#include "rudra/core/aces_v2_trims.hpp"
#include "rudra/core/measure.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct AcesDeliverOptions {
    std::string exr = "float";
    std::string exr_gamut = "acescg";
    AcesLogOptions copies;
    std::vector<int> frames;
    int range_first = -1, range_last = -1;
    double fps = 0;
    std::string name;
    int start = 0;
    std::string timecode, reel;
    aces_v2::AcesTrims trims;
    int padding = 6;
    std::string separator = ".";   // between the name and the frame number: ".", "_" or ""
    bool into_existing = false, create_dirs = true;
    // The display the SDR was made for (28 Oct 2026 review): "bt1886" (Rec.709, 2.4: what
    // the model reads), "srgb" (piecewise sRGB) or "gamma22". Others are re-encoded to
    // BT.1886 code values first, so the same display light reaches the model.
    std::string source_display = "bt1886";
    bool writes_exr() const { return exr != "none"; }
};

// Re-encodes display codes made for `source_display` as BT.1886 (2.4) codes, same
// display light; bt1886 leaves them as they are. False for an unknown name.
bool aces_valid_source_display(const std::string& s);
void aces_adapt_source(PlanarBuffer& codes, const std::string& source_display);
// The provenance words of the source display.
std::string aces_source_display_label(const std::string& s);

// From a serve request (or the command line's flags gathered as JSON). Refuses
// unknown values, a bad name, and a run that would write nothing.
Result<AcesDeliverOptions> aces_deliver_from_json(const nlohmann::json& req);

// The file stem of frame `i` (0-based in the run): <name>.<start + i, 6 digits>,
// else the source's own EXR name without ".exr".
std::string aces_frame_stem(const AcesDeliverOptions& o, std::size_t i, const std::string& exr_name);
// Into an existing folder (into_existing): make it when allowed, then refuse if
// any file this run writes is already there. `total` frames, `movie` the movie's file name ("" for none).
Result<void> aces_prepare_out(const AcesDeliverOptions& o, const std::filesystem::path& out, std::size_t total,
                              const std::string& movie);
// Where the run's provenance goes: out/provenance.json, or out/<name>_provenance.json into an existing folder.
std::filesystem::path aces_provenance_file(const AcesDeliverOptions& o, const std::filesystem::path& out);
// The HDR10 sidecar of a render with PQ copies: out/hdr10.json or out/<name>_hdr10.json (28 Oct 2026).
std::filesystem::path aces_hdr10_file(const AcesDeliverOptions& o, const std::filesystem::path& out);
nlohmann::json aces_hdr10_json(double maxcll, double maxfall, double peak_nits = 1000);
// The OCIO view of a display target (aces_v2_log.hpp), threads as the engine's
// (inline: only the engine, which links OpenColorIO, calls it).
inline Result<std::unique_ptr<aces_v2::AcesView>> aces_target_view(const AcesDisplayTarget& t, int threads) {
    return aces_v2::make_ocio_display_view(t.display, t.view, threads, true, t.config);
}
// Written beside as "<file>.part" and renamed when whole (28 Oct 2026 review): a
// crash or a kill never leaves a short file under a finished name.
Result<void> aces_write_whole(const std::filesystem::path& file, const std::string& bytes);
Result<void> aces_finish_part(const std::filesystem::path& part, const std::filesystem::path& file);
std::filesystem::path aces_part_file(const std::filesystem::path& file);
// An ACES 2065-1 frame's reconstruction mask: <folder>/mask/<frame>.exr (ST 2065-4 holds RGB only).
std::filesystem::path aces_mask_file(const std::filesystem::path& exr);
// What the movie is called: <name>.mov, else <stem>_<encoding>.mov.
std::string aces_movie_file(const AcesDeliverOptions& o, const std::string& stem);
// What provenance.json says the EXR is ("ACEScg scene-linear, signed, unclamped, float32", ...).
std::string aces_exr_label(const AcesDeliverOptions& o);
// The EXR in the chosen gamut and depth, with the rudra.reconstructed mask.
Result<void> write_aces_master_exr(const std::filesystem::path& path, const PlanarBuffer& acescg, const PlanarBuffer& mask,
                                   const AcesDeliverOptions& o, const ExrAttributes& provenance,
                                   const ExrTypedAttributes& typed = {});

// Frame `i` of the run: its timecode (`timecode` + i, or + the movie's frame
// number, at the rate; else the movie's own; "" when neither) and reel
// (`reel`, else the movie's, else "").
struct AcesFrameStamp {
    std::string timecode, reel;
    double fps = 0;
};
AcesFrameStamp aces_frame_stamp(const AcesDeliverOptions& o, std::size_t i, const AcesFrameIn& f, double run_fps);
// The master after the trims (the model's own when there are none); the
// baseline is the plain inverse of `codes` through `view`.
PlanarBuffer aces_trimmed(const AcesDeliverOptions& o, const PlanarBuffer& acescg, const PlanarBuffer& mask,
                          const aces_v2::AcesView& view, const PlanarBuffer& codes);
// A frame's light level (28 Oct 2026): MaxCLL is the brightest pixel's largest
// Rec.2020 component and MaxFALL the frame's mean of that, in nits at diffuse
// white = 203 nits, scene-referred (before any output transform).
// HDR10 light level (28 Oct 2026 review). maxcll / maxfall: CTA-861.3 on the
// deliverable, the ACES 2.0 HDR 1000 nits (P3 D65) PQ output in display nits
// (max of R, G, B per pixel; MaxFALL the frame's mean of it). Without the HDR
// view they are 0. scene_peak / scene_average: the scene-linear master in
// Rec.2020 x 203 nits, before any output transform (a grading readout, never metadata).
struct AcesLightLevel {
    double maxcll = 0, maxfall = 0;
    double scene_peak = 0, scene_average = 0;
    FrameStats stats;   // the PQ output's frame statistics in display nits (Dolby Vision L1), with hdr_pq
};
AcesLightLevel aces_light_level(const PlanarBuffer& acescg, const aces_v2::AcesView* hdr_pq = nullptr, int index = 0);

// Dolby Vision CM v4.0 of a learned PQ render (28 Oct 2026 review): the dovi_tool
// generate config (profile 8.1, L1 per shot from `stats`, L6) and the metadata XML
// Resolve and the Dolby tools import. Shots by the analytic route's cut detection.
struct AcesDolbyVision {
    std::string generate_json, xml;
    int shots = 0;
};
Result<AcesDolbyVision> aces_dolby_vision(const std::vector<FrameStats>& stats, double peak_nits, const std::string& name,
                                          double fps, int width, int height);
// out/dolby_vision.xml (+ _dovi_generate.json), or out/<name>_dolby_vision.xml into an existing folder.
std::filesystem::path aces_dovi_xml_file(const AcesDeliverOptions& o, const std::filesystem::path& out);
std::filesystem::path aces_dovi_generate_file(const AcesDeliverOptions& o, const std::filesystem::path& out);
inline constexpr const char* kAcesLightLevelNote =
    "maxcll/maxfall: HDR10 (CTA-861.3) of the ACES 2.0 HDR 1000 nits (P3 D65) PQ output, display nits; "
    "scene_peak_nits/scene_average_nits: scene-linear Rec.2020 x 203 before any output transform";
// The trims from a request's "trims" object (absent: none); refuses out-of-range values.
Result<aces_v2::AcesTrims> aces_trims_from_json(const nlohmann::json& trims);
nlohmann::json aces_trims_json(const aces_v2::AcesTrims& t);

// The frames a master reads: every frame of the sources (AcesInputs), or only
// `frames` of one video (each read on its own, as the preview does).
class AcesFrameFeed {
public:
    static Result<std::unique_ptr<AcesFrameFeed>> open(const std::vector<std::filesystem::path>& sources,
                                                       const std::vector<int>& frames,
                                                       const std::filesystem::path& decode_log, int range_first = -1,
                                                       int range_last = -1);
    std::size_t total() const noexcept;
    double fps() const noexcept { return fps_; }
    Result<AcesFrameIn> next();

private:
    AcesFrameFeed() = default;
    std::unique_ptr<AcesInputs> all_;
    std::filesystem::path video_, log_;
    std::vector<int> frames_;
    std::size_t at_ = 0;
    double fps_ = 24.0;
};

}  // namespace rudra
