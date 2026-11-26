#pragma once
// The OFX plug-in's picture (roadmap 5.1, 20 Oct 2026): one SDR frame from the
// host to scene-linear or PQ HDR through RUDRA's own master chain, the
// analytic reconstruction (no model: no GPU, no package, nothing to install
// beside the plug-in). The same grade_master the app's Master EXR and movie
// exports run, so a frame from Resolve or Nuke is the frame RUDRA masters.
//
// No OFX here: the plug-in (ofx/plugin.cpp) hands this the host's pixels, and
// the tests call it directly.
//
// In: RGBA float, the SDR as display code in [0, 1] (the host's Rec.709
// timeline, Gamma 2.4 / BT.1886 or sRGB), Rec.709 primaries; values outside
// [0, 1] are clamped. Out: RGBA float in one of
//   linear_rec2020   scene-linear Rec.2020, 1.0 = diffuse white (203 nits)
//   linear_acescg    scene-linear ACEScg (AP1), 1.0 = diffuse white
//   pq_rec2020       ST 2084 code in [0, 1], Rec.2020 (an HDR10 timeline)
// Alpha: the source's (default), or, for a matte in the host, the luminance limit or what AI drew.
//
// Two routes (25 Nov 2026, phase 2): "analytic", in this process on the CPU, and "learned",
// RUDRA's learned ACES 2.0 engine run beside the host (engine_client.hpp). The learned route
// uses Exposure, Strength (the engine's trim), the luminance limit and Output; Source curve, Recovery,
// Fill clipped highlights and Anchor are the analytic route's.
//
// 26 Nov 2026 (phase 3): the luminance limit (the app's Reconstruct > Region, core/composite.hpp
// ReconstructionRegion) on both routes and for AI fill, and AI fill itself (fill_store.hpp). Exposure is
// applied after the master (grade_master scales by 2^EV only, below its ceiling), so a stored fill, made at
// 0 EV, takes it too.
//
// Rows: the first row given is the picture's top (the plug-in hands the host's rows top first, with a
// negative stride for OFX's bottom-up images), so the learned model and AI fill see the picture upright.

#include <cstddef>
#include <string>
#include <vector>

// What the plug-in itself passes (ofx/plugin.cpp keeps to this plain interface): no RUDRA types.
namespace rudra_ofx {

struct Settings {
    std::string route = "analytic";        // analytic, learned
    std::string source_curve = "bt1886";   // core/source_curve.hpp ids
    double exposure_ev = 0.0;              // -3 .. +3
    double strength = 1.0;                 // the reconstruction's, 0 .. 1
    std::string recovery_mode = "all";     // all, highlights, shadows, off
    bool clip_fill = true;
    bool anchor = true;
    std::string output = "linear_rec2020";
    // The luminance limit, by the SDR's Rec.709 luma (0 .. 1): off, highlights (above highlights_from),
    // shadows (below shadows_below), both, band (band_from .. band_to); each edge `limit_softness` wide.
    std::string limit = "off";
    double highlights_from = 0.75, shadows_below = 0.15, band_from = 0.50, band_to = 0.90, limit_softness = 0.10;
    std::string alpha = "source";          // source, limit, ai_fill
    // AI fill (phase 3).
    bool ai_fill = false;
    std::string shot_id;                   // the instance's own (a hidden parameter, made at the first Generate)
    std::string prompt;
    int seed = 0;
    int frame = 0, first_frame = 0, last_frame = 0;   // this frame and the clip's
    bool full_size = true;                 // render scale 1: only then is a frame stored or a stored fill shown
};

// What a render did: `error` empty on success; `note` AI fill's state in one line ("" when off).
struct Outcome {
    std::string error;
    std::string note;
};

std::vector<std::string> source_ids();      // the Source menu, in order
std::vector<std::string> source_labels();
int default_source();                       // BT.1886's index

// The picture (above). `src`/`dst`: the first pixel of the top row; `*_row_floats`: floats from one row
// to the next down (negative for OFX's bottom-up rows).
Outcome render(const Settings& s, const float* src, std::ptrdiff_t src_row_floats, float* dst, std::ptrdiff_t dst_row_floats,
               int width, int height);
// Phase 1's call, every newer setting at its default; "" on success, else why not.
std::string render(const std::string& route, const std::string& source_curve, double exposure_ev, double strength,
                   const std::string& recovery_mode, bool clip_fill, bool anchor, const std::string& output, const float* src,
                   std::ptrdiff_t src_row_floats, float* dst, std::ptrdiff_t dst_row_floats, int width, int height);

// AI fill's buttons, from the host's main thread: "" on success, else why not.
std::string ai_generate(const Settings& s);
std::string ai_cancel(const Settings& s);
std::string ai_clear(const Settings& s);
std::string ai_status(const Settings& s);   // one line for the Status parameter
std::string new_shot_id();                  // 16 hex digits, for the hidden parameter

// Ends the learned engine, if one runs (the plug-in's unload). A running AI fill ends with the host.
void stop_engine();

}  // namespace rudra_ofx

#ifndef RUDRA_OFX_PLAIN
#include "rudra/core/composite.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct OfxGrade {
    std::string route = "analytic";        // analytic, learned
    std::string source_curve = "bt1886";   // core/source_curve.hpp ids
    double exposure_ev = 0.0;              // -3 .. +3
    double strength = 1.0;                 // the reconstruction's, 0 .. 1
    std::string recovery_mode = "all";     // all, highlights, shadows, off
    bool clip_fill = true;
    bool anchor = true;
    std::string output = "linear_rec2020";
    ReconstructionRegion region;           // the luminance limit (off: the whole frame)
    std::string alpha = "source";          // source, limit, ai_fill (no fill here: 0)
};

// The output choices, in the plug-in's menu order.
inline constexpr const char* kOfxOutputs[] = {"linear_rec2020", "linear_acescg", "pq_rec2020"};

// The luminance limit of the plain settings (off: disabled).
ReconstructionRegion ofx_region(const rudra_ofx::Settings& s);
OfxGrade ofx_grade(const rudra_ofx::Settings& s);

// `src`/`dst`: the first pixel of the top row; `*_row_floats`: floats from one row to the next down.
// The whole frame: the anchor and the grain settling look at neighbours.
Result<void> render_sdr_to_hdr(const OfxGrade& grade, const float* src, std::ptrdiff_t src_row_floats, float* dst,
                               std::ptrdiff_t dst_row_floats, int width, int height);

}  // namespace rudra
#endif  // RUDRA_OFX_PLAIN
