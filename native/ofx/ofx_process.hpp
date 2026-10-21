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
// Alpha is carried as it came.

#include <cstddef>
#include <string>
#include <vector>

#ifndef RUDRA_OFX_PLAIN
#include "rudra/platform/result.hpp"

namespace rudra {

struct OfxGrade {
    std::string source_curve = "bt1886";   // core/source_curve.hpp ids
    double exposure_ev = 0.0;              // -3 .. +3
    double strength = 1.0;                 // the reconstruction's, 0 .. 1
    std::string recovery_mode = "all";     // all, highlights, shadows, off
    bool clip_fill = true;
    bool anchor = true;
    std::string output = "linear_rec2020";
};

// The output choices, in the plug-in's menu order.
inline constexpr const char* kOfxOutputs[] = {"linear_rec2020", "linear_acescg", "pq_rec2020"};

// `src`/`dst`: the first pixel of the first row; `*_row_floats`: floats from one
// row to the next (may be negative: OFX rows go bottom up). The whole frame:
// the anchor and the grain settling look at neighbours.
Result<void> render_sdr_to_hdr(const OfxGrade& grade, const float* src, std::ptrdiff_t src_row_floats, float* dst,
                               std::ptrdiff_t dst_row_floats, int width, int height);

}  // namespace rudra
#endif  // RUDRA_OFX_PLAIN

// What the plug-in itself calls (ofx/plugin.cpp builds as C++14, as the OFX
// Support library of the 1.4 SDK requires): no RUDRA types.
namespace rudra_ofx {
std::vector<std::string> source_ids();      // the Source menu, in order
std::vector<std::string> source_labels();
int default_source();                       // BT.1886's index
// The picture (above), every setting as plain values; "" on success, else why not.
std::string render(const std::string& source_curve, double exposure_ev, double strength, const std::string& recovery_mode,
                   bool clip_fill, bool anchor, const std::string& output, const float* src, std::ptrdiff_t src_row_floats,
                   float* dst, std::ptrdiff_t dst_row_floats, int width, int height);
}  // namespace rudra_ofx
