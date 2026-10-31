#pragma once
// Runtime pieces of the ACES v2 staged pipeline (core/aces_v2.hpp):
//  - the declared ACES 2.0 view through OpenColorIO (RUDRA_WITH_OCIO), and
//  - the G1 network stage through LibTorch (RUDRA_WITH_LIBTORCH).
// Both refuse anything but the identities the model package pins.

#include <filesystem>
#include <memory>
#include <string>

#include "rudra/core/aces_v2.hpp"
#include "rudra/platform/result.hpp"

namespace rudra::aces_v2 {

inline constexpr const char* kOcioConfig = "cg-config-v4.0.0_aces-v2.0_ocio-v2.5";
inline constexpr const char* kOcioDisplay = "Rec.1886 Rec.709 - Display";
inline constexpr const char* kOcioView = "ACES 2.0 - SDR 100 nits (Rec.709)";
inline constexpr const char* kOcioVersion = "2.5.2";   // checkpoint contract; other versions refused

// The ACES 2.0 HDR output the Deliver tab offers as "PQ 1000" (27 Oct 2026):
// a display-referred deliverable, not the model's view.
inline constexpr const char* kOcioHdrDisplay = "Rec.2100-PQ - Display";
inline constexpr const char* kOcioHdrView = "ACES 2.0 - HDR 1000 nits (P3 D65)";

// The ACES studio config (same ACES 2.0 transforms, more outputs): HDR 2000 / 4000
// nits and Rec.2100-HLG, which the CG config does not carry (28 Oct 2026 review).
inline constexpr const char* kOcioStudioConfig = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5";

// threads: OCIO is pointwise; >1 splits packed spans across threads.
Result<std::unique_ptr<AcesView>> make_ocio_view(int threads = 1, bool require_exact_version = true);
// Any display and view of a built-in config (forward: ACEScg -> display codes); the CG config by default.
Result<std::unique_ptr<AcesView>> make_ocio_display_view(const char* display, const char* view, int threads = 1,
                                                         bool require_exact_version = true, const char* config = nullptr);

// Loads rudra_g1.ts (tools/export_native_v2.py). device: "cpu", "cuda", "cuda:N".
// Loads rudra_g1.onnx through ONNX Runtime. device: "cpu", "cuda", "directml".
// Tolerance contract vs LibTorch (manifest onnx_tolerance), not bit parity.
Result<G1Fn> load_g1_onnxruntime(const std::filesystem::path& onnx, const std::string& device = "cpu");

Result<G1Fn> load_g1_libtorch(const std::filesystem::path& torchscript, const std::string& device = "cpu");

}  // namespace rudra::aces_v2
