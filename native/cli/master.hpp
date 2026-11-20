#pragma once
// `rudra-native master`: one still to a master EXR and its sidecar with no
// Python (Phase 1 step 7). Decode (media) and the network (infer) here; the
// stages after it are deliver/master.hpp, which the app's master job shares.

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/core/model_manifest.hpp"
#include "rudra/deliver/correction_log.hpp"
#include "rudra/deliver/dolby_vision.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// RUDRA 1.1 (5 Nov 2026): the analytic master with no model package at all, what
// `rudra-native master analytic <image> ...` runs (tools/validate_rudra_1_1.py uses it).
Result<MasterResult> render_analytic_master(const std::filesystem::path& image, const MasterRequest& q,
                                            const std::filesystem::path& out);
Result<MasterResult> render_master(const ModelManifest& package, InferenceBackend& backend,
                                   const std::filesystem::path& image, const MasterRequest& request,
                                   const std::filesystem::path& out);

// 1.2.7 (multi-role review of 1.2.6): the app's defaults for a headless render, so a script and the
// app make the same master from the same shot. Into `params_json` (an object, or empty for {}):
// "source_curve": "bt1886" (the app's Source default; the engine's own is "unknown", which made the
// 09V frame's MaxFALL 142 nits against 39) and, for a lossy source (a JPEG still; a movie when
// `lossy_movie`), "lossy_source": true, each only when the parameters do not say. `notes` gets one
// line per default taken. Not JSON: returned as it is (the parser then says why).
std::string with_app_defaults(const std::string& params_json, const std::filesystem::path& source, bool lossy_movie,
                              std::vector<std::string>* notes = nullptr);

// Correction logging (roadmap 5.2): the same still and settings as a master,
// the accepted picture logged as a training pair in `log_dir`. `package` and
// `backend` null for an analytic request.
Result<CorrectionEntry> log_master_correction(const ModelManifest* package, InferenceBackend* backend,
                                              const std::filesystem::path& log_dir, const std::filesystem::path& image,
                                              const MasterRequest& request, const std::string& label = {});

}  // namespace rudra
