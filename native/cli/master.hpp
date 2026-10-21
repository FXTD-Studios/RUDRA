#pragma once
// `rudra-native master`: one still to a master EXR and its sidecar with no
// Python (Phase 1 step 7). Decode (media) and the network (infer) here; the
// stages after it are deliver/master.hpp, which the app's master job shares.

#include <filesystem>

#include "rudra/core/model_manifest.hpp"
#include "rudra/deliver/correction_log.hpp"
#include "rudra/deliver/dolby_vision.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

Result<MasterResult> render_master(const ModelManifest& package, InferenceBackend& backend,
                                   const std::filesystem::path& image, const MasterRequest& request,
                                   const std::filesystem::path& out);

// Correction logging (roadmap 5.2): the same still and settings as a master,
// the accepted picture logged as a training pair in `log_dir`. `package` and
// `backend` null for an analytic request.
Result<CorrectionEntry> log_master_correction(const ModelManifest* package, InferenceBackend* backend,
                                              const std::filesystem::path& log_dir, const std::filesystem::path& image,
                                              const MasterRequest& request, const std::string& label = {});

}  // namespace rudra
