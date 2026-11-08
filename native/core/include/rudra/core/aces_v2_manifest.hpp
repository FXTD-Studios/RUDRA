#pragma once
// manifest.json of an ACES v2 staged package (tools/export_native_v2.py).
// Separate from the v1 ModelManifest: a v1 build refuses these packages
// (contract major 2) and this reader refuses anything that is not exactly
// the staged contract this build implements.

#include <filesystem>
#include <string>

#include "rudra/core/aces_v2.hpp"
#include "rudra/platform/result.hpp"

namespace rudra::aces_v2 {

inline constexpr int kContractMajor = 2;
// Major 3 (7 Nov 2026): the same stages plus the lock (stage 6b), which major 3 requires.
// An app built before the lock reads major 3 as "a different version of RUDRA" and refuses
// it, instead of running the model without the lock it was released with.
inline constexpr int kContractMajorLock = 3;
inline constexpr const char* kContractId = "rudra_native_aces_v2_staged";

struct Manifest {
    std::filesystem::path root, g1;          // g1: absolute path of rudra_g1.ts
    std::filesystem::path g1_onnx;           // rudra_g1.onnx (ONNX Runtime / DirectML); empty if absent
    std::string g1_onnx_sha256;
    double onnx_limit_display_codes = 0.0;   // export-measured tolerance vs LibTorch
    std::string g1_sha256, checkpoint_sha256, model_contract_id;
    std::string ocio_version, ocio_config, ocio_view, ocio_display;
    std::string smoothing;                   // "hybrid64"
    bool forbid_new_negative_y = false;      // appearance_constraint.forbid_new_negative_y (optional, default false)
    // "lock" (optional, 7 Nov 2026): stage 6b, the registered arm the package was released with
    // (aces_lock.hpp). Absent: off. A lock this build does not implement is refused.
    LockSettings lock{};
    std::string lock_name;                   // the registered arm, e.g. "lock_dqs050g"
    bool parity_pass = false, release_authorized = false;
};

// Reads root/manifest.json; checks contract id/major, the smoothing and
// appearance-constraint contract, and that the export's parity check passed.
Result<Manifest> read_manifest(const std::filesystem::path& package_root);
// Recomputes the G1 file hash.
Result<void> verify_files(const Manifest& m);

}  // namespace rudra::aces_v2
