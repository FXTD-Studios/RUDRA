#pragma once
// manifest.json of an ACES v2 staged package (tools/export_native_v2.py).
// Separate from the v1 ModelManifest: a v1 build refuses these packages
// (contract major 2) and this reader refuses anything that is not exactly
// the staged contract this build implements.

#include <filesystem>
#include <string>

#include "rudra/platform/result.hpp"

namespace rudra::aces_v2 {

inline constexpr int kContractMajor = 2;
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
    bool parity_pass = false, release_authorized = false;
};

// Reads root/manifest.json; checks contract id/major, the smoothing and
// appearance-constraint contract, and that the export's parity check passed.
Result<Manifest> read_manifest(const std::filesystem::path& package_root);
// Recomputes the G1 file hash.
Result<void> verify_files(const Manifest& m);

}  // namespace rudra::aces_v2
