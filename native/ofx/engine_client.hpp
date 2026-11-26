#pragma once
// The OFX plug-in's learned ACES 2.0 route (phase 2, 25 Nov 2026): RUDRA's own engine,
// `rudra-native aces-v2-serve`, run as a child process, once per host session and shared by
// every instance of the effect. No model is loaded inside Resolve or Nuke, so nothing clashes
// with the host's own LibTorch or CUDA.
//
// Where the engine is: a small JSON file the RUDRA app (or tmp\BUILD_OFX.ps1 -Install) writes,
//   Windows  %LOCALAPPDATA%\FXTD Studios\RUDRA\ofx_engine.json
//   macOS    ~/Library/Application Support/FXTD Studios/RUDRA/ofx_engine.json
//   Linux    ~/.local/share/FXTD Studios/RUDRA/ofx_engine.json
// or the file RUDRA_OFX_ENGINE_CONFIG names:
//   {"cli": "<rudra-native>", "package": "<RUDRA_ACES_2 folder>", "runtime": "libtorch",
//    "device": "cuda", "path_prepend": ["<folder with the runtime's DLLs>", ...],
//    "genfill": {"python": "<python.exe>", "repo": "<RUDRA's Python source>", "base": "<Wan2.1-VACE-1.3B>",
//                "lora": "<rudra_genfill.safetensors, AI fill v5>"}}
// 26 Nov 2026: path_prepend is for a development engine only. Nuke hands its own DLL folder (its LibTorch) to
// every program it starts, and Windows looks there before PATH (platform/process.cpp); an engine whose DLLs sit
// beside it (the installed RUDRA's, or tmp\BUILD_OFX.ps1's build\ofx_engine) is not affected.
//
// Per frame: the host's codes as a 16-bit TIFF in the temp folder, a `preview` request with
// "model_out", and the master read back as raw float32 ACEScg planes (1.0 = diffuse white).

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra::ofx_engine {

// AI fill (phase 3, 26 Nov 2026): what the RUDRA app's AI fill runs on (Deliver > AI fill > Setup).
struct GenFillConfig {
    std::filesystem::path python, repo, base, lora;
};

struct EngineConfig {
    std::filesystem::path cli, package;
    std::string runtime = "libtorch", device = "cuda";
    std::vector<std::filesystem::path> path_prepend;
    std::filesystem::path from;   // the file it was read from
    GenFillConfig genfill;        // empty paths when the file has none
};

// The config file's path (RUDRA_OFX_ENGINE_CONFIG, else the per-user file above).
std::filesystem::path engine_config_path();
// Reads and checks it: the CLI and the package's manifest.json must exist.
Result<EngineConfig> load_engine_config();
// Reads the AI fill's part and checks it: the Python, rudra/genfill/run.py in the repository, the base's
// transformer/config.json and the LoRA must exist. Needs no engine.
Result<GenFillConfig> load_genfill_config();

// 16-bit RGB TIFF, uncompressed, one strip: what the engine's still reader (OpenCV) reads at full depth.
Result<void> write_tiff16(const std::filesystem::path& path, const SdrImage& codes);

// The SDR codes through the learned ACES 2.0 master: ACEScg planes, 1.0 = diffuse white.
// `strength` is the engine's trim (1 = as the model made it). `region_json`: the luminance limit, the app's
// reconstruction region as deliver/master.hpp writes it ("" = the whole frame). Thread-safe: one frame at a
// time. The engine starts on the first call (its model loads once, which takes seconds) and is started
// again once if it has stopped.
Result<PlanarBuffer> learned_master(const SdrImage& codes, double strength, const std::string& region_json = {});

// Ends the engine (the plug-in's unload).
void stop_engine();

}  // namespace rudra::ofx_engine
