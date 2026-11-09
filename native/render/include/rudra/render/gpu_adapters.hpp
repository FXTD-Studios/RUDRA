#pragma once
// The machine's GPUs, found without a model (the first-run check, 9 Nov 2026):
// DXGI's hardware adapters on Windows, the Metal GPU every supported Mac has.

#include <optional>
#include <string>
#include <vector>

namespace rudra {

struct GpuAdapter {
    std::string name;      // UTF-8, as the driver reports it
    double vram_gb = 0.0;  // dedicated memory; 0 when none is reported (integrated)
};

// nullopt: not checked on this system (Linux). Software rasterizers are left out.
std::optional<std::vector<GpuAdapter>> gpu_adapters();

}  // namespace rudra
