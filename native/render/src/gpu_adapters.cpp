#include "rudra/render/gpu_adapters.hpp"

#include <algorithm>

#if defined(_WIN32)
#include <dxgi1_6.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace rudra {

std::optional<std::vector<GpuAdapter>> gpu_adapters() {
#if defined(_WIN32)
    std::vector<GpuAdapter> out;
    IDXGIFactory1* factory = nullptr;
    // No factory: nothing was checked, which is not the same as no GPU.
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return std::nullopt;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
        DXGI_ADAPTER_DESC1 d{};
        // Skip the software rasterizer (Microsoft Basic Render Driver, vendor 0x1414).
        if (SUCCEEDED(adapter->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.VendorId != 0x1414) {
            const int n = WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, nullptr, 0, nullptr, nullptr);
            std::string name(n > 1 ? std::size_t(n - 1) : 0, '\0');
            if (n > 1) WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name.data(), n, nullptr, nullptr);
            while (!name.empty() && name.back() == ' ') name.pop_back();
            const bool seen = std::any_of(out.begin(), out.end(), [&](const GpuAdapter& o) { return o.name == name; });
            if (!seen && !name.empty())
                out.push_back({name, double(d.DedicatedVideoMemory) / (1024.0 * 1024.0 * 1024.0)});
        }
        adapter->Release();
    }
    factory->Release();
    return out;
#elif defined(__APPLE__)
    // Apple silicon's GPU is on the chip, so the chip names it ("Apple M2 Pro"); its memory
    // is the Mac's own, shared, so none is reported as the GPU's.
    char brand[256] = {};
    std::size_t len = sizeof brand;
    std::string chip = sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0 ? std::string(brand) : std::string();
    while (!chip.empty() && (chip.back() == ' ' || chip.back() == '\0')) chip.pop_back();
    if (chip.rfind("Apple", 0) != 0) return std::nullopt;   // an Intel Mac: its GPU is not the CPU's name
    return std::vector<GpuAdapter>{{chip + " GPU", 0.0}};
#else
    return std::nullopt;
#endif
}

}  // namespace rudra
