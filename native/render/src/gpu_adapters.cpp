#include "rudra/render/gpu_adapters.hpp"

#include <algorithm>

#ifdef _WIN32
#include <dxgi1_6.h>
#endif

namespace rudra {

std::optional<std::vector<GpuAdapter>> gpu_adapters() {
#if defined(_WIN32)
    std::vector<GpuAdapter> out;
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return out;
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
    return std::vector<GpuAdapter>{{"Apple GPU (Metal)", 0.0}};
#else
    return std::nullopt;
#endif
}

}  // namespace rudra
