#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

namespace rudra::detail {
// Direct LibTorch imports link only CPU libraries. Loading CUDA registers its
// device guards; a CPU-only installation remains valid for CPU requests.
inline void load_torch_cuda() noexcept {
#if defined(_WIN32)
    LoadLibraryA("torch_cuda.dll");
#elif defined(__linux__)
    dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_GLOBAL);
#endif
}
}  // namespace rudra::detail
