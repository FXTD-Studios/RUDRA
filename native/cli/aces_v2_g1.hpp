#pragma once
// The learned ACES 2.0 model's G1, with the runtime asked for or the one this build has
// (31 Oct 2026). The release packages for Windows and macOS carry ONNX Runtime only; an
// app whose setting still says "LibTorch on CUDA" (the default) gets ONNX Runtime on the
// nearest device instead of a refusal, and the engine says which it used.

#include <string>
#include <vector>

#include "rudra/core/aces_v2.hpp"
#include "rudra/core/aces_v2_manifest.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
#include "rudra/platform/result.hpp"

namespace rudra::aces_v2 {

struct G1Choice {
    G1Fn fn;
    std::string runtime, device;
};

// ONNX Runtime devices to try for a request this build cannot serve with LibTorch:
// Windows' GPU through DirectML, else the CPU. On macOS the CPU (exact within the
// manifest's ONNX tolerance); Core ML is offered when asked for by name.
inline std::vector<std::string> onnx_fallback_devices(const std::string& device) {
    if (device == "cpu") return {"cpu"};
#ifdef _WIN32
    return {"directml", "cpu"};
#else
    return {"cpu"};
#endif
}

inline Result<G1Choice> load_g1_choice(const Manifest& m, const std::string& runtime, const std::string& device) {
    if (runtime != "libtorch" && runtime != "onnxruntime")
        return make_error(ErrorCode::Unsupported, "Unknown runtime (libtorch or onnxruntime).", runtime);
    if (runtime == "libtorch") {
#ifdef RUDRA_HAVE_LIBTORCH_ACES
        auto g = load_g1_libtorch(m.g1, device);
        if (!g) return g.error();
        return G1Choice{std::move(*g), "libtorch", device};
#elif defined(RUDRA_HAVE_ORT_ACES)
        if (m.g1_onnx.empty()) return make_error(ErrorCode::Unsupported, "This build has no LibTorch and the package has no ONNX model.");
        Error last = make_error(ErrorCode::Unsupported, "No ONNX Runtime device answered.");
        for (const auto& d : onnx_fallback_devices(device)) {
            auto g = load_g1_onnxruntime(m.g1_onnx, d);
            if (g) return G1Choice{std::move(*g), "onnxruntime", d};
            last = g.error();
        }
        return last;
#else
        return make_error(ErrorCode::Unsupported, "This build has no LibTorch.");
#endif
    }
    if (m.g1_onnx.empty()) return make_error(ErrorCode::Unsupported, "This package has no ONNX model; re-export it.");
#ifdef RUDRA_HAVE_ORT_ACES
    auto g = load_g1_onnxruntime(m.g1_onnx, device);
    if (!g) return g.error();
    return G1Choice{std::move(*g), "onnxruntime", device};
#else
    return make_error(ErrorCode::Unsupported, "This build has no ONNX Runtime.");
#endif
}

}  // namespace rudra::aces_v2
