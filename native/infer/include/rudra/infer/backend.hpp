#pragma once
// InferenceBackend (NATIVE_ARCHITECTURE.md 5.4): one interface, several runtimes.
// A backend runs the two entry points of a model package; the tiler, the
// stitching and everything after the fields live outside it.

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/model_manifest.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

enum class Runtime { LibTorch, OnnxRuntime };
// TensorRT: ONNX Runtime's TensorRT provider on an NVIDIA GPU (CUDA after it
// for what TensorRT does not take), the real-time path on the RTX cards.
enum class Device { Cpu, Cuda, Mps, DirectML, CoreML, Rocm, OpenVino, TensorRT };
// Fp16: the package's half-precision tile graph (model.tile.fp16.onnx, roadmap
// R1), held to the manifest's "fp16" tolerance. The frame pass stays fp32.
enum class Precision { Fp32, Fp16 };

const char* to_string(Runtime r) noexcept;
const char* to_string(Device d) noexcept;
const char* to_string(Precision p) noexcept;
// "cpu", "cuda", ..., "tensorrt": the names the command line and settings use.
std::optional<Device> device_from_string(std::string_view name) noexcept;
std::optional<Precision> precision_from_string(std::string_view name) noexcept;
// Every device, in enum order.
inline constexpr Device kAllDevices[] = {Device::Cpu,    Device::Cuda, Device::Mps,      Device::DirectML,
                                         Device::CoreML, Device::Rocm, Device::OpenVino, Device::TensorRT};

struct BackendInfo {
    Runtime runtime;
    Device device;
    std::string version;       // runtime version
    std::string detail;        // device name or provider list
    Precision precision = Precision::Fp32;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual BackendInfo info() const = 0;
    virtual Result<FrameScalars> frame_pass(const SdrImage& frame) = 0;
    virtual Result<Fields> tile_pass(const SdrImage& tile, const FrameScalars& scalars) = 0;
};

// Factories. Each returns an Unsupported error when this build lacks the runtime.
// Fp16 needs a package with the fp16 graph; LibTorch is fp32 only for now.
Result<std::unique_ptr<InferenceBackend>> make_libtorch_backend(const ModelManifest& m, Device device,
                                                                Precision precision = Precision::Fp32);
Result<std::unique_ptr<InferenceBackend>> make_onnxruntime_backend(const ModelManifest& m, Device device,
                                                                   Precision precision = Precision::Fp32);

// Which runtimes this binary was built with.
std::vector<Runtime> compiled_runtimes();

}  // namespace rudra
