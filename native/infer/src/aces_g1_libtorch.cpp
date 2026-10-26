// G1 (tools/export_native_v2.py) through LibTorch. Inputs and outputs are
// planar float32 NCHW with N = 1, the PlanarBuffer layout.

#include <torch/script.h>
#include <ATen/Context.h>
#include <torch/cuda.h>

#include <cstring>
#include <memory>

#include "rudra/infer/aces_v2_runtime.hpp"
#include "torch_cuda_loader.hpp"

namespace rudra::aces_v2 {
namespace {

torch::Tensor as_tensor(const PlanarBuffer& b, const torch::Device& dev) {
    auto t = torch::from_blob(const_cast<float*>(b.span().data()),
                              {1, std::int64_t{b.channels()}, std::int64_t{b.height()}, std::int64_t{b.width()}},
                              torch::TensorOptions().dtype(torch::kFloat32));
    return dev.is_cpu() ? t : t.to(dev);
}

PlanarBuffer as_planar(const torch::Tensor& t) {
    auto c = t.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const auto s = c.sizes();
    std::vector<float> v(static_cast<std::size_t>(c.numel()));
    std::memcpy(v.data(), c.data_ptr<float>(), v.size() * sizeof(float));
    return PlanarBuffer(static_cast<int>(s[1]), static_cast<int>(s[2]), static_cast<int>(s[3]), std::move(v));
}

}  // namespace

Result<G1Fn> load_g1_libtorch(const std::filesystem::path& torchscript, const std::string& device) {
    try {
        torch::Device dev(device);
        if (dev.is_cuda()) {
            detail::load_torch_cuda();
            if (!torch::cuda::is_available())
                return make_error(ErrorCode::Unsupported, "No CUDA device is available to LibTorch.");
            at::globalContext().setAllowTF32CuDNN(false);
            at::globalContext().setAllowTF32CuBLAS(false);
            at::globalContext().setFloat32MatmulPrecision("highest");
        }
        auto module = std::make_shared<torch::jit::Module>(torch::jit::load(torchscript.string(), dev));
        module->eval();
        G1Fn fn = [module, dev](const PlanarBuffer& source, const PlanarBuffer& baseline) -> Result<G1Output> {
            try {
                c10::InferenceMode guard;
                auto out = module->forward({as_tensor(source, dev), as_tensor(baseline, dev)}).toTuple();
                const auto& e = out->elements();
                if (e.size() != 4) return make_error(ErrorCode::ContractMismatch, "The model returned the wrong outputs.");
                return G1Output{as_planar(e[0].toTensor()), as_planar(e[1].toTensor()), as_planar(e[2].toTensor()),
                                as_planar(e[3].toTensor())};
            } catch (const std::exception& ex) {
                return make_error(ErrorCode::BackendError, "LibTorch could not run the ACES model.", ex.what());
            }
        };
        return fn;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::BackendError, "LibTorch could not load the ACES model.", e.what());
    }
}

}  // namespace rudra::aces_v2
