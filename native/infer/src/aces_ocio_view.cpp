// The declared ACES 2.0 SDR100 Rec.709 view through OpenColorIO's CPU
// processor: the same calls as rudra/aces_view.py (getDefaultCPUProcessor,
// packed RGB float32), so the C++ and Python masters agree.

#include <OpenColorIO/OpenColorIO.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#include "rudra/infer/aces_v2_runtime.hpp"

namespace OCIO = OCIO_NAMESPACE;

namespace rudra::aces_v2 {
namespace {

class OcioView final : public AcesView {
public:
    OcioView(OCIO::ConstCPUProcessorRcPtr fwd, OCIO::ConstCPUProcessorRcPtr inv, int threads, std::string id)
        : fwd_(std::move(fwd)), inv_(std::move(inv)), threads_(threads), id_(std::move(id)) {}

    void forward(float* rgb, std::size_t pixels) const override { apply(*fwd_, rgb, pixels); }
    void inverse(float* rgb, std::size_t pixels) const override { apply(*inv_, rgb, pixels); }
    std::string identity() const override { return id_; }

private:
    void apply(const OCIO::CPUProcessor& p, float* rgb, std::size_t pixels) const {
        if (pixels == 0) return;
        // Same split rule as rudra/aces_view.py: one span below 262144 pixels.
        if (threads_ <= 1 || pixels < 262144) {
            run(p, rgb, pixels);
            return;
        }
        std::vector<std::thread> pool;
        const std::size_t t = static_cast<std::size_t>(threads_);
        const std::size_t base = pixels / t, extra = pixels % t;   // numpy.array_split sizes
        std::size_t start = 0;
        for (std::size_t i = 0; i < t; ++i) {
            const std::size_t count = base + (i < extra ? 1 : 0);
            pool.emplace_back([&p, rgb, start, count] { run(p, rgb + start * 3, count); });
            start += count;
        }
        for (auto& th : pool) th.join();
    }
    static void run(const OCIO::CPUProcessor& p, float* rgb, std::size_t pixels) {
        if (pixels == 0) return;
        OCIO::PackedImageDesc desc(rgb, static_cast<long>(pixels), 1, 3);
        p.apply(desc);
    }

    OCIO::ConstCPUProcessorRcPtr fwd_, inv_;
    int threads_;
    std::string id_;
};

}  // namespace

Result<std::unique_ptr<AcesView>> make_ocio_view(int threads, bool require_exact_version) {
    if (threads < 1 || threads > 16) return make_error(ErrorCode::InvalidArgument, "OCIO threads must be 1 to 16.");
    if (require_exact_version && std::strcmp(OCIO::GetVersion(), kOcioVersion) != 0)
        return make_error(ErrorCode::ContractMismatch, "This build has the wrong OpenColorIO version for the ACES model.",
                          std::string("need ") + kOcioVersion + ", have " + OCIO::GetVersion());
    try {
        auto config = OCIO::Config::CreateFromBuiltinConfig(kOcioConfig);
        auto t = OCIO::DisplayViewTransform::Create();
        t->setSrc("ACEScg");
        t->setDisplay(kOcioDisplay);
        t->setView(kOcioView);
        auto fwd = config->getProcessor(t)->getDefaultCPUProcessor();
        t->setDirection(OCIO::TRANSFORM_DIR_INVERSE);
        auto inv = config->getProcessor(t)->getDefaultCPUProcessor();
        std::string id = std::string("ocio ") + OCIO::GetVersion() + " " + kOcioConfig + " | " + kOcioDisplay + " | " + kOcioView;
        return std::unique_ptr<AcesView>(new OcioView(fwd, inv, threads, std::move(id)));
    } catch (const std::exception& e) {
        return make_error(ErrorCode::BackendError, "OpenColorIO could not build the ACES 2.0 view.", e.what());
    }
}

}  // namespace rudra::aces_v2
