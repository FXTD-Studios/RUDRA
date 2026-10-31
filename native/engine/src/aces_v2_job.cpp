#include "rudra/engine/aces_v2_job.hpp"

#include <chrono>

#include "rudra/deliver/exr.hpp"

namespace rudra {

AcesV2Write default_aces_v2_writer() {
    return [](const std::filesystem::path& exr, const aces_v2::FrameResult& f,
              const std::vector<std::pair<std::string, std::string>>& provenance) -> Result<void> {
        return write_acescg_exr(exr, f.acescg, Primaries::Ap1, 1.0, provenance, /*half=*/false, &f.mask);
    };
}

AcesV2Job::AcesV2Job(const aces_v2::AcesView& view, aces_v2::G1Fn g1, AcesV2Request request, AcesV2Decode decode,
                     AcesV2Write write)
    : view_(view), g1_(std::move(g1)), request_(std::move(request)), decode_(std::move(decode)), write_(std::move(write)) {}

AcesV2Job::~AcesV2Job() {
    cancel_ = true;
    if (thread_.joinable()) thread_.join();
}

AcesV2Outcome AcesV2Job::run(const std::function<void(const AcesV2FrameReport&)>& on_frame) {
    AcesV2Outcome out;
    out.total = request_.sources.size();
    std::error_code ec;
    if (std::filesystem::exists(request_.out_dir, ec)) {
        out.error = make_error(ErrorCode::InvalidArgument, "Choose a new output folder.", request_.out_dir.string());
        return out;
    }
    std::filesystem::create_directories(request_.out_dir, ec);
    if (ec) {
        out.error = make_error(ErrorCode::IoError, "Could not create the output folder.", ec.message());
        return out;
    }
    for (std::size_t i = 0; i < request_.sources.size(); ++i) {
        if (cancel_) {
            out.cancelled = true;
            break;
        }
        const auto& src = request_.sources[i];
        auto decoded = decode_(src);
        if (!decoded) {
            out.error = decoded.error();
            break;
        }
        const auto t0 = std::chrono::steady_clock::now();
        auto frame = aces_v2::run_frame(decoded->rgb, view_, g1_, request_.smoothing, true, request_.forbid_new_negative_y);
        if (!frame) {
            out.error = frame.error();
            break;
        }
        AcesV2FrameReport rep;
        rep.index = i;
        rep.source = src;
        rep.bits = decoded->bits;
        rep.exr = request_.out_dir / (src.stem().string() + ".exr");
        rep.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::size_t marked = 0;
        for (float v : frame->mask.span()) marked += v > 0.0f;
        rep.reconstructed_fraction = frame->mask.plane_size() ? double(marked) / double(frame->mask.plane_size()) : 0.0;
        rep.smoothing = frame->smoothing;
        rep.constraint = frame->constraint;
        const std::vector<std::pair<std::string, std::string>> prov = {
            {"contract", "rudra_native_aces_v2_staged"}, {"g1_sha256", request_.g1_sha256},
            {"source", src.filename().string()}, {"view", view_.identity()},
            {"clipped_smoothing", request_.smoothing ? "hybrid64" : "none"},
            {"luminance_guard", request_.forbid_new_negative_y ? "forbid_new_negative_y" : "none"}};
        if (auto w = write_(rep.exr, *frame, prov); !w) {
            out.error = w.error();
            break;
        }
        out.appearance_pass = out.appearance_pass && rep.constraint.remaining_display_errors == 0 &&
                              (!request_.forbid_new_negative_y || rep.constraint.new_negative_luminance == 0);
        out.frames.push_back(rep);
        ++out.completed;
        if (on_frame) on_frame(rep);
    }
    return out;
}

void AcesV2Job::start(std::function<void(const AcesV2FrameReport&)> on_frame,
                      std::function<void(const AcesV2Outcome&)> on_done) {
    if (running_) return;
    if (thread_.joinable()) thread_.join();
    cancel_ = false;
    running_ = true;
    thread_ = std::thread([this, on_frame = std::move(on_frame), on_done = std::move(on_done)] {
        AcesV2Outcome o = run(on_frame);
        running_ = false;
        if (on_done) on_done(o);
    });
}

}  // namespace rudra
