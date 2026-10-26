#pragma once
// ACES v2 master as a background job: for each source frame, decode, run the
// staged pipeline (core/aces_v2.hpp), write the float32 ACEScg EXR with the
// rudra.reconstructed mask, report. The app owns the view and the network
// (infer/aces_v2_runtime.hpp) and hands them in; decode and write are
// injectable so the job is testable without OpenCV, OCIO or LibTorch.
// Cancel stops after the frame in hand; a failure stops the job and says
// after how many frames.

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "rudra/core/aces_v2.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct AcesV2Request {
    std::vector<std::filesystem::path> sources;   // stills, in render order
    std::filesystem::path out_dir;                // must not exist yet
    bool smoothing = true;                        // the package's hybrid64 contract
    std::string g1_sha256;                        // recorded in each frame's provenance
    bool forbid_new_negative_y = false;           // the package's appearance_constraint.forbid_new_negative_y (last: keeps aggregate order)
};

struct AcesV2Decoded {
    PlanarBuffer rgb;   // display codes [0,1], 3 planes
    int bits = 8;
};
using AcesV2Decode = std::function<Result<AcesV2Decoded>(const std::filesystem::path& source)>;
// Default writer: deliver/write_acescg_exr, float32, mask channel (aces_v2_job.cpp).
using AcesV2Write = std::function<Result<void>(const std::filesystem::path& exr, const aces_v2::FrameResult& frame,
                                               const std::vector<std::pair<std::string, std::string>>& provenance)>;
AcesV2Write default_aces_v2_writer();

struct AcesV2FrameReport {
    std::size_t index = 0;
    std::filesystem::path source, exr;
    int bits = 8;
    double milliseconds = 0;
    double reconstructed_fraction = 0;
    aces_v2::SmoothingStats smoothing;
    aces_v2::ConstraintStats constraint;
};

struct AcesV2Outcome {
    std::size_t completed = 0, total = 0;
    bool cancelled = false;
    bool appearance_pass = true;            // every frame: 0 remaining display errors
    std::optional<Error> error;
    std::vector<AcesV2FrameReport> frames;
};

class AcesV2Job {
public:
    AcesV2Job(const aces_v2::AcesView& view, aces_v2::G1Fn g1, AcesV2Request request, AcesV2Decode decode,
              AcesV2Write write = default_aces_v2_writer());
    ~AcesV2Job();   // cancels and waits
    AcesV2Job(const AcesV2Job&) = delete;
    AcesV2Job& operator=(const AcesV2Job&) = delete;

    // Callbacks run on the job's thread: hand them to the UI thread yourself.
    void start(std::function<void(const AcesV2FrameReport&)> on_frame, std::function<void(const AcesV2Outcome&)> on_done);
    // Same work on the calling thread (CLI, tests).
    AcesV2Outcome run(const std::function<void(const AcesV2FrameReport&)>& on_frame = {});
    void cancel() { cancel_ = true; }
    bool running() const { return running_; }

private:
    const aces_v2::AcesView& view_;
    aces_v2::G1Fn g1_;
    AcesV2Request request_;
    AcesV2Decode decode_;
    AcesV2Write write_;
    std::atomic<bool> cancel_{false}, running_{false};
    std::thread thread_;
};

}  // namespace rudra
