#pragma once

#include <memory>
#include "rudra/platform/stop_token.hpp"

#include "rudra/core/headroom_policy.hpp"
#include "rudra/media/video_decode_float.hpp"

namespace rudra {
// Source-level namespace for future float-frame caches, never a legacy PNG
// name. Includes full source/decoder SHA-256, resolved paths, policy, contract,
// dimensions, clock and exact decoding arguments. Reads both files in full;
// compute at source-open, not for every frame. This function creates no cache.
// Unprotected helper: use PreparedHeadroomSource for lifetime-safe reuse.
Result<std::string> headroom_source_cache_id(const std::filesystem::path& source,
    const VideoSource& video, HeadroomPolicy policy, const std::filesystem::path& decoder_executable,
    rudra::stop_token stop = {});

struct HeadroomPrepareOptions {
    rudra::stop_token stop;
    // Empty resolves ffprobe on PATH; explicit path supports pinned tooling.
    std::filesystem::path probe_executable;
};

// Immutable validated source/decoder identity shared by independent ranges.
// Full probe and hashes run once, under read leases; no persistent/global cache.
// Hold this handle across seeks. Source/decoder stay protected until the final
// handle/owner is released. Windows only; other platforms fail closed.
// prepare() is synchronous: run off the UI thread. Stop covers child stdout/
// exit waits, hash chunks and phase boundaries, not an OS file I/O deadline.
// Its stop token has no effect after successful preparation. Each decoder
// owner retains independent cancellation, receipt identity and pixel budget.
class PreparedHeadroomSource {
public:
    using Handle = std::shared_ptr<const PreparedHeadroomSource>;
    static Result<Handle> prepare(const std::filesystem::path& source, HeadroomPolicy policy,
                                 const std::filesystem::path& decoder, HeadroomPrepareOptions options = {});
    ~PreparedHeadroomSource();
    PreparedHeadroomSource(const PreparedHeadroomSource&) = delete;
    PreparedHeadroomSource& operator=(const PreparedHeadroomSource&) = delete;
    const VideoSource& video() const;
    const std::string& cache_id() const;
    const std::filesystem::path& source_path() const;
    const std::filesystem::path& decoder_path() const;
    // Thread-safe. An observed identity failure is terminal for every sharing
    // owner, even if metadata is later restored; prepare a fresh source.
    // Owners must verify before publishing/using frames or receipts.
    Result<void> verify() const;
private:
    PreparedHeadroomSource();
    struct Impl;
    std::unique_ptr<Impl> d_;
};
}  // namespace rudra
