#pragma once

#include <memory>
#include "rudra/platform/result.hpp"

namespace rudra {
struct HeadroomPixelStats {
    std::size_t limit = 0, live = 0, peak = 0;
};

// Shared native pixel allowance. Charges staging and retained immutable pixels,
// including frames whose original decoder owner has gone away. Excludes FFmpeg,
// metadata and explicit pixel copies made by callers. Thread-safe, non-blocking
// reservation: release a frame and retry when Busy.
class HeadroomPixelBudget {
public:
    explicit HeadroomPixelBudget(std::size_t bytes);
    HeadroomPixelStats stats() const;
private:
    friend class HeadroomStream;
    struct State;
    struct Reservation;
    Result<std::shared_ptr<Reservation>> reserve(std::size_t bytes);
    std::shared_ptr<State> state_;
};
} // namespace rudra
