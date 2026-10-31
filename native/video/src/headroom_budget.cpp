#include "rudra/video/headroom_budget.hpp"

#include <algorithm>
#include <mutex>

namespace rudra {
struct HeadroomPixelBudget::State {
    mutable std::mutex mutex;
    HeadroomPixelStats stats;
};
struct HeadroomPixelBudget::Reservation {
    explicit Reservation(std::shared_ptr<State> s) : state(std::move(s)) {}
    ~Reservation() {
        std::lock_guard lock(state->mutex);
        state->stats.live -= bytes;
    }
    std::shared_ptr<State> state;
    std::size_t bytes = 0;
};
HeadroomPixelBudget::HeadroomPixelBudget(std::size_t bytes) : state_(std::make_shared<State>()) {
    state_->stats.limit = bytes;
}
HeadroomPixelStats HeadroomPixelBudget::stats() const {
    std::lock_guard lock(state_->mutex);
    return state_->stats;
}
Result<std::shared_ptr<HeadroomPixelBudget::Reservation>> HeadroomPixelBudget::reserve(std::size_t bytes) {
    // Allocate before charging so allocation failure cannot leak an allowance.
    auto ticket = std::make_shared<Reservation>(state_);
    std::lock_guard lock(state_->mutex);
    auto& s = state_->stats;
    if (!bytes || bytes > s.limit)
        return make_error(ErrorCode::InvalidArgument, "Shared headroom budget cannot hold a frame");
    if (bytes > s.limit - s.live)
        return make_error(ErrorCode::Busy, "Release retained headroom pixels before decoding another frame");
    ticket->bytes = bytes; s.live += bytes; s.peak = std::max(s.peak, s.live);
    return ticket;
}
} // namespace rudra
