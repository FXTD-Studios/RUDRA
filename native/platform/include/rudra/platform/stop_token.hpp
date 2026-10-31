#pragma once
// rudra::stop_token, stop_source and stop_callback (31 Oct 2026).
//
// The standard ones where the library has them (libstdc++, MSVC, newer libc++);
// a small equivalent where it does not: Apple's libc++ in Xcode 15/16 ships
// <stop_token> only behind -fexperimental-library, so the macOS builds did not
// compile. Same names, same behaviour for what RUDRA uses: a default token never
// stops; request_stop runs the registered callbacks once, on the requesting
// thread; a callback registered after the stop runs at once; destroying a
// callback while it runs on another thread waits for it to finish.

#include <version>

#if defined(__cpp_lib_jthread) && !defined(RUDRA_FORCE_STOP_TOKEN_SHIM)

#include <stop_token>

namespace rudra {
using std::nostopstate;
using std::nostopstate_t;
using std::stop_callback;
using std::stop_source;
using std::stop_token;
}  // namespace rudra

#else

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace rudra {

namespace detail {
struct StopState {
    std::mutex m;
    std::condition_variable done;
    std::atomic<bool> stopped{false};
    std::map<std::uint64_t, std::function<void()>> callbacks;
    std::uint64_t next = 1;
    std::uint64_t running = 0;
    std::thread::id requester;

    bool request_stop() {
        std::unique_lock lock(m);
        if (stopped.load()) return false;
        stopped.store(true);
        requester = std::this_thread::get_id();
        while (!callbacks.empty()) {
            auto it = callbacks.begin();
            running = it->first;
            auto fn = std::move(it->second);
            callbacks.erase(it);
            lock.unlock();
            fn();
            lock.lock();
            running = 0;
            done.notify_all();
        }
        return true;
    }
};
}  // namespace detail

struct nostopstate_t {
    explicit nostopstate_t() = default;
};
inline constexpr nostopstate_t nostopstate{};

class stop_token {
public:
    stop_token() noexcept = default;
    [[nodiscard]] bool stop_requested() const noexcept { return state_ && state_->stopped.load(); }
    [[nodiscard]] bool stop_possible() const noexcept { return state_ != nullptr; }
    friend bool operator==(const stop_token& a, const stop_token& b) noexcept { return a.state_ == b.state_; }
    void swap(stop_token& o) noexcept { state_.swap(o.state_); }

private:
    friend class stop_source;
    template <class> friend class stop_callback;
    explicit stop_token(std::shared_ptr<detail::StopState> s) noexcept : state_(std::move(s)) {}
    std::shared_ptr<detail::StopState> state_;
};

class stop_source {
public:
    stop_source() : state_(std::make_shared<detail::StopState>()) {}
    explicit stop_source(nostopstate_t) noexcept {}
    [[nodiscard]] stop_token get_token() const noexcept { return stop_token(state_); }
    [[nodiscard]] bool stop_possible() const noexcept { return state_ != nullptr; }
    [[nodiscard]] bool stop_requested() const noexcept { return state_ && state_->stopped.load(); }
    bool request_stop() noexcept { return state_ && state_->request_stop(); }
    friend bool operator==(const stop_source& a, const stop_source& b) noexcept { return a.state_ == b.state_; }
    void swap(stop_source& o) noexcept { state_.swap(o.state_); }

private:
    std::shared_ptr<detail::StopState> state_;
};

template <class Callback>
class stop_callback {
public:
    using callback_type = Callback;

    template <class C>
    explicit stop_callback(const stop_token& token, C&& cb) : cb_(std::forward<C>(cb)) {
        auto s = token.state_;
        if (!s) return;
        std::unique_lock lock(s->m);
        if (s->stopped.load()) {
            lock.unlock();
            cb_();
            return;
        }
        id_ = s->next++;
        s->callbacks.emplace(id_, [this] { cb_(); });
        state_ = std::move(s);
    }

    ~stop_callback() {
        if (!state_) return;
        std::unique_lock lock(state_->m);
        if (state_->callbacks.erase(id_)) return;
        // Running now: wait for it unless this thread is the one running it.
        if (state_->running == id_ && state_->requester != std::this_thread::get_id())
            state_->done.wait(lock, [&] { return state_->running != id_; });
    }

    stop_callback(const stop_callback&) = delete;
    stop_callback& operator=(const stop_callback&) = delete;

private:
    Callback cb_;
    std::shared_ptr<detail::StopState> state_;
    std::uint64_t id_ = 0;
};

template <class Callback>
stop_callback(stop_token, Callback) -> stop_callback<Callback>;

}  // namespace rudra

#endif
