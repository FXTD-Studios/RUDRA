#pragma once
// The learned ACES 2.0 model kept loaded (26 Oct 2026): `rudra-native
// aces-v2-serve` started once (at app start when aces/preload is on, or when
// the ACES window needs it) and asked over JSON lines. The app shows a frame
// through the model at any exposure and masters with nothing loaded twice.
// No Q_OBJECT (as elsewhere in the app): listeners are callbacks.

#include <QByteArray>
#include <QJsonObject>
#include <QString>

#include "json_lines.hpp"

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

class QObject;
class QProcess;

namespace rudra::app {

class AcesEngine {
public:
    enum class State { Stopped, Loading, Ready, Failed };
    struct Config {
        std::filesystem::path cli, package;
        std::string runtime = "libtorch", device = "cuda";
        int ocio_threads = 8;
        bool operator==(const Config&) const = default;
    };
    using Reply = std::function<void(const QJsonObject&)>;

    explicit AcesEngine(QObject* parent);
    ~AcesEngine();
    AcesEngine(const AcesEngine&) = delete;
    AcesEngine& operator=(const AcesEngine&) = delete;

    // Starts it with this config; a running engine with another config is
    // stopped first, one with the same config is left alone. Refused (false)
    // while a master runs on it: a render is never killed by a reload.
    bool start(const Config& c);
    bool stop();
    State state() const { return state_; }
    const Config& config() const { return config_; }
    QString status_text() const;          // for a label or the status bar
    double load_ms() const { return load_ms_; }

    // Sends one request ({"op": ...}); on_reply gets the answer (ok or not).
    // on_progress gets the progress lines of a master. Not ready, or a master
    // while another runs (one at a time: the Deliver tab, the batch, the ACES
    // window share it): an error at once.
    int request(QJsonObject req, Reply on_reply, Reply on_progress = {});
    // Stops that master (running: after the frame in hand; still queued: before it starts).
    void cancel_master(int id);
    bool busy() const { return !masters_.empty(); }   // a master is running on it
    static QString busy_text();

    // Called on every state change; removed when `owner` is destroyed.
    void listen(QObject* owner, std::function<void()> on_change);

private:
    void set_state(State s, const QString& why = {});
    void on_stdout();
    void on_finished();

    QObject* parent_;
    QObject* alive_ = nullptr;   // context of the listeners' connections: gone with the engine
    QProcess* proc_ = nullptr;
    Config config_;
    State state_ = State::Stopped;
    QString why_;
    double load_ms_ = 0;
    QString ready_device_, ready_runtime_;
    JsonLines lines_;
    QByteArray stderr_tail_;   // the engine's last words, for the exit message
    int next_id_ = 1;
    std::map<int, std::pair<Reply, Reply>> waiting_;
    std::set<int> masters_;   // the master requests in flight (at most one)
    std::map<int, std::pair<QObject*, std::function<void()>>> listeners_;
    int next_listener_ = 1;
    bool stopping_ = false;
};

// What the app would start without asking: the engine beside RUDRA (else the
// saved aces/cli), the saved package (else the first ACES v2 package in
// `roots` and aces/roots) and the saved target. Empty if there is no engine or package.
std::optional<AcesEngine::Config> aces_engine_config(const std::vector<std::filesystem::path>& roots);

}  // namespace rudra::app
