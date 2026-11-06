#pragma once
// The AI fill preview (4 Nov 2026): a few frames around the playhead filled
// and shown in the viewer before the whole shot is rendered.
//
// GenFillServer keeps one `python -m rudra.genfill.run --serve` running, so
// the fill model (Wan2.1-VACE-1.3B + the RUDRA LoRA, a minute or two to load)
// loads once: a setting changed and previewed again costs the fill only. One
// request at a time; each is one JSON line on its stdin, and its events come
// back tagged with its id (run.py serve()).
// No Q_OBJECT (as elsewhere in the app): connections are lambdas.

#include <QJsonObject>
#include <QString>

#include <functional>

#include "genfill_job.hpp"
#include "json_lines.hpp"

class QObject;
class QProcess;

namespace rudra::app {

// The preview's last sentence: why it may look like nothing (no clipped pixels, a low display peak).
QString genfill_preview_hint(double changed, bool region_on, double display_nits);

class GenFillServer {
public:
    using Event = std::function<void(const QJsonObject&)>;
    using Finished = std::function<void(bool ok, const QJsonObject& last)>;

    explicit GenFillServer(QObject* parent);
    ~GenFillServer();
    GenFillServer(const GenFillServer&) = delete;
    GenFillServer& operator=(const GenFillServer&) = delete;

    // Starts the server when it is not running (or runs another setup), then
    // sends `req` (its "id" is set here). Events of this request go to
    // `on_event` ("ready" too, once the model is loaded); `on_done` once, with
    // the "done" or "error" object. False and `why` while another runs.
    bool request(const GenFillSetup& s, QJsonObject req, Event on_event, Finished on_done, QString* why = nullptr);
    void cancel();                       // the request running stops at its next step
    bool busy() const noexcept { return busy_; }
    bool loaded() const noexcept { return loaded_; }   // the model is in memory
    double load_seconds() const noexcept { return load_seconds_; }

private:
    bool ensure(const GenFillSetup& s, QString* why);
    void on_stdout();
    void finish(bool ok, const QJsonObject& last);

    QObject* parent_;
    QProcess* proc_ = nullptr;
    JsonLines lines_;
    GenFillSetup setup_;
    bool loaded_ = false, busy_ = false;
    double load_seconds_ = 0;
    int next_id_ = 1, current_id_ = 0;
    Event on_event_;
    Finished on_done_;
    QString stderr_tail_;
};

}  // namespace rudra::app
