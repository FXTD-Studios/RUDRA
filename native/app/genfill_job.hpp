#pragma once
// The AI fill (RUDRA layer 3, 27 Oct 2026): `python -m rudra.genfill.run` over a learned
// master's EXR folder (a Wan2.1-VACE LoRA trained on RUDRA's own data fills only the
// fully clipped highlights and crushed shadows; no 8-bit code of the master's SDR view
// changes). The app runs it as a QProcess after a learned EXR render and reads its JSON
// lines: ready, progress (window / step), frame (index of N), done, error.
//
// Where things are (QSettings, else the first that exists of: <app>/models/genfill,
// <repo>/models/genfill, $RUDRA_GENFILL_DIR, then the training workstation's folders;
// Deliver > AI fill > Setup… sets and checks them, 28 Oct 2026 review):
//   genfill/python   the Python with torch, diffusers, peft, OpenEXR, OCIO  (<repo>/.venv/Scripts/python.exe)
//   genfill/repo     the RUDRA repository (rudra/genfill/run.py)            (found beside the app, upwards)
//   genfill/base     Wan2.1-VACE-1.3B-diffusers                              (<root>/Wan2.1-VACE-1.3B-diffusers)
//   genfill/lora     rudra_genfill.safetensors                               (<root>/rudra_genfill.safetensors)
// No Q_OBJECT (as elsewhere in the app): listeners are callbacks.

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include "json_lines.hpp"

#include <filesystem>
#include <functional>
#include <optional>

class QObject;
class QProcess;

namespace rudra::app {

struct GenFillSetup {
    std::filesystem::path python, repo, base, lora;
};

struct GenFillOptions {
    int tau_high = 250;   // 8-bit code: >= is a clipped highlight to fill (256 = none)
    int tau_low = -1;     // 8-bit code: <= is a crushed shadow to fill (-1 = none; v1.1 default off)
    int seed = 0;
    int steps = 30;
    double brightness = 0.0;  // v3: the artist's offset in stops over the estimated brightness (-3..+3)
    QString prompt_highlight, prompt_shadow;
    QString glob;         // the master folder's frames to fill ("SH010.*.exr"); "" = every EXR there
    bool operator==(const GenFillOptions&) const = default;
};

// The command line (after the Python): -m rudra.genfill.run --master ... --json.
QStringList genfill_arguments(const GenFillSetup& s, const GenFillOptions& o, const QString& master, const QString& out);

// A new folder beside the master: <master>_aifill, else _aifill_2, _3 ... (nothing replaced).
QString genfill_out_dir(const QString& master);

// The setup from QSettings and the defaults; nullopt and `why` when a part is missing.
std::optional<GenFillSetup> genfill_setup(QString* why = nullptr);

class GenFillJob {
public:
    using Event = std::function<void(const QJsonObject&)>;
    using Finished = std::function<void(bool ok, const QJsonObject& last)>;

    explicit GenFillJob(QObject* parent);
    ~GenFillJob();
    GenFillJob(const GenFillJob&) = delete;
    GenFillJob& operator=(const GenFillJob&) = delete;

    // Starts it; false (and `why`) if one is running or it cannot start.
    bool start(const GenFillSetup& s, const GenFillOptions& o, const QString& master, const QString& out,
               Event event_cb, Finished finished_cb, QString* why = nullptr);
    void cancel();                 // asks it to stop at the next step (frames written so far stay); twice = kill
    bool running() const;
    QString out_dir() const { return out_; }

private:
    void on_stdout();
    void on_finished(int code);

    QObject* parent_;
    QProcess* proc_ = nullptr;
    JsonLines lines_;
    QString out_;
    QJsonObject last_;
    QString stderr_tail_;
    Event on_event_;
    Finished on_finished_;
    bool cancelled_ = false;
};

}  // namespace rudra::app
