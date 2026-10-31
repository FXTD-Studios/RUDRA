// The window's side of the batch queue (28 Oct 2026, batch_queue.hpp,
// batch_window.hpp): the queue file, the runner on the kept-loaded engine, and
// adding the shot on screen with the Deliver tab's choice.

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QLineEdit>
#include <QSettings>
#include <QSpinBox>
#include <QStandardPaths>
#include <QStatusBar>

#include <algorithm>

#include "aces_engine.hpp"
#include "paths.hpp"
#include "batch_window.hpp"
#include "edl.hpp"

#include "main_window.hpp"

namespace fs = std::filesystem;

namespace rudra::app {

BatchQueue& MainWindow::batch_queue() {
    if (!batch_) {
        QString file = QSettings().value("batch/file").toString();
        if (file.isEmpty())
            file = QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation)).filePath("batch_queue.json");
        batch_ = std::make_unique<BatchQueue>(to_path(file));
        batch_->changed = [this] { batch_changed(); };
        if (!batch_->load_warning().isEmpty()) log("Batch: " + batch_->load_warning());
    }
    return *batch_;
}

BatchRunner& MainWindow::batch_runner() {
    if (!batch_runner_) {
        batch_runner_ = std::make_unique<BatchRunner>(batch_queue(), [this] { return aces_engine(); });
        batch_runner_->changed = [this] { batch_changed(); };
        batch_runner_->log = [this](const QString& line) { log("Batch: " + line); };
        // A shot's AI fill after its master (29 Oct 2026): the window's genfill job, into <out>_aifill.
        batch_runner_->fill = [this](const BatchJob& j, QString* why, std::function<void(bool, const QString&)> done) {
            return start_genfill_with(j.out, j.choice.ai, j.out,
                                      [done](bool ok, const QString& text) { done(ok, text); }, why);
        };
    }
    return *batch_runner_;
}

void MainWindow::batch_changed() {
    if (batch_window_) static_cast<BatchWindow*>(batch_window_.data())->refresh();
}

void MainWindow::open_batch() {
    batch_runner();   // the queue and its runner
    if (!batch_window_) batch_window_ = new BatchWindow(this);
    static_cast<BatchWindow*>(batch_window_.data())->refresh();
    batch_window_->show();
    batch_window_->raise();
}

QString MainWindow::render_dir() const {
    auto* d = findChild<QLineEdit*>("renderDir");
    return d ? d->text().trimmed() : QString();
}

int MainWindow::render_start() const {
    auto* s = findChild<QSpinBox*>("renderStart");
    return s ? s->value() : 1001;
}

bool MainWindow::batch_add_current(QString* why, const QString& dir) {
    auto refuse = [why](const QString& w) {
        if (why) *why = w;
        return false;
    };
    if (frames_.empty()) return refuse(QStringLiteral("Open a shot first."));
    const QString folder = dir.isEmpty() ? render_dir() : dir;
    if (folder.isEmpty()) return refuse(QStringLiteral("Choose a render folder first (Deliver tab)."));
    BatchShot shot;
    shot.title = shot_title();
    if (shot_is_video() && !shot_sources_.empty()) {
        shot.kind = "movie";
        shot.sources = {shot_sources_.front()};
    } else {
        shot.kind = frames_.size() > 1 ? "sequence" : "still";
        shot.sources = frames_;
    }
    if (deliver_.output == "still" && shot.kind != "movie") {   // the frame on screen, as the Deliver tab renders it
        shot.kind = "still";
        shot.sources = {frames_[std::size_t(std::clamp(current_, 0, int(frames_.size()) - 1))]};
    }
    auto& q = batch_queue();
    const auto ids = q.add({shot}, deliver_, folder, render_start(),
                           {aces_trims_.strength, aces_trims_.saturation, aces_trims_.peak_nits, aces_trims_.exposure_ev, aces_trims_.contrast});
    if (BatchJob* j = q.find(ids.front())) {   // the conform's timecode and reel, as the Deliver tab has them
        if (deliver_.output == "still" && shot.kind == "movie") j->first = j->last = current_;
        j->timecode = render_timecode();
        j->reel = render_reel();
        j->source_display = aces_source_display_;
        q.touch();
    }
    // The Deliver tab's name, when there is one, is the render's name.
    if (const QString name = findChild<QLineEdit*>("renderName")->text().trimmed(); !name.isEmpty() && name != "master")
        q.rename(ids.front(), name);
    log(QStringLiteral("Batch: added %1 (%2 in the queue)").arg(shot.title).arg(q.jobs().size()));
    if (statusBar()) statusBar()->showMessage(QStringLiteral("Added %1 to the batch queue").arg(shot.title), 5000);
    return true;
}

int MainWindow::batch_import_edl(const QString& edl, const QString& media_folder, int handles, const QString& render_dir,
                                 QString* report) {
    QFile f(edl);
    if (!f.open(QIODevice::ReadOnly)) {
        if (report) *report = QStringLiteral("Cannot read ") + edl;
        return 0;
    }
    QString title;
    const auto events = parse_edl(QString::fromUtf8(f.readAll()), &title);
    const auto media = edl_media_in(to_path(media_folder));
    const EdlMatch m = edl_match(events, media, handles);
    // A range renders as frames (or the MOV the Deliver tab chose), never a single still.
    DeliverChoice c = deliver_;
    if (c.output == "still") c.output = "sequence";
    std::vector<BatchShot> shots;
    for (const auto& s : m.shots) shots.push_back(s.shot);
    auto& q = batch_queue();
    const auto ids = q.add(shots, c, render_dir, render_start(), {aces_trims_.strength, aces_trims_.saturation, aces_trims_.peak_nits, aces_trims_.exposure_ev, aces_trims_.contrast});
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (BatchJob* j = q.find(ids[i])) {
            j->first = m.shots[i].first;
            j->last = m.shots[i].last;
            j->name = m.shots[i].name;
            j->source_display = aces_source_display_;
            const EdlShot& es = m.shots[i];
            j->edl = QJsonObject{{"event", es.event}, {"reel", es.name.section('_', 1)}, {"rec_in", es.rec_in},
                                 {"rec_out", es.rec_out}, {"src_in", es.src_in}, {"edl", title.isEmpty() ? QFileInfo(edl).fileName() : title},
                                 {"warnings", QJsonArray::fromStringList(es.warnings)}};
            if (!es.warnings.isEmpty()) j->note = es.warnings.join(" · ");
        }
    }
    if (!ids.empty()) q.touch();
    int video_events = 0;
    for (const auto& e : events) video_events += e.video() ? 1 : 0;
    const QString text = QStringLiteral("EDL %1: %2 of %3 video events added (%4 handles), %5 movies searched")
                             .arg(title.isEmpty() ? QFileInfo(edl).fileName() : title)
                             .arg(m.shots.size())
                             .arg(video_events)
                             .arg(handles)
                             .arg(media.size()) +
                         (m.unmatched.isEmpty() ? QString() : QStringLiteral("; not matched: ") + m.unmatched.join("; ")) +
                         (m.warnings.isEmpty() ? QString() : QStringLiteral("; check: ") + m.warnings.join("; "));
    log("Batch: " + text);
    if (report)
        *report = text + (m.unmatched.isEmpty() ? QString() : "\n" + m.unmatched.join("\n")) +
                  (m.warnings.isEmpty() ? QString() : "\n" + m.warnings.join("\n"));
    return int(m.shots.size());
}

bool MainWindow::batch_start(QString* why) {
    if (aces_delivering() || genfill_running()) {
        if (why) *why = QStringLiteral("A Deliver render is running; wait for it or stop it.");
        return false;
    }
    return batch_runner().start(why);
}

}  // namespace rudra::app
