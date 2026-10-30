#include "batch_window.hpp"
#include "theme.hpp"
#include "paths.hpp"

#include <QApplication>
#include <QDir>
#include <QJsonArray>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QProgressBar>
#include <QFileInfo>
#include <QInputDialog>
#include <QTimer>
#include <QRegularExpression>
#include <QColor>
#include <cmath>
#include <QPushButton>
#include <QSpinBox>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>

#include "aces_engine.hpp"
#include "batch_queue.hpp"
#include "main_window.hpp"

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

QLabel* label(const QString& text, const QString& id, const QString& role, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    if (!id.isEmpty()) l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    return l;
}

QPushButton* button(const QString& text, const QString& id, const QString& tip, QWidget* parent) {
    auto* b = new QPushButton(text, parent);
    b->setObjectName(id);
    b->setToolTip(tip);
    return b;
}

QString kind_text(const BatchJob& j) {
    if (j.shot.kind == "movie") return QStringLiteral("Movie");
    if (j.shot.kind == "sequence") return QStringLiteral("Sequence \u00b7 %1 frames").arg(j.shot.sources.size());
    return QStringLiteral("Still");
}

QString status_text(const BatchJob& j) {
    if (j.status == "running")
        return j.total > 0 ? QStringLiteral("Rendering %1 / %2").arg(j.done).arg(j.total) : QStringLiteral("Rendering %1").arg(j.done);
    if (j.status == "filling") return QStringLiteral("AI fill\u2026 (the master is written)");
    if (j.status == "done")
        return QStringLiteral("Done \u00b7 %1 frame(s)").arg(j.done) + (j.note.isEmpty() ? QString() : " \u00b7 " + j.note);
    if (j.status == "failed") return QStringLiteral("Failed: ") + j.error;
    if (j.status == "stopped") return QStringLiteral("Stopped after %1 frame(s)").arg(j.done);
    return j.note.isEmpty() ? QStringLiteral("Waiting") : QStringLiteral("Waiting (") + j.note + ")";
}

}  // namespace

BatchWindow::BatchWindow(MainWindow* w) : QDialog(w), w_(w) {
    setObjectName("batchWindow");
    setWindowTitle("Batch queue");
    setModal(false);
    setAcceptDrops(true);
    resize(980, 560);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(22, 18, 22, 14);
    v->setSpacing(10);
    v->addWidget(label("Batch queue", "batchTitle", "title", this));
    sub_ = label("", "batchSub", "note", this);
    sub_->setWordWrap(true);
    v->addWidget(sub_);

    // Where new shots render, and how to add them.
    auto* top = new QWidget(this);
    auto* th = new QHBoxLayout(top);
    th->setContentsMargins(0, 0, 0, 0);
    th->setSpacing(8);
    th->addWidget(label("Render folder", {}, "note", top));
    dir_ = new QLineEdit(top);
    dir_->setObjectName("batchDir");
    dir_->setPlaceholderText("C:\\Renders");
    dir_->setToolTip("Where shots added from now on render: a new folder per shot, named after it");
    th->addWidget(dir_, 1);
    auto* browse = button("Browse\u2026", "batchBrowse", "Choose the render folder", top);
    connect(browse, &QPushButton::clicked, this, [this] {
        const QString d = QFileDialog::getExistingDirectory(this, "Render folder", dir_->text());
        if (!d.isEmpty()) dir_->setText(QDir::toNativeSeparators(d));
    });
    th->addWidget(browse);
    v->addWidget(top);

    auto* adds = new QWidget(this);
    auto* ah = new QHBoxLayout(adds);
    ah->setContentsMargins(0, 0, 0, 0);
    ah->setSpacing(8);
    auto* add_files = button("Add files\u2026", "batchAddFiles", "Movies, stills or the frames of sequences (numbered names are grouped)", adds);
    auto* add_folder = button("Add folder\u2026", "batchAddFolder", "A folder: its movies, sequences and stills, and its sub-folders' (one level)", adds);
    auto* add_current = button("Add current shot", "batchAddCurrent", "The shot on screen, with the Deliver tab's settings (Ctrl+B)", adds);
    auto* edl = button("Import EDL…", "batchImportEdl",
                       "A CMX 3600 EDL from Resolve, Premiere or Avid: each video event becomes a shot of only the frames "
                       "it uses (with handles), matched to the movies in a folder by clip name, reel and timecode", adds);
    connect(edl, &QPushButton::clicked, this, [this] {
        const QString file = QFileDialog::getOpenFileName(this, "Import an EDL", {}, "EDL (*.edl);;All files (*)");
        if (file.isEmpty()) return;
        const QString media = QFileDialog::getExistingDirectory(this, "The source movies (searched one folder deep)",
                                                                QFileInfo(file).absolutePath());
        if (media.isEmpty()) return;
        bool ok = false;
        const int handles = QInputDialog::getInt(this, "Handles", "Frames before and after each edit:", 12, 0, 1000, 1, &ok);
        if (!ok) return;
        if (dir_->text().trimmed().isEmpty()) {
            status_->setText("Choose a render folder first.");
            return;
        }
        QString report;
        w_->batch_import_edl(file, media, handles, dir_->text().trimmed(), &report);
        status_->setText(report.section('\n', 0, 0));
        if (report.contains('\n')) status_->setToolTip(report);
    });
    auto* apply = button("Apply Deliver settings", "batchApply",
                         "The Deliver tab's output, format, colour, encoding and bit depth, the learned trims (and this "
                         "render folder) on the selected shots, or on every waiting shot when none is selected", adds);
    connect(add_files, &QPushButton::clicked, this, [this] {
        const QStringList f = QFileDialog::getOpenFileNames(
            this, "Add to the batch", {},
            "Movies and stills (*.mov *.mp4 *.mkv *.mxf *.avi *.m4v *.webm *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp *.dpx);;All files (*)");
        add_paths(f);
    });
    connect(add_folder, &QPushButton::clicked, this, [this] {
        const QString d = QFileDialog::getExistingDirectory(this, "Add a folder to the batch");
        if (!d.isEmpty()) add_paths({d});
    });
    connect(add_current, &QPushButton::clicked, this, [this] {
        QString why;
        if (!w_->batch_add_current(&why, dir_->text().trimmed())) status_->setText(why);
    });
    connect(apply, &QPushButton::clicked, this, [this] {
        const auto& t = w_->aces_trims();
        w_->batch_queue().apply(selected_ids(), w_->deliver_choice(), dir_->text().trimmed(), w_->render_start(),
                                BatchTrims{t.strength, t.saturation, t.peak_nits, t.exposure_ev, t.contrast});
    });
    for (auto* b : {add_files, add_folder, add_current, edl, apply}) ah->addWidget(b);
    ah->addStretch(1);
    v->addWidget(adds);

    list_ = new QTreeWidget(this);
    list_->setObjectName("batchList");
    list_->setColumnCount(8);
    list_->setHeaderLabels({"#", "Shot", "Kind", "Range", "Output", "Name", "MaxCLL / FALL", "Status"});
    list_->headerItem()->setToolTip(3, "A movie's frames to render, first-last (0-based); empty for all. Double-click to edit");
    list_->headerItem()->setToolTip(5, "The render's name: its folder and files. Double-click to edit");
    list_->headerItem()->setToolTip(6, "HDR10 MaxCLL / MaxFALL of the rendered shot, in display nits, measured on its ACES 2.0 "
                                       "HDR 1000 nits (P3 D65) PQ output: the numbers for the delivery's metadata. "
                                       "A shot far above the others stands out");
    list_->setRootIsDecorated(false);
    list_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    list_->setAlternatingRowColors(true);
    list_->header()->setStretchLastSection(true);
    list_->setColumnWidth(0, 34);
    list_->setColumnWidth(1, 170);
    list_->setColumnWidth(2, 120);
    list_->setColumnWidth(3, 90);
    list_->setColumnWidth(4, 270);
    list_->setColumnWidth(5, 120);
    list_->setColumnWidth(6, 110);
    connect(list_, &QTreeWidget::itemChanged, this, [this](QTreeWidgetItem* it, int col) {
        if (filling_) return;
        const QString id = it->data(0, Qt::UserRole).toString();
        if (col == 5) {
            QString name = it->text(5);
            if (name.startsWith(QStringLiteral("\u26a0 "))) name = name.mid(2);   // the EDL warning mark is not the name
            w_->batch_queue().rename(id, name);
        }
        else if (col != 3) QTimer::singleShot(0, this, [this] { refresh(); });   // only the range and the name edit
        if (col == 3) {
            const QString t = it->text(3).trimmed();
            const QStringList p = t.split(QRegularExpression("\\s*[-:\u2013]\\s*"));
            bool a = false, b = false;
            const int first = p.size() == 2 ? p[0].toInt(&a) : -1, last = p.size() == 2 ? p[1].toInt(&b) : -1;
            if (t.isEmpty() || t.compare("all", Qt::CaseInsensitive) == 0) w_->batch_queue().set_range(id, -1, -1);
            else if (!(a && b && w_->batch_queue().set_range(id, first, last))) {
                status_->setText("A range is first-last, for a movie (e.g. 120-480)");
                refresh();
            }
        }
    });
    v->addWidget(list_, 1);

    auto* edits = new QWidget(this);
    auto* eh = new QHBoxLayout(edits);
    eh->setContentsMargins(0, 0, 0, 0);
    eh->setSpacing(8);
    auto* up = button("Up", "batchUp", "Render the selected shot earlier", edits);
    auto* down = button("Down", "batchDown", "Render the selected shot later", edits);
    auto* remove = button("Remove", "batchRemove", "Take the selected shots out of the queue (nothing on disk is deleted)", edits);
    auto* retry = button("Retry", "batchRetry", "Put failed or stopped shots (the selected ones, or all) back in the queue", edits);
    auto* clear = button("Clear done", "batchClear", "Take the finished shots off the list", edits);
    connect(up, &QPushButton::clicked, this, [this] {
        for (const auto& id : selected_ids()) w_->batch_queue().move(id, -1);
    });
    connect(down, &QPushButton::clicked, this, [this] {
        auto ids = selected_ids();
        for (auto it = ids.rbegin(); it != ids.rend(); ++it) w_->batch_queue().move(*it, +1);
    });
    connect(remove, &QPushButton::clicked, this, [this] { w_->batch_queue().remove(selected_ids()); });
    connect(retry, &QPushButton::clicked, this, [this] { w_->batch_queue().retry(selected_ids()); });
    connect(clear, &QPushButton::clicked, this, [this] { w_->batch_queue().clear_finished(); });
    // The AI fill (layer 3) after a shot's master: EXR stills and sequences only.
    auto* ai_on = button("AI fill on", "batchAiOn",
                         "Run the AI fill after the selected shots' masters (every waiting shot when none is selected); "
                         "EXR stills and sequences only, written to <name>_aifill", edits);
    auto* ai_off = button("AI fill off", "batchAiOff", "No AI fill for the selected shots (every waiting shot when none is selected)", edits);
    connect(ai_on, &QPushButton::clicked, this, [this] { w_->batch_queue().set_ai_fill(selected_ids(), true); });
    connect(ai_off, &QPushButton::clicked, this, [this] { w_->batch_queue().set_ai_fill(selected_ids(), false); });
    for (auto* b : {up, down, remove, retry, clear}) eh->addWidget(b);
    eh->addSpacing(16);
    eh->addWidget(ai_on);
    eh->addWidget(ai_off);
    eh->addStretch(1);
    v->addWidget(edits);

    auto* foot = new QWidget(this);
    auto* fh = new QHBoxLayout(foot);
    fh->setContentsMargins(0, 6, 0, 0);
    fh->setSpacing(10);
    progress_ = new QProgressBar(foot);
    progress_->setObjectName("batchProgress");
    progress_->setTextVisible(true);
    fh->addWidget(progress_, 1);
    status_ = label("", "batchStatus", "note", foot);
    fh->addWidget(status_, 2);
    stop_now_ = button("Stop now", "batchStopNow", "Stop the shot in hand after its current frame, and the batch", foot);
    connect(stop_now_, &QPushButton::clicked, this, [this] { w_->batch_runner().stop_now(); });
    start_ = button("Start", "batchStart", "Render every waiting shot, one after another", foot);
    start_->setProperty("role", "primary");
    connect(start_, &QPushButton::clicked, this, [this] {
        auto& r = w_->batch_runner();
        if (r.running()) {
            r.stop_after_current();
            return;
        }
        QString why;
        if (!w_->batch_start(&why)) status_->setText(why);
    });
    fh->addWidget(stop_now_);
    fh->addWidget(start_);
    v->addWidget(foot);
    refresh();
    if (const QString warn = w_->batch_queue().load_warning(); !warn.isEmpty()) status_->setText(warn);
}

std::vector<QString> BatchWindow::selected_ids() const {
    std::vector<QString> ids;
    for (int i = 0; i < list_->topLevelItemCount(); ++i)
        if (auto* it = list_->topLevelItem(i); it->isSelected()) ids.push_back(it->data(0, Qt::UserRole).toString());
    return ids;
}

void BatchWindow::add_paths(const QStringList& paths) {
    if (paths.isEmpty()) return;
    std::vector<fs::path> p;
    for (const auto& s : paths) p.push_back(to_path(s));
    const auto shots = batch_shots_from(p);
    if (shots.empty()) {
        status_->setText("No movies, sequences or stills there.");
        return;
    }
    QString dir = dir_->text().trimmed();
    if (dir.isEmpty()) {
        status_->setText("Choose a render folder first.");
        dir_->setFocus();
        return;
    }
    const auto& t = w_->aces_trims();
    w_->batch_queue().add(shots, w_->deliver_choice(), dir, w_->render_start(), {t.strength, t.saturation, t.peak_nits});
    status_->setText(QStringLiteral("Added %1 shot(s)").arg(shots.size()));
}

void BatchWindow::refresh() {
    auto& q = w_->batch_queue();
    auto& r = w_->batch_runner();
    if (dir_->text().isEmpty()) dir_->setText(w_->render_dir());
    const auto keep = selected_ids();
    // The rendered shots' MaxCLL: one far above the median stands out.
    std::vector<double> clls;
    for (const auto& j : q.jobs())
        if (j.maxcll > 0) clls.push_back(j.maxcll);
    std::sort(clls.begin(), clls.end());
    const std::size_t n_cll = clls.size();
    const double max_cll = n_cll ? clls.back() : 0.0, median_cll = n_cll ? clls[n_cll / 2] : 0.0;
    filling_ = true;
    // The rows are updated in place when the queue still holds the same shots in the same
    // order (a run's progress): a name or range being edited survives (28 Oct 2026 review).
    bool same = list_->topLevelItemCount() == int(q.jobs().size());
    for (int i = 0; same && i < list_->topLevelItemCount(); ++i)
        same = list_->topLevelItem(i)->data(0, Qt::UserRole).toString() == q.jobs()[std::size_t(i)].id;
    if (!same) list_->clear();
    // An inline editor has the focus inside the list while a name or range is typed.
    QWidget* focus = QApplication::focusWidget();
    const bool editing = focus && focus != list_ && list_->isAncestorOf(focus);
    int n = 0, done = 0, waiting = 0, failed = 0;
    for (const auto& j : q.jobs()) {
        ++n;
        done += j.status == "done";
        waiting += j.status == "pending";
        failed += j.status == "failed" || j.status == "stopped";
        const QString range = j.first >= 0 ? QStringLiteral("%1-%2").arg(j.first).arg(j.last)
                              : j.shot.kind == "movie" ? QStringLiteral("all") : QString();
        const QString light = j.maxcll > 0 ? QStringLiteral("%1 / %2").arg(std::lround(std::ceil(j.maxcll))).arg(std::lround(std::ceil(j.maxfall))) : QString();
        const QJsonArray ew = j.edl.value("warnings").toArray();
        const QStringList texts = {QString::number(n), j.shot.title, kind_text(j), range, batch_output_text(j),
                                   ew.isEmpty() ? j.name : QStringLiteral("\u26a0 ") + j.name, light, status_text(j)};
        QTreeWidgetItem* it = same ? list_->topLevelItem(n - 1) : new QTreeWidgetItem(list_, texts);
        if (same) {
            if (editing && it == list_->currentItem()) continue;   // the row being typed into keeps its editor
            for (int c = 0; c < texts.size(); ++c)
                if (it->text(c) != texts[c]) it->setText(c, texts[c]);
        }
        if (j.maxcll > 0 && max_cll > 0 && j.maxcll >= 2.0 * median_cll && n_cll >= 3) {
            it->setForeground(6, theme_colour("gold"));
            it->setToolTip(6, QStringLiteral("More than twice the median MaxCLL of the rendered shots (%1 nits)").arg(std::lround(median_cll)));
        } else {
            it->setForeground(6, list_->palette().text());
            it->setToolTip(6, {});
        }
        // The EDL's warnings (a retime, short handles) stay on the shot: the conform needs them.
        if (!ew.isEmpty()) {
            QStringList w;
            for (const auto& x : ew) w << x.toString();
            it->setForeground(5, theme_colour("gold"));
            it->setToolTip(5, QStringLiteral("EDL event %1, record %2-%3\n").arg(j.edl.value("event").toInt())
                                  .arg(j.edl.value("rec_in").toString(), j.edl.value("rec_out").toString()) + w.join("\n"));
        } else if (!j.edl.isEmpty()) {
            it->setToolTip(5, QStringLiteral("EDL event %1, record %2-%3").arg(j.edl.value("event").toInt())
                                  .arg(j.edl.value("rec_in").toString(), j.edl.value("rec_out").toString()));
        }
        it->setData(0, Qt::UserRole, j.id);
        it->setToolTip(1, j.shot.sources.empty() ? QString() : from_path(j.shot.sources.front()));
        it->setToolTip(7, j.out.isEmpty() ? status_text(j) : QDir::toNativeSeparators(j.out));
        it->setToolTip(4, QDir::toNativeSeparators(j.render_dir));
        it->setFlags(j.status != "running" ? it->flags() | Qt::ItemIsEditable : it->flags() & ~Qt::ItemIsEditable);
        if (!same && std::find(keep.begin(), keep.end(), j.id) != keep.end()) it->setSelected(true);
    }
    filling_ = false;
    AcesEngine* e = w_->aces_engine();
    const bool ready = e && e->state() == AcesEngine::State::Ready;
    sub_->setText(QStringLiteral("%1 shot(s) \u00b7 %2 waiting \u00b7 %3 done%4 \u00b7 %5")
                      .arg(n)
                      .arg(waiting)
                      .arg(done)
                      .arg(failed ? QStringLiteral(" \u00b7 %1 failed or stopped").arg(failed) : QString())
                      .arg(ready ? e->status_text() : QStringLiteral("the learned ACES 2.0 model is not loaded")));
    const int total = n ? n : 1;
    progress_->setRange(0, total);
    progress_->setValue(done);
    progress_->setFormat(QStringLiteral("%1 / %2 shots").arg(done).arg(n));
    if (r.running()) {
        start_->setText(r.stopping() ? "Stopping\u2026" : "Stop after this shot");
        start_->setEnabled(!r.stopping());
        if (const BatchJob* j = q.find(r.current())) status_->setText(j->shot.title + ": " + status_text(*j));
        was_running_ = true;
    } else {
        if (was_running_)   // the run just ended: what it did
            status_->setText(QStringLiteral("Batch ended: %1 done, %2 waiting%3")
                                 .arg(done)
                                 .arg(waiting)
                                 .arg(failed ? QStringLiteral(", %1 failed or stopped").arg(failed) : QString()));
        was_running_ = false;
        start_->setText("Start");
        start_->setEnabled(ready && waiting > 0);
        start_->setToolTip(ready ? QStringLiteral("Render every waiting shot, one after another")
                                 : QStringLiteral("Load the learned ACES 2.0 model first"));
    }
    stop_now_->setEnabled(r.running());
}

void BatchWindow::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}

void BatchWindow::dropEvent(QDropEvent* e) {
    QStringList paths;
    for (const QUrl& u : e->mimeData()->urls())
        if (u.isLocalFile()) paths << u.toLocalFile();
    add_paths(paths);
    e->acceptProposedAction();
}

}  // namespace rudra::app
