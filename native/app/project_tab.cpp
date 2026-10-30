// Projects, autosave, updates and help (split from main_window.cpp, 28 Oct 2026 review: one file per part of the window).
#include "main_window.hpp"
#include "batch_queue.hpp"
#include "rudra/deliver/video_encode.hpp"
#include "paths.hpp"
#include "aces_engine.hpp"
#include "rudra/engine/headroom_export.hpp"
#include "rudra/video/headroom_render.hpp"
#include "startup.hpp"
#include "getting_started.hpp"
#include "updates.hpp"
#include "playback.hpp"

#include "rudra/video/ffmpeg_check.hpp"
#include "rudra/deliver/sequence_encode.hpp"

#include <QActionGroup>
#include <QApplication>
#include <QDialog>
#include <QFileDialog>
#include <QInputDialog>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QGridLayout>
#include <QKeySequence>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QPointer>
#include <QStatusBar>
#include <QFrame>
#include <QKeyEvent>
#include <QClipboard>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QCloseEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QUrl>
#include <QLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QLocale>
#include <QSpinBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSettings>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QStyle>
#include <QVBoxLayout>

#include "export_sheet.hpp"
#include "model_dialogs.hpp"
#include "region_editor.hpp"
#include "scope_widgets.hpp"
#include "source_panel.hpp"
#include "widgets.hpp"

#include <algorithm>
#include <fstream>
#include <cctype>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <thread>

#include "rudra/core/baseline.hpp"
#include "rudra/core/clip_fill.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/core/video_predict.hpp"
#include "rudra/core/reference_fit.hpp"
#include "rudra/engine/reference_image.hpp"
#include "rudra/core/copy_texts.hpp"
#include "rudra/core/readouts.hpp"
#include "rudra/engine/actions.hpp"
#include "rudra/engine/frame_engine.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/infer/tiler.hpp"
#include "rudra/media/sequence.hpp"

#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif
#ifdef RUDRA_HAVE_STILL_DECODE
#ifdef RUDRA_HAVE_STILL_DECODE
#include "rudra/media/still.hpp"
#endif
#endif

namespace rudra::app {

namespace {

QString qs(std::string_view s) { return QString::fromUtf8(s.data(), qsizetype(s.size())); }

}  // namespace

// ---------------------------------------------------------------------------
// Projects, autosave, updates and help (product items 3 to 5, 10 Oct 2026)
// ---------------------------------------------------------------------------

Project MainWindow::current_project() const {
    Project p = project_from_session(session_);
    if (pending_shot_) {   // waiting for its model: still the shot
        p.source_kind = pending_shot_->kind;
        p.sources = pending_shot_->sources;
        p.frame = pending_shot_->frame;
    } else {
        p.source_kind = shot_kind_;
        p.sources = shot_sources_;
        p.frame = frames_.empty() ? 0 : current_;
    }
    if (project_model_) {   // loading, or queued: still the project's model
        p.package = project_model_->package;
        p.backend = project_model_->backend;
    } else {
        p.package = model_package();
        p.backend = backend_choice_ ? backend_choice_->key() : std::string();
    }
    p.use_model = use_model_;
    // The window has no grain control: the legacy path always settles grain
    // at the anchor's knee, the headroom preview never does (it needs the
    // canonical SDR v1 cannot represent). The project records what was shown.
    if (p.grade.headroom_policy != HeadroomPolicy::Disabled) p.settle_grain = false;
    p.app_version = QCoreApplication::applicationVersion().toStdString();
    return p;
}

void MainWindow::mark_dirty() {
    if (restoring_) return;
    if (!dirty_) {
        dirty_ = true;
        update_title();
    }
    if (!autosave_path_.empty()) autosave_timer_.start();
}

void MainWindow::update_title() {
    if (project_path_.empty()) {
        setWindowTitle("RUDRA");
        setWindowModified(false);
        return;
    }
    // [*] is where Qt puts the unsaved mark (and macOS the dot in the close button).
    setWindowTitle(from_path(project_path_.filename()) + QStringLiteral("[*] — RUDRA"));
    setWindowModified(dirty_);
}

bool MainWindow::save_project_to(const std::filesystem::path& path) {
    auto r = save_project(path, current_project());
    if (!r) {
        problem_ = "The project was not saved: " + qs(r.error().message);
        log(problem_);
        return false;
    }
    project_path_ = *r;
    dirty_ = false;
    update_title();
    remember_source(from_path(*r));
    log("saved project " + from_path(*r));
    return true;
}

bool MainWindow::autosave_now() {
    autosave_timer_.stop();
    if (autosave_path_.empty()) return false;
    auto r = save_project(autosave_path_, current_project());
    if (!r) {
        log("autosave failed: " + qs(r.error().message));
        return false;
    }
    return true;
}

void MainWindow::set_autosave_path(const std::filesystem::path& file, int delay_ms) {
    autosave_path_ = file;
    autosave_timer_.setInterval(delay_ms);
    reopen_path_.clear();
    std::error_code ec;
    if (!file.empty() && std::filesystem::is_regular_file(file, ec)) {
        // What the last run left: kept aside before this run's first change
        // writes over it, and offered as File > Reopen last session.
        const std::filesystem::path prev = file.parent_path() / ("previous-session" + std::string(kProjectSuffix));
        std::filesystem::rename(file, prev, ec);
        if (!ec) {
            const auto masks = project_masks_path(file), prev_masks = project_masks_path(prev);
            std::filesystem::remove(prev_masks, ec);
            if (std::filesystem::is_regular_file(masks, ec)) std::filesystem::rename(masks, prev_masks, ec);
            reopen_path_ = prev;
        }
    }
    if (reopen_path_.empty() && !file.empty()) {
        const auto prev = file.parent_path() / ("previous-session" + std::string(kProjectSuffix));
        if (std::filesystem::is_regular_file(prev, ec)) reopen_path_ = prev;
    }
    if (reopen_path_.empty()) pending_["reopen-session"] = "No earlier session was saved.";
    else {
        pending_.erase("reopen-session");
        log("the last session was saved: File > Reopen last session opens it");
    }
    refresh_enabled();
}

void MainWindow::open_shot(const std::string& kind, const std::vector<std::filesystem::path>& sources, int frame) {
    std::vector<std::filesystem::path> present;
    std::error_code ec;
    for (const auto& s : sources) {
        if (std::filesystem::exists(s, ec)) present.push_back(s);
        else log("the project's footage is missing: " + from_path(s));
    }
    if (present.empty()) return;
    if (needs_model()) {
        // Opened when a model is in (the project's, loading, or one opened next).
        pending_shot_ = PendingShot{kind, present, frame};
        if (!loading_model_) log("the shot opens when a model package is open (File > Model packages…)");
        return;
    }
    if (kind == "folder" || kind == "movie") {
        open_source(from_path(present.front()), kind == "folder");
    } else {
        close_frames();
        add_files(present);
    }
    if (frame > 0 && frame < int(frames_.size())) step_to(frame);
}

bool MainWindow::open_project(const std::filesystem::path& path, bool adopt) {
    // Only a capable window reads a headroom project; others refuse it before any change.
    auto p = load_project(path, headroom_capable_ ? ProjectReadSupport::HeadroomV1 : ProjectReadSupport::LegacyOnly);
    if (!p) {
        problem_ = "The project was not opened: " + qs(p.error().message);
        log(problem_);
        return false;
    }
    // Changes not yet autosaved go there now, so what was on screen stays recoverable.
    if (autosave_timer_.isActive()) autosave_now();
    restoring_ = true;
    for (const auto& note : p->notes) log(QString::fromStdString(note));
    // The model: the project's when it is another one and still there. A load
    // already running finishes first; the project's follows it.
    project_model_.reset();
    // Its model only matters when the project runs one.
    if (p->use_model && !p->package.empty() && p->package != model_package()) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(p->package / "manifest.json", ec)) {
            project_model_ = ProjectModel{p->package, p->backend};
            use_project_model();
        } else {
            log("the project's model package is missing (" + from_path(p->package) + "); " +
                (backend_ ? QStringLiteral("the model in use stays") : QStringLiteral("open one to see the shot")));
        }
    }
    // Before the shot: it decides whether the shot waits for a model. The
    // project's choice, not this user's default for the next bare start.
    set_use_model(p->use_model, false);
    // The grade, then the shot: a new shot keeps the session's grade.
    if (auto loaded = session_.load_state(p->grade, p->peak_ev, p->anchor, p->carry_chroma,
                                          p->anchor_knee, p->container); !loaded) {
        restoring_ = false;
        problem_ = qs(loaded.error().message);
        log(problem_);
        return false;
    }
    pending_shot_.reset();
    close_frames();
    if (!p->source_kind.empty()) open_shot(p->source_kind, p->sources, p->frame);
    restoring_ = false;
    project_path_ = adopt ? path : std::filesystem::path();
    dirty_ = !adopt;   // a reopened session is unsaved work until it is saved as a project
    update_title();
    if (adopt) remember_source(from_path(path));
    log((adopt ? "opened project " : "reopened the last session from ") + from_path(path));
    if (!autosave_path_.empty() && !adopt) autosave_timer_.start();
    return true;
}

QString MainWindow::default_updates_feed() {
    return QStringLiteral("https://api.github.com/repos/FXTD-Studios/RUDRA/releases?per_page=20");
}

void MainWindow::check_for_updates(bool interactive, std::function<void(QString)> done) {
    if (!net_) net_ = new QNetworkAccessManager(this);
    const QString current = QCoreApplication::applicationVersion();
    QNetworkRequest req{QUrl(updates_feed_)};
    req.setHeader(QNetworkRequest::UserAgentHeader, QStringLiteral("RUDRA/") + current);
    req.setRawHeader("Accept", "application/vnd.github+json");
    req.setTransferTimeout(15000);
    QNetworkReply* reply = net_->get(req);
    QPointer<MainWindow> self(this);
    connect(reply, &QNetworkReply::finished, this, [self, reply, interactive, done, current] {
        reply->deleteLater();
        if (!self) return;
        QString line, url;
        bool newer = false;
        if (reply->error() != QNetworkReply::NoError) {
            line = "update check failed: " + reply->errorString();
        } else {
            const QByteArray body = reply->readAll();
            const std::string cur = current.toStdString();
            // A beta follows the betas; a release only releases.
            const auto latest = newest_release(body.toStdString(), is_prerelease(cur));
            if (!latest) line = "update check: no release found";
            else if (cur.empty() || compare_versions(latest->version, cur) > 0) {
                newer = true;
                url = QString::fromStdString(latest->url);
                line = "RUDRA " + QString::fromStdString(latest->version) + " is available (this is " +
                       (current.isEmpty() ? QStringLiteral("an unversioned build") : current) + "): " + url;
            } else {
                line = "RUDRA is up to date (" + current + ")";
            }
        }
        self->update_url_ = url;
        if (reply->error() == QNetworkReply::NoError)
            self->update_state_ = newer ? QStringLiteral("RUDRA %1 is available").arg(url.section('/', -1)) : QStringLiteral("Up to date");
        self->log(line);
        if (newer) self->statusBar()->showMessage(line);
        if (interactive) {
            if (newer) {
                const auto b = QMessageBox::information(self, "Updates", line, QMessageBox::Open | QMessageBox::Close,
                                                        QMessageBox::Open);
                if (b == QMessageBox::Open) QDesktopServices::openUrl(QUrl(url));
            } else {
                QMessageBox::information(self, "Updates", line);
            }
        }
        if (done) done(line);
    });
}

void MainWindow::maybe_check_for_updates() {
    if (qEnvironmentVariableIsSet("RUDRA_NO_UPDATE_CHECK")) return;
    QSettings st;
    if (!st.value("updates/auto", true).toBool()) return;
    const QDateTime last = st.value("updates/last").toDateTime();
    const QDateTime now = QDateTime::currentDateTimeUtc();
    if (last.isValid() && last.secsTo(now) < 24 * 3600) return;
    st.setValue("updates/last", now);
    check_for_updates(false);
}

QString MainWindow::user_guide_url() {
    return QStringLiteral("https://github.com/FXTD-Studios/RUDRA/blob/native/docs/USER_GUIDE.md");
}

void MainWindow::show_getting_started() {
    if (sheet_) sheet_->close();
    sheet_ = make_getting_started(this);
    sheet_->show();
}

void MainWindow::apply_tooltips() {
    // Only where the control has none of its own.
    auto tip = [](QWidget* w, const char* text) {
        if (w && w->toolTip().isEmpty()) w->setToolTip(QString::fromUtf8(text));
    };
    tip(mode_seg_, "What the model may change: everything, only highlights, only shadows, or nothing (1 2 3 4)");
    tip(strength_, "How much of the model's residual is applied on top of the analytic inverse ([ and ])");
    tip(exposure_, "The whole master brighter or darker, in stops, before the Region EV bands; in the master, "
                   "movie exports and projects (Ctrl+Shift+Up / Down by a third of a stop, double-click for 0)");
    tip(peak_, "The view peak: how bright the display shows. It changes the view only, never the master");
    tip(preserve_, "Outside painted masks, keep the analytic inverse untouched");
    tip(regions_, "Region EV: drag a band up or down to push the brightness of that range of nits");
    tip(anchor_, "Hold the master's exposure to the SDR's below the knee, so mid-tones match the source");
    tip(anchor_knee_, "Where the anchor lets go: below it the master follows the SDR, above it the reconstruction");
    tip(carry_chroma_, "Keep the SDR's colour below the clip, where the source still had it");
    tip(btn_master_, "Render the master for the frames chosen below (M)");
    tip(btn_reprocess_, "Run the network again on the frame on screen");
    tip(findChild<QWidget*>("renderDir"), "Where masters are written");
    tip(findChild<QWidget*>("renderName"), "The file name; frames get a number after it");
    tip(findChild<QWidget*>("renderMode"), "Which frames to render: this one, all of them, or a range");
    tip(findChild<QWidget*>("renderStart"), "The first frame number written");
    tip(findChild<QWidget*>("calibrate"), "Anchor the inverse to patches of known brightness picked on the frame");
    tip(findChild<QWidget*>("reference"), "Fit the inverse to a graded HDR version of this frame");
    tip(findChild<QWidget*>("maskSize"), "Brush size in frame pixels (the mouse wheel over the picture too)");
    tip(findChild<QWidget*>("maskSoft"), "Brush softness: how much of the radius fades out");
    tip(findChild<QWidget*>("maskFlow"), "Brush flow: how much each stamp adds or takes away");
}

void MainWindow::use_project_model() {
    if (!project_model_ || loading_model_) return;   // model_settled() comes back
    if (mastering()) {
        // Not queued behind a render: it could switch models long after,
        // over a choice made meanwhile.
        log("a render is running, so the project's model (" + from_path(project_model_->package.filename()) +
            ") was not loaded; File > Model packages… switches to it");
        project_model_.reset();
        return;
    }
    const auto want = *project_model_;
    from_project_ = true;
    use_model(want.package, BackendChoice::from_key(want.backend), [this, want](bool, const QString&) {
        // In use, or refused (and logged): either way no longer pending.
        if (project_model_ && project_model_->package == want.package) project_model_.reset();
    });
    from_project_ = false;
}

void MainWindow::model_settled() {
    if (!project_model_) return;
    QTimer::singleShot(0, this, [this] {
        if (project_model_ && project_model_->package == model_package()) project_model_.reset();
        else use_project_model();
    });
}

bool MainWindow::save_interactive() {
    if (project_path_.empty()) {
        run("save-project-as");
        return !project_path_.empty() && !dirty_;
    }
    if (save_project_to(project_path_)) return true;
    QMessageBox::warning(this, "RUDRA", problem_);
    return false;
}

bool MainWindow::confirm_replace(bool closing) {
    if (!dirty_ || !isVisible()) return true;
    // Closing an untitled session loses nothing: it is autosaved, and the
    // next start offers it (File > Reopen last session).
    if (project_path_.empty() && (closing || (frames_.empty() && !pending_shot_))) return true;
    const QString name = project_path_.empty() ? QStringLiteral("this session") : from_path(project_path_.filename());
    const auto b = QMessageBox::question(this, "RUDRA", "Save the changes to " + name + "?",
                                         QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
                                         QMessageBox::Save);
    if (b == QMessageBox::Cancel) return false;
    if (b == QMessageBox::Save) return save_interactive();
    return true;
}

void MainWindow::show_getting_started_once() {
    QSettings st;
    // Once per edition of the steps: "2" is RUDRA 1.0's (Source, exposure, the
    // clip fill), so a beta user sees the new ones once too.
    if (st.value("help/gettingStarted", 0).toInt() >= 2) return;
    st.setValue("help/gettingStarted", 2);
    // A new user (29 Oct 2026): the welcome card and the four tips, in place of the five
    // pages (Help > Getting started still has them).
    show_tips_once();
}


}  // namespace rudra::app
