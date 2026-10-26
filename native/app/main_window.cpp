#include "main_window.hpp"
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

struct MainWindow::FeederSlot {
    std::mutex mu;
    std::shared_ptr<MovieFeeder> feeder;
};

namespace {

QString qs(std::string_view s) { return QString::fromUtf8(s.data(), qsizetype(s.size())); }

// Paths through UTF-16 both ways: std::string paths are the ANSI code page on Windows.
std::filesystem::path to_path(const QString& s) { return std::filesystem::path(s.toStdU16String()); }
QString from_path(const std::filesystem::path& p) { return QString::fromStdU16String(p.u16string()); }

QString describe(const ModelManifest& m) {
    return QStringLiteral("%1  ·  contract %2  ·  residual gate %3  ·  shadow gate %4  ·  curve %5  ·  corpus EV %6")
        .arg(QString::fromStdString(m.name), QString::fromStdString(m.contract), m.has_residual_gate ? "yes" : "no",
             m.has_shadow_gate ? "yes" : "no", m.has_curve ? "yes" : "no")
        .arg(m.corpus_ev, 0, 'f', 1);
}

// The page's key (KeyboardEvent.key) as a Qt shortcut. Letters without
// Shift: the page maps both cases to one action.
QKeySequence key_sequence(std::string_view k) {
    if (k == " ") return QKeySequence(Qt::Key_Space);
    if (k == ",") return QKeySequence(Qt::Key_Comma);
    if (k == ".") return QKeySequence(Qt::Key_Period);
    if (k == "?") return QKeySequence(Qt::Key_Question);
    if (k == "[") return QKeySequence(Qt::Key_BracketLeft);
    if (k == "]") return QKeySequence(Qt::Key_BracketRight);
    if (k == "Home") return QKeySequence(Qt::Key_Home);
    if (k == "End") return QKeySequence(Qt::Key_End);
    if (k.size() == 1) return QKeySequence(qs(k).toUpper());
    return QKeySequence(qs(k));
}

// Which actions are one choice among several: the part of data-check before
// the colon ("mode:all" is in the "mode" group).
QString check_group(std::string_view check) {
    const auto colon = check.find(':');
    return colon == std::string_view::npos ? QString() : qs(check.substr(0, colon));
}

}  // namespace

MainWindow::MainWindow(bool with_viewer, bool headroom_capable)
    : session_(headroom_capable ? Session::SourceSupport::HeadroomV1 : Session::SourceSupport::LegacyOnly),
      headroom_capable_(headroom_capable) {
    setWindowTitle("RUDRA");
    resize(1600, 1000);
    session_.on_change([this](std::uint32_t what) { session_changed(what); });
    build_ui(with_viewer);
    setAcceptDrops(true);   // a drop anywhere opens (the page's window drop)
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        viewer_->on_status([this](const ViewerStatus&) {
            // A wipe dragged or dropped on the plate itself: the session follows.
            const double w = viewer_->view().wipe;
            const std::optional<double> now = w >= 0.0 ? std::optional<double>(w) : std::nullopt;
            if (now != session_.wipe) {
                session_.wipe = now;
                sync_checks();
            }
            show_status();
        });
    }
#endif
    bind_handlers();
    build_menus();
    apply_tooltips();
    // Autosave (set_autosave_path): the trailing edge of the last change.
    autosave_timer_.setSingleShot(true);
    connect(&autosave_timer_, &QTimer::timeout, this, [this] { autosave_now(); });
    pending_["reopen-session"] = "No earlier session was saved.";
    use_model_ = QSettings().value("model/use", false).toBool();
    mode_label_ = new QLabel(this);
    mode_label_->setObjectName("reconstructionMode");
    statusBar()->addPermanentWidget(mode_label_);
    show_mode();
    sync_checks();
    sync_ui();
    refresh_enabled();
    update_pipe();

    // The trailing edge of the page's scheduleStats: measure 90 ms after the
    // grade last moved, on another thread; the picture never waits for it.
    stats_timer_.setSingleShot(true);
    stats_timer_.setInterval(90);
    connect(&stats_timer_, &QTimer::timeout, this, [this] { run_stats(); });
    // 24 fps, the Studio's default; a frame that is not ready is held, not skipped.
    // Connected once here: Qt::UniqueConnection does not apply to a lambda, so
    // connecting on every Play added a handler each time and playback ran fast.
    play_.setInterval(1000 / 24);
    connect(&play_, &QTimer::timeout, this, [this] { play_tick(); });
    feeder_slot_ = std::make_shared<FeederSlot>();
    {
        QSettings st;
        play_state_->side = st.value("playback/side", 960).toInt();
        realtime_ = st.value("playback/realtime", true).toBool();
        steady_ = st.value("video/steady", true).toBool();
    }
    // While playing: the rate reached, once a second, in the status bar.
    play_status_.setInterval(1000);
    connect(&play_status_, &QTimer::timeout, this, [this] {
        const int side = play_state_->side.load();
        statusBar()->showMessage(
            QStringLiteral("Playing %1 fps of %2").arg(play_shown_window_).arg(play_fps(), 0, 'f', 3) +
            (use_model_ ? (side > 0 ? QStringLiteral(" \u00b7 model at %1 px").arg(side) : QStringLiteral(" \u00b7 model at full size"))
                        : QStringLiteral(" \u00b7 analytic")) +
            (play_skipped_ ? QStringLiteral(" \u00b7 %1 skipped").arg(play_skipped_) : QString()));
        play_shown_window_ = 0;
    });

    // The floating probe box (#probeBox): it follows the cursor over the viewer,
    // a window of its own so it can sit over the viewer's swapchain.
    probe_box_ = new QFrame(this, Qt::ToolTip | Qt::FramelessWindowHint);
    probe_box_->setObjectName("probeBox");
    probe_box_->setAttribute(Qt::WA_ShowWithoutActivating);
    probe_box_->setAttribute(Qt::WA_TransparentForMouseEvents);
    auto* pbl = new QGridLayout(probe_box_);
    pbl->setContentsMargins(8, 6, 8, 6);
    pbl->setHorizontalSpacing(10);
    pbl->setVerticalSpacing(1);
    probe_box_->hide();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        viewer_->on_drop([this](const QStringList& paths) { open_paths(paths); });
        // Calibrate (3.2): the armed anchor takes the next click on the picture.
        viewer_->on_pick([this](double x, double y) {
            if (clip_pick_armed_) clip_peak_picked(x, y);
            else calibration_picked(x, y);
        });
        // Painted masks (3.3): strokes while a band is armed, the wheel sizes the brush.
        viewer_->on_paint([this](double x0, double y0, double x1, double y1, bool erase) { paint_stroke(x0, y0, x1, y1, erase); });
        viewer_->on_paint_end([this] { session_.stroke_end(); });
        viewer_->on_brush_wheel([this](double notches) {
            Brush b = session_.brush();
            b.size_px = std::clamp(b.size_px * std::pow(1.15, notches), 2.0, 600.0);
            session_.set_brush(b);
        });
        viewer_->on_hover([this](const ViewerWindow::Hover& h) {
            if (!(probe_on_ || h.alt) || !h.x) {
                probe_pixel(std::nullopt);
                return;
            }
            QWidget* host = findChild<QWidget*>("viewerHost");
            probe_pixel(std::pair{*h.x, *h.y}, host ? host->mapToGlobal(h.window.toPoint()) : QPoint());
        });
    }
#endif

    QStringList runtimes;
    for (auto r : compiled_runtimes()) runtimes << to_string(r);
    runtimes_ = runtimes.isEmpty() ? "no inference runtime" : runtimes.join(", ");
    log("runtimes: " + runtimes_);
    log(QString("container: ") + container_field_->text());
}

MainWindow::~MainWindow() {
    headroom_export_cancel_ = true;   // a headroom export publishes nothing once cancelled
    if (headroom_export_worker_.joinable()) headroom_export_worker_.join();
    headroom_.reset();   // cancels its decoder and joins its worker; queued frames become no-ops
    if (feeder_slot_) {   // its decoder thread first: the engine's loader reads from it
        std::lock_guard lock(feeder_slot_->mu);
        if (feeder_slot_->feeder) feeder_slot_->feeder->stop();   // releases a loader waiting in take()
        feeder_slot_->feeder.reset();
    }
    queues_->finished = nullptr;
    queues_->stop();      // a running export is left interrupted, as Ctrl+C leaves it
    queues_.reset();
    if (model_worker_.joinable()) model_worker_.join();   // a package still loading
    if (stats_worker_.joinable()) stats_worker_.join();   // a measurement in flight uses the frame
    master_job_.reset();   // cancels and waits: it uses the backend
    encode_cancel_ = true;
    if (encode_worker_.joinable()) encode_worker_.join();   // a movie being encoded from mastered frames
    play_.stop();
    engine_.reset();   // joins the worker before the backend goes
}

QAction* MainWindow::action(std::string_view id) const {
    const auto it = actions_.find(id);
    return it == actions_.end() ? nullptr : it->second;
}

QString MainWindow::pending_reason(std::string_view id) const {
    const auto it = pending_.find(id);
    return it == pending_.end() ? QString() : it->second;
}

void MainWindow::run(std::string_view id) {
    QAction* a = action(id);
    if (a && !a->isEnabled()) return;
    const auto it = handlers_.find(id);
    if (it != handlers_.end()) it->second();
    sync_checks();
}

void MainWindow::bind_handlers() {
    auto& h = handlers_;
    h["open"] = [this] {
        // The page's file input: several stills at once, added to the frames.
        const QStringList files = QFileDialog::getOpenFileNames(
            this, "Open SDR frames", {}, "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)");
        std::vector<std::filesystem::path> paths;
        for (const auto& f : files) paths.emplace_back(f.toStdString());
        if (!paths.empty()) add_files(paths);
    };
    h["open-folder"] = [this] { open_source({}, true); };
    h["open-package"] = [this] { open_package(); };
    h["models"] = [this] { open_model_manager(); };
    h["aces-master"] = [this] { open_aces_master(); };
    h["first-run"] = [this] { open_first_run(); };
    // Through close(), so unsaved work is asked about and the settings and the autosave are written.
    h["quit"] = [this] {
        if (close()) QApplication::quit();
    };
    h["open-project"] = [this] {
        if (!confirm_replace(false)) return;
        const QString f = QFileDialog::getOpenFileName(this, "Open project", {}, "RUDRA projects (*.rudra)");
        if (!f.isEmpty() && !open_project(to_path(f)))
            QMessageBox::warning(this, "RUDRA", problem_);
    };
    auto save_as = [this] {
        QString suggest = from_path(project_path_);
        if (suggest.isEmpty()) {
            const QString title = frames_.empty() ? QStringLiteral("untitled") : QFileInfo(shot_title()).completeBaseName();
            const QString dir = shot_sources_.empty()
                                    ? QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
                                    : QFileInfo(from_path(shot_sources_.front())).absolutePath();
            suggest = QDir(dir).filePath((title.isEmpty() ? QStringLiteral("untitled") : title) + ".rudra");
        }
        const QString f = QFileDialog::getSaveFileName(this, "Save project", suggest, "RUDRA projects (*.rudra)");
        if (!f.isEmpty() && !save_project_to(to_path(f)))
            QMessageBox::warning(this, "RUDRA", problem_);
    };
    h["save-project-as"] = [save_as] { save_as(); };
    h["save-project"] = [this] { save_interactive(); };
    h["reopen-session"] = [this] {
        if (reopen_path_.empty() || !confirm_replace(false)) return;
        if (!open_project(reopen_path_, false) && isVisible()) QMessageBox::warning(this, "RUDRA", problem_);
    };
    h["getting-started"] = [this] { show_getting_started(); };
    h["use-model"] = [this] { set_use_model(!use_model_); };
    h["clip-fill"] = [this] { session_.toggle_clip_fill(); };
    h["clip-peak"] = [this] { arm_clip_peak(!clip_pick_armed_); };
    h["clip-peaks-clear"] = [this] { session_.clear_clip_peaks(); };
    h["playback-side-0"] = [this] { set_playback_side(0); };
    h["playback-side-960"] = [this] { set_playback_side(960); };
    h["playback-side-640"] = [this] { set_playback_side(640); };
    h["playback-realtime"] = [this] { set_playback_realtime(!realtime_); };
    h["steady-shots"] = [this] { set_steady_shots(!steady_); };
    h["keep-super-whites"] = [this] { toggle_super_whites(); };
    h["user-guide"] = [this] {
        log("user guide: " + user_guide_url());
        QDesktopServices::openUrl(QUrl(user_guide_url()));
    };
    h["check-updates"] = [this] { check_for_updates(true); };
    h["open-log-folder"] = [this] {
        const QString f = app_log_path();
        if (f.isEmpty()) {
            log("no log file: this run keeps its log in the window only");
            return;
        }
        QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(f).absolutePath()));
    };
    h["close"] = [this] { close_frames(); };
    h["first"] = [this] { step_to(0); };
    h["prev"] = [this] { step_to(current_ - 1); };
    h["next"] = [this] { step_to(current_ + 1); };
    h["last"] = [this] { step_to(int(frames_.size()) - 1); };
    h["play"] = [this] { toggle_play(); };
    // The page's grading state and its undo: engine/session.
    for (const auto& spec : action_specs())
        if (Session::owns(spec.id)) h[std::string(spec.id)] = [this, id = std::string(spec.id)] { session_.run(id); };
    h["shortcuts"] = [this] {
        std::vector<std::pair<QString, QString>> rows;
        for (const auto& r : shortcut_sheet()) rows.emplace_back(qs(r.keys), qs(r.what));
        rows.emplace_back("F", "Fit image to window");
        show_sheet("Keyboard", rows);
    };
    h["about"] = [this] {
        const QString device = device_->text();   // the page's $("device").textContent
        QString composite = "CPU";
#ifdef RUDRA_APP_VIEWER
        if (viewer_) composite = QStringLiteral("GPU, QRhi %1 — render/shaders/composite.frag")
                                     .arg(QString::fromStdString(viewer_->status().backend));
#endif
        // head.checkpoint: the model that made the frame on screen.
        show_sheet("RUDRA Studio", {{"Checkpoint", manifest_ && current_frame_ ? QString::fromStdString(manifest_->name) : "—"},
                                    {"Step", "—"},
                                    {"Device", device},
                                    {"Composite", composite},
                                    {"Storage", "log2_extended, 0.005–1 000 000 nits"},
                                    {"Master", "ACES 2065-1 (AP0), ST 2065-4 chromaticities"},
                                    {"Built by", "FXTD Studios / Radiance Research"}});
    };

    h["view-image"] = [this] { session_.set_view_layer(0); };
    h["view-false-colour"] = [this] { session_.set_view_layer(1); };
    h["view-difference"] = [this] { session_.set_view_layer(2); };

    // Waiting on later steps of Phase 3: off, and they say why.
    // copy(): JSON.stringify(value, null, 2) to the clipboard (core/copy_texts).
    h["copy-metrics"] = [this] {
        if (measure_) copy_text("measurements", metrics_value(measure_->measured.metrics, measure_->compose_ms).stringify());
    };
    h["copy-scopes"] = [this] {
        if (measure_) copy_text("scope data", scopes_value(measure_->measured.scopes).stringify());
    };
    h["copy-delivery"] = [this] { copy_text("delivery metadata", delivery_text()); };
    h["recent-clear"] = [this] {
        QSettings().remove("recent/sources");
        fill_recent();
    };
    h["remeasure"] = [this] {
        measure_now();
        log("re-measured");
    };
    h["master"] = [this] {
        if (mastering()) {   // while it renders, the button stops it after this frame
            master_job_->cancel();
            findChild<QLabel*>("renderStatus")->setText("Stopping after this frame\u2026");
            return;
        }
        if (encoding_movie_) {
            encode_cancel_ = true;
            findChild<QLabel*>("renderStatus")->setText("Stopping the movie\u2026");
            return;
        }
        master();
    };

#ifdef RUDRA_APP_VIEWER
    {
        auto guides = [this](auto&& edit) {
            return [this, edit] {
                if (!viewer_) return;
                auto g = viewer_->guides();
                edit(g);
                viewer_->set_guides(g);
            };
        };
        h["zoom-fit"] = [this] { if (viewer_) viewer_->zoom_fit(); };
        h["zoom-actual"] = [this] { if (viewer_) viewer_->zoom_actual(); };
        h["guides-action"] = guides([](GuideOptions& g) { g.action_safe = !g.action_safe; });
        h["guides-title"] = guides([](GuideOptions& g) { g.title_safe = !g.title_safe; });
        h["guides-centre"] = guides([](GuideOptions& g) { g.centre = !g.centre; });
        const std::pair<const char*, double> aspects[] = {{"aspect-none", 0.0}, {"aspect-2.39", 2.39},
                                                          {"aspect-1.85", 1.85}, {"aspect-16:9", 16.0 / 9.0},
                                                          {"aspect-4:3", 4.0 / 3.0}, {"aspect-1:1", 1.0}};
        for (const auto& [id, r] : aspects) h[id] = guides([aspect = r](GuideOptions& g) { g.aspect = aspect; });
    }
#endif
    for (const auto& a : action_specs()) {
        const std::string id(a.id);
        if (!handlers_.count(id) && !pending_.count(id)) pending_[id] = "This build has no viewer.";
    }
}

void MainWindow::build_menus() {
    std::map<QString, QActionGroup*> groups;
    for (const auto& spec : action_specs()) {
        auto* a = new QAction(qs(spec.label), this);
        a->setObjectName(QStringLiteral("act:") + qs(spec.id));
        QList<QKeySequence> keys;
        for (auto k : mapped_keys()) {
            if (action_for_key(k) != spec.id) continue;
            const QKeySequence seq = key_sequence(k);
            if (!keys.contains(seq)) keys << seq;
            // The page maps "z" and "Z" alike: a letter works with Shift (or
            // Caps Lock) too.
            if (k.size() == 1 && std::isalpha(static_cast<unsigned char>(k[0]))) {
                const QKeySequence shifted(QStringLiteral("Shift+") + seq.toString(QKeySequence::PortableText));
                if (!keys.contains(shifted)) keys << shifted;
            }
        }
        if (!spec.native_key.empty()) keys << QKeySequence(qs(spec.native_key));
        // The hint the page prints comes first, so the menu shows it.
        if (!spec.hint.empty()) {
            const QKeySequence hint = key_sequence(spec.hint == "Space" ? " " : spec.hint);
            keys.removeAll(hint);
            keys.prepend(hint);
        }
        if (spec.id == "zoom-fit") keys.prepend(QKeySequence(Qt::Key_F));
        a->setShortcuts(keys);
        a->setShortcutContext(Qt::WindowShortcut);
        if (!spec.check.empty()) {
            a->setCheckable(true);
            const QString g = check_group(spec.check);
            if (!g.isEmpty()) {
                auto*& group = groups[g];
                if (!group) group = new QActionGroup(this);
                group->addAction(a);
            }
        }
        const std::string id(spec.id);
        connect(a, &QAction::triggered, this, [this, id] { run(id); });
        actions_[id] = a;
        addAction(a);   // shortcuts work with the menubar hidden too
    }
    for (const auto& m : menu_specs()) {
        QMenu* menu = menuBar()->addMenu(qs(m.title));
        std::vector<QMenu*> stack{menu};
        for (auto e : m.entries) {
            if (e == "-") stack.back()->addSeparator();
            else if (e == "<") stack.pop_back();
            else if (e.substr(0, 1) == ">") {
                QMenu* sub = stack.back()->addMenu(qs(e.substr(1)));
                if (e == ">Open recent") {
                    recent_menu_ = sub;
                    connect(sub, &QMenu::aboutToShow, this, [this] { fill_recent(); });
                }
                stack.push_back(sub);
            }
            else stack.back()->addAction(actions_.at(std::string(e)));
        }
        connect(menu, &QMenu::aboutToShow, this, [this] {
            sync_checks();
            refresh_enabled();
        });
        // shell.js TAB_FOR_ACT: a menu item that changes something on a tab
        // that is not showing brings that tab forward (menus only, not keys).
        connect(menu, &QMenu::triggered, this, [this](QAction* a) {
            static const std::map<QString, QString> tab_for = {
                {"act:reset-regions", "grade"},  {"act:container-aces", "deliver"}, {"act:container-linear", "deliver"},
                {"act:copy-delivery", "deliver"}, {"act:mode-all", "rec"},         {"act:mode-highlights", "rec"},
                {"act:mode-shadows", "rec"},      {"act:mode-off", "rec"},         {"act:preserve", "rec"},
                {"act:source-unknown", "rec"},    {"act:source-bt1886", "rec"},   {"act:source-aces", "rec"},
                {"act:source-bt1886_shadows", "rec"},
                {"act:source-hable", "rec"},
                {"act:source-agx", "rec"},        {"act:source-camera_log", "rec"}, {"act:source-clip", "rec"},
                {"act:exposure-up", "rec"},      {"act:exposure-down", "rec"},  {"act:exposure-reset", "rec"},
                {"act:strength-up", "rec"},       {"act:strength-down", "rec"},    {"act:reset-recon", "rec"}};
            const auto it = tab_for.find(a->objectName());
            if (it != tab_for.end()) show_tab(it->second);
        });
    }
}

void MainWindow::refresh_enabled() {
    const bool any = !frames_.empty(), many = frames_.size() > 1;
    for (const auto& spec : action_specs()) {
        QAction* a = action(spec.id);
        const QString why = pending_reason(spec.id);
        bool on = true;
        switch (spec.enable) {
            case EnableRule::Always: break;
            case EnableRule::AnyFrames: on = any; break;
            case EnableRule::ManyFrames: on = many; break;
            case EnableRule::CanMaster: on = (any && !needs_model()) || mastering(); break;
            case EnableRule::CanUndo: on = session_.undo_depth() > 0; break;
            case EnableRule::CanRedo: on = session_.redo_depth() > 0; break;
            case EnableRule::HasMetrics:
            case EnableRule::HasScopes: on = measure_ != nullptr; break;
        }
        a->setEnabled(on && why.isEmpty());
        a->setToolTip(why);
        a->setStatusTip(why);
    }
}

void MainWindow::sync_checks() {
    auto set = [this](std::string_view id, bool on) {
        if (QAction* a = action(id)) a->setChecked(on);
    };
    const auto& grade = session_.grade;
    set("mode-all", grade.mode == "all");
    set("mode-highlights", grade.mode == "highlights");
    set("mode-shadows", grade.mode == "shadows");
    set("mode-off", grade.mode == "off");
    set("preserve", grade.preserve);
    set("use-model", use_model_);
    set("clip-fill", session_.grade.clip_fill);
    set("playback-side-0", play_state_->side.load() == 0);
    set("playback-side-960", play_state_->side.load() == 960);
    set("playback-side-640", play_state_->side.load() == 640);
    set("playback-realtime", realtime_);
    set("steady-shots", steady_);
    set("keep-super-whites", session_.grade.headroom_policy != HeadroomPolicy::Disabled);
    for (SourceCurve c : all_source_curves())
        set("source-" + std::string(source_curve_id(c)), grade.source == source_curve_id(c));
    set("container-aces", session_.container == "aces");
    set("container-linear", session_.container == "linear");
    set("wipe", session_.wipe.has_value());
    set("view-image", session_.view_layer == 0);
    set("view-false-colour", session_.view_layer == 1);
    set("view-difference", session_.view_layer == 2);
    set("rail-left", session_.rail_left);
    set("rail-right", session_.rail_right);
    set("scopes", session_.scopes_open);
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        const auto g = viewer_->guides();
        const bool fit = viewer_->viewport().scale <= 0.0;
        set("zoom-fit", fit);
        set("zoom-actual", !fit);
        set("guides-action", g.action_safe);
        set("guides-title", g.title_safe);
        set("guides-centre", g.centre);
        set("aspect-none", g.aspect <= 0.0);
        set("aspect-2.39", std::abs(g.aspect - 2.39) < 1e-9);
        set("aspect-1.85", std::abs(g.aspect - 1.85) < 1e-9);
        set("aspect-16:9", std::abs(g.aspect - 16.0 / 9.0) < 1e-9);
        set("aspect-4:3", std::abs(g.aspect - 4.0 / 3.0) < 1e-9);
        set("aspect-1:1", std::abs(g.aspect - 1.0) < 1e-9);
        return;
    }
#endif
    set("zoom-fit", true);
}

namespace {

// The page's overlay sheet: Esc, or a click anywhere, closes it.
class Sheet : public QDialog {
public:
    using QDialog::QDialog;

protected:
    void mousePressEvent(QMouseEvent*) override { accept(); }
};

}  // namespace

void MainWindow::show_sheet(const QString& title, const std::vector<std::pair<QString, QString>>& rows) {
    if (sheet_) sheet_->close();
    auto* d = new Sheet(this);
    d->setAttribute(Qt::WA_DeleteOnClose);
    d->setObjectName("sheet");
    d->setWindowTitle(title);
    auto* grid = new QGridLayout(d);
    auto* head = new QLabel(title, d);
    head->setObjectName("sheetTitle");
    head->setProperty("role", "plabel");
    grid->addWidget(head, 0, 0, 1, 2);
    int r = 1;
    for (const auto& [k, v] : rows) {
        auto* kl = new QLabel(k, d);
        kl->setProperty("role", "value");
        auto* vl = new QLabel(v, d);
        vl->setTextInteractionFlags(Qt::TextSelectableByMouse);
        grid->addWidget(kl, r, 0);
        grid->addWidget(vl, r, 1);
        ++r;
    }
    auto* close = new QLabel("Esc, or click anywhere, to close", d);
    close->setObjectName("sheetClose");
    close->setProperty("role", "note");
    grid->addWidget(close, r, 0, 1, 2);
    sheet_ = d;
    d->show();
}

void MainWindow::session_changed(std::uint32_t what) {
    if (what & (Session::Grade | Session::Peak | Session::Delivery)) mark_dirty();   // what a project keeps
    // The clip fill is in the fields, not the composite: a change (or its
    // undo) composes the frames again. With the model it does nothing.
    if ((what & Session::Grade) && !use_model_ && !frames_.empty() && !headroom_routed()) {
        const ClipFillOptions now = session_clip_options();
        const bool same = now.enabled == engine_clip_.enabled && now.overrides.size() == engine_clip_.overrides.size() &&
                          std::equal(now.overrides.begin(), now.overrides.end(), engine_clip_.overrides.begin(),
                                     [](const ClipOverride& a, const ClipOverride& b) {
                                         return a.x == b.x && a.y == b.y && a.nits == b.nits;
                                     });
        if (!same) start_engine(frames_, current_);
    }
    if (what & (Session::Grade | Session::Delivery)) {
        // The headroom route snapshots the grade after every change, undo and
        // restore included, and takes the frame on screen again (or switches
        // back to the legacy engine when the policy was turned off).
        if (headroom_capable_ && (headroom_ || headroom_routed())) route_headroom();
        schedule_stats();   // Delivery: the anchor knee moves the hold (3.3)
    }
    if (what & (Session::Peak | Session::Delivery)) update_pipe();
    if (what & (Session::Peak | Session::View)) update_compare();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        if (what & Session::Grade) viewer_->set_composite(session_.composite_params());
        if (what & (Session::Peak | Session::Wipe | Session::View | Session::Delivery)) {
            auto v = viewer_->view();
            v.display_nits = session_.display_nits();
            // The anchor (3.3): the Deliver setting, live, with the hold the
            // last measure computed for this knee (a stale hold until the
            // measure after a knee move lands; the per-pixel part is current).
            v.anchor = session_.anchor;
            v.anchor_knee = session_.anchor_knee;
            v.anchor_hold = measure_ ? shown_hold_ : 1.0;
            v.wipe = session_.wipe ? *session_.wipe : -1.0;
            v.mode = session_.view_layer == 1   ? ViewMode::FalseColour
                     : session_.view_layer == 2 ? ViewMode::Difference
                     : session_.view_layer == 3 ? ViewMode::Invented
                                                : ViewMode::Image;
            v.show = session_.show == "baseline" ? ViewSource::Baseline : ViewSource::Model;
            v.show_changes = session_.show_changes;
            // The mask being painted (3.3), tinted in its band's colour.
            static const std::array<float, 3> kBandTints[kMaxMaskBands] = {
                {1.0f, 0.70f, 0.25f}, {0.75f, 0.35f, 0.95f}, {0.04f, 0.52f, 1.0f}, {0.19f, 0.82f, 0.35f}};
            const int band = session_.show_masks ? session_.painted_band() : -1;
            v.paint_band = band;
            if (band >= 0) v.paint_tint = kBandTints[band];
            viewer_->set_view(v);
        }
    }
#endif
    sync_checks();
    sync_ui();
    refresh_enabled();
}

void MainWindow::open_package(const QString& preset) {
    const QString dir = preset.isEmpty() ? QFileDialog::getExistingDirectory(this, "Open model package") : preset;
    if (dir.isEmpty()) return;
    use_model(dir.toStdString(), backend_choice_);
}

// ---------------------------------------------------------------------------
// The checkpoint manager (step 10)
// ---------------------------------------------------------------------------

struct MainWindow::LoadedModel {
    ModelManifest manifest;
    OpenedBackend opened;
    std::optional<SelfTestReport> report;   // nullopt: this package passed on this backend before
    std::optional<Error> error;
};

namespace {

// The package's bytes (the checkpoint's and every file's hash) and the backend.
QString verified_key(const ModelManifest& m, const BackendChoice& c) {
    std::string k = m.source_sha256;
    for (const auto& [file, sha] : m.file_sha256) k += "|" + sha;
    return QString::fromStdString(k + "|" + c.key());
}

}  // namespace

void MainWindow::set_model_roots(std::vector<std::filesystem::path> roots) {
    roots_override_ = std::move(roots);
    rescan_models();
}

std::vector<std::filesystem::path> MainWindow::model_roots() const {
    if (roots_override_) return *roots_override_;
    QSettings st;
    std::vector<std::filesystem::path> extra;
    for (const auto& r : st.value("model/roots").toStringList()) extra.emplace_back(r.toStdString());
    // The packages shipped with the app: beside the executable, or in a macOS
    // bundle's Resources (RUDRA.app/Contents/MacOS/../Resources/models).
    auto app_models = std::filesystem::path(QCoreApplication::applicationDirPath().toStdString()) / "models";
#ifdef __APPLE__
    if (const auto res = app_models.parent_path().parent_path() / "Resources" / "models"; std::filesystem::is_directory(res))
        app_models = res;
#endif
    const QString user = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    return package_roots(extra, app_models,
                         user.isEmpty() ? std::filesystem::path() : std::filesystem::path(user.toStdString()) / "models");
}

void MainWindow::add_model_root(const std::filesystem::path& root) {
    if (roots_override_) {
        if (std::find(roots_override_->begin(), roots_override_->end(), root) == roots_override_->end())
            roots_override_->push_back(root);
    } else {
        QSettings st;
        QStringList r = st.value("model/roots").toStringList();
        const QString q = QString::fromStdString(root.string());
        if (!r.contains(q)) r << q;
        st.setValue("model/roots", r);
    }
    rescan_models();
}

void MainWindow::rescan_models() { catalog_ = scan_packages(model_roots()); }

void MainWindow::use_model(const std::filesystem::path& package, std::optional<BackendChoice> choice,
                           std::function<void(bool, QString)> done) {
    auto refuse = [&](const QString& why) {
        log(why);
        if (done) done(false, why);
    };
    // A model the user asks for replaces a project's model still waiting its turn.
    if (!from_project_) project_model_.reset();
    if (mastering()) return refuse("A render is running: stop it before switching models.");
    if (loading_model_) return refuse("A model is still loading.");
    auto m = read_manifest(package);
    if (!m) return refuse(QString::fromStdString(m.error().message + " " + m.error().detail));
    // Which pairs of this package and a backend have passed before.
    QSettings st;
    const QStringList verified = st.value("model/verified").toStringList();
    loading_model_ = true;
    ckpt_->setText(QStringLiteral("loading %1…").arg(QString::fromStdString(m->name)));
    log("loading " + describe(*m));
    refresh_enabled();
    if (model_worker_.joinable()) model_worker_.join();
    QPointer<MainWindow> self(this);
    auto hooks = hooks_;
    model_worker_ = std::thread([self, man = *m, choice, verified, hooks, done]() mutable {
        auto loaded = std::make_shared<LoadedModel>();
        loaded->manifest = man;
        auto run = [&]() -> std::optional<Error> {
            if (hooks.verify_files)
                if (auto v = verify_package_files(man); !v) return v.error();
            auto o = hooks.open ? hooks.open(man, choice) : open_backend(man, choice);
            if (!o) return o.error();
            loaded->opened = std::move(*o);
            if (!verified.contains(verified_key(man, loaded->opened.choice))) {
                auto r = hooks.test ? hooks.test(man, *loaded->opened.backend) : self_test(man, *loaded->opened.backend);
                if (!r) return r.error();
                loaded->report = std::move(*r);
            }
            return std::nullopt;
        };
        loaded->error = run();
        QMetaObject::invokeMethod(qApp, [self, loaded, done] {
            if (self) self->adopt_model(loaded, done);
        });
    });
}

void MainWindow::adopt_model(std::shared_ptr<LoadedModel> loaded, const std::function<void(bool, QString)>& done) {
    if (model_worker_.joinable()) model_worker_.join();
    loading_model_ = false;
    const ModelManifest& m = loaded->manifest;
    QString why;
    if (loaded->error) {
        why = describe(m) + "  ·  " + QString::fromStdString(loaded->error->message) +
              (loaded->error->detail.empty() ? "" : " " + QString::fromStdString(loaded->error->detail));
    } else if (loaded->report && !loaded->report->pass()) {
        why = QStringLiteral("model %1 refused on %2: self-test %3")
                  .arg(QString::fromStdString(m.name), QString::fromStdString(loaded->opened.choice.label()),
                       QString::fromStdString(loaded->report->summary()));
    }
    if (!why.isEmpty()) {
        log(why);
        set_model_pills();   // the model in use, if any, is still in use
        refresh_enabled();
        if (done) done(false, why);
        model_settled();
        return;
    }
    if (loaded->report) {
        log("self-test " + QString::fromStdString(loaded->report->summary()));
        QSettings st;
        QStringList v = st.value("model/verified").toStringList();
        v << verified_key(m, loaded->opened.choice);
        st.setValue("model/verified", v);
    }
    // The engine's worker uses the backend: it goes first. The session and
    // the frames stay; the frame on screen is asked for again.
    const bool had_frames = !frames_.empty();
    halt_playback();
    btn_play_->set_glyph(IconButton::Glyph::Play);
    {
        std::lock_guard lock(*backend_mutex_);
        engine_.reset();
        manifest_ = std::make_unique<ModelManifest>(m);
        backend_ = std::move(loaded->opened.backend);
        backend_choice_ = loaded->opened.choice;
    }
    QSettings st;
    st.setValue("model/package", QString::fromStdString(m.root.string()));
    st.setValue("model/backend", QString::fromStdString(backend_choice_->key()));
    // Whether the package in use is the catalog's default. A bare start
    // follows the default when it moves (shadow_v1 -> v8 on 9 Oct 2026);
    // a package picked over the default stays picked.
    {
        const auto in_use = catalog_.find(m.root);
        st.setValue("model/followsDefault", bool(in_use && catalog_.chosen && *in_use == *catalog_.chosen));
    }
    set_model_pills();
    log("model " + describe(m));
    log("device " + device_->text());
    if (had_frames) start_engine(frames_, current_);   // the engine was stopped above, whichever reconstruction
    else if (!pending_source_) log("drop frames — the network runs once each, then the grade is local");
    if (pending_source_) {
        // Copies, not a structured binding: the lambda captures them.
        const QString path = pending_source_->first;
        const bool folder = pending_source_->second;
        pending_source_.reset();
        QTimer::singleShot(0, this, [this, path, folder] { open_source(path, folder); });
    }
    if (pending_shot_) {
        auto shot = std::move(*pending_shot_);
        pending_shot_.reset();
        QTimer::singleShot(0, this, [this, shot] {
            restoring_ = true;
            open_shot(shot.kind, shot.sources, shot.frame);
            restoring_ = false;
        });
    }
    sync_ui();
    refresh_enabled();
    if (done) done(true, {});
    model_settled();
}

const Fields* MainWindow::frame_fields() const { return current_frame_ ? &current_frame_->fields : nullptr; }
int MainWindow::presented_index() const {
    if (headroom_routed()) return headroom_shown_ ? headroom_shown_->index : -1;
    return current_frame_ && frame_ready_ok_ ? current_frame_->index : -1;
}
const FrameScalars* MainWindow::frame_scalars() const { return current_frame_ ? &current_frame_->scalars : nullptr; }

void MainWindow::set_model_pills() {
    show_mode();
    const bool on = backend_ != nullptr && manifest_ != nullptr;
    ckpt_->setText(on ? QString::fromStdString(manifest_->name) : QStringLiteral("no model"));
    if (on) {
        const auto info = backend_->info();
        device_->setText(QString::fromStdString(device_pill(info)));
        device_->setToolTip(QString::fromStdString(backend_choice_ ? backend_choice_->label() : std::string()) + " " +
                            QString::fromStdString(info.version) + (info.detail.empty() ? "" : " (" + QString::fromStdString(info.detail) + ")"));
    } else {
        device_->setText("—");
    }
    if (auto* st = findChild<QLabel*>("modelCardState")) {
        st->setText(on ? QStringLiteral("Verified \u00b7 ") + QString::fromStdString(backend_choice_ ? backend_choice_->label() : "")
                       : QStringLiteral("File > Model packages\u2026"));
        st->setProperty("state", on ? "ok" : "");
        st->style()->unpolish(st);
        st->style()->polish(st);
    }
    lamp_->setProperty("state", on ? "on" : "off");
    lamp_->style()->unpolish(lamp_);
    lamp_->style()->polish(lamp_);
}

void MainWindow::boot() {
    rescan_models();
    QSettings st;
    if (!st.value("firstRun/done", false).toBool()) {
        open_first_run();
        return;
    }
    const std::filesystem::path last = st.value("model/package").toString().toStdString();
    const auto choice = BackendChoice::from_key(st.value("model/backend").toString().toStdString());
    std::error_code ec;
    if (!last.empty() && std::filesystem::is_regular_file(last / "manifest.json", ec)) {
        // The last package wins, unless it was only ever the default and the
        // default has moved on. Settings from before this flag existed say
        // nothing either way: there a package outside every folder this
        // install searches (an older beta's own models) gives way to the
        // default, and one inside them is taken as picked. Before this, a
        // beta 1 user upgraded to beta 3 kept opening beta 1's retired
        // shadow_v1 with v8 sitting unused beside the app.
        const QVariant follows = st.value("model/followsDefault");
        const auto known = catalog_.find(last);
        const bool moved = catalog_.chosen && known != catalog_.chosen &&
                           (follows.isValid() ? follows.toBool() : !known.has_value());
        if (moved) {
            const CatalogEntry& def = catalog_.entries[*catalog_.chosen];
            log("the default model is now " + QString::fromStdString(def.label()) + "; opening it instead of " +
                QString::fromStdString(last.filename().string()) +
                " (File > Model packages… switches back)");
            use_model(def.package, choice);
            return;
        }
        use_model(last, choice);
        return;
    }
    if (const auto i = catalog_.pick()) {
        use_model(catalog_.entries[*i].package, choice);
        return;
    }
    ckpt_->setText("no model package found");
    log("no model loaded: no model package found in " + QString::number(catalog_.roots.size()) +
        " folder(s); File > Model packages… adds one");
}

void MainWindow::open_source(const QString& preset, bool folder) {
// The still decode is all a shot needs (the frames go to the engine); a
// window without the viewer opens them as well, for the tests.
#ifdef RUDRA_HAVE_STILL_DECODE
    if (needs_model() && loading_model_ && !preset.isEmpty()) {
        pending_source_ = {preset, folder};   // opened when the model is in
        return;
    }
    if (needs_model()) {
        log("Open a model package first, or turn off Reconstruct > Use the model.");
        return;
    }
    QString path = preset;
    if (path.isEmpty())
        path = folder ? QFileDialog::getExistingDirectory(this, "Open a folder of SDR frames")
                      : QFileDialog::getOpenFileName(this, "Open an SDR still or movie", {},
                                                     "Stills and movies (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp "
                                                     "*.mov *.mp4 *.mxf *.mkv *.avi *.m2ts *.ts *.webm *.m4v)");
    if (path.isEmpty()) return;
    std::vector<std::filesystem::path> frames;
    std::optional<FrameSequence> video;
    const std::string suffix = QFileInfo(path).suffix().toLower().prepend('.').toStdString();
    const bool movie = QFileInfo(path).isFile() &&
                       std::find(video_suffixes().begin(), video_suffixes().end(), suffix) != video_suffixes().end();
    if (QFileInfo(path).isDir() || movie) {
        // A folder of frames, or a movie read a frame at a time as the Studio reads it.
        auto seq = open_sequence(path.toStdString());
        if (!seq) {
            log(QString::fromStdString(seq.error().message));
            return;
        }
        frames = seq->frames;
        if (seq->kind == "video") video = std::move(*seq);
    } else {
        frames = {std::filesystem::path(path.toStdString())};
    }
    remember_source(path);
    shot_kind_ = video ? "movie" : QFileInfo(path).isDir() ? "folder" : "files";
    shot_sources_ = {to_path(QFileInfo(path).absoluteFilePath())};
    shot_video_ = std::move(video);
    start_engine(std::move(frames));
    mark_dirty();
    if (shot_video_)
        log(QStringLiteral("opened %1: %2 frames at %3 fps").arg(shot_title()).arg(frame_count()).arg(*shot_video_->fps, 0, 'g', 6));
#else
    (void)preset;
    (void)folder;
    QMessageBox::information(this, "RUDRA", "This build has no still decoder (RUDRA_WITH_OPENCV).");
#endif
}

void MainWindow::close_frames() {
    halt_playback();
    stop_headroom();
    engine_.reset();
    ++engine_gen_;
    frames_.clear();
    shot_video_.reset();
    if (!shot_kind_.empty()) {
        shot_kind_.clear();
        shot_sources_.clear();
        mark_dirty();
    }
    current_ = 0;
    frame_info_.clear();
    current_frame_.reset();
    measure_.reset();
    ++stats_gen_;
    clear_scopes();
    clip_bar_->hide();
    scrub_->set_count(0);
    refresh_library();
    probe_pixel(std::nullopt);
    update_pipe();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->clear_frame();
#endif
    refresh_enabled();
    sync_ui();
    show_status();
}

void MainWindow::start_engine(std::vector<std::filesystem::path> frames, int at) {
    play_hold_.reset();   // another model or shot: nothing of the last one is held
    halt_playback();
    engine_.reset();   // joins the old worker before the new one starts
    frames_ = std::move(frames);
    auto* backend = use_model_ ? backend_.get() : nullptr;
    const ModelConstants model = frame_constants();
    engine_clip_ = session_clip_options();
    auto files = frames_;
    source_sizes_ = std::make_shared<SourceSizes>();
    source_sizes_->size.assign(files.size(), {0, 0});
    source_sizes_->colour.resize(files.size());
    engine_ = std::make_unique<FrameEngine>(
        [files, sizes = source_sizes_, side = preview_max_side_, slot = feeder_slot_](int i) -> Result<SdrImage> {
#ifdef RUDRA_HAVE_STILL_DECODE
            // Playing a movie: the decoder that runs in order (MovieFeeder).
            std::shared_ptr<MovieFeeder> feeder;
            {
                std::lock_guard lock(slot->mu);
                feeder = slot->feeder;
            }
            if (feeder && feeder->covers(i))
                if (auto img = feeder->take(i)) return std::move(*img);
            auto file = ensure_frame_file(files[std::size_t(i)]);   // a movie's frame is extracted on first use
            if (!file) return file.error();
            auto d = decode_sdr_file(*file);
            if (!d) return d.error();
            {
                std::lock_guard lock(sizes->mu);
                sizes->size[std::size_t(i)] = {d->rgb.width(), d->rgb.height()};
                sizes->colour[std::size_t(i)] = d->colour;
            }
            // The Studio's preview: 1600 on the long side (ui/server.py _fit);
            // masters decode again at full size.
            return fit_max_side(d->rgb, side);
#else
            // A build without media's still decode (no OpenCV) opens no
            // frames; the viewer and the rest of the shell still work.
            (void)files;
            (void)i;
            return make_error(ErrorCode::Unsupported, "This build has no still decode (RUDRA_WITH_OPENCV=OFF).");
#endif
        },
        [backend, mu = backend_mutex_, play = play_state_, clip = engine_clip_](const SdrImage& sdr) -> Result<FrameResult> {
            // The analytic reconstruction runs no network: its fields are the
            // clip fill's (core/clip_fill.hpp), or none when that is off.
            if (!backend)
                return FrameResult{FrameScalars{}, clip.enabled ? clip_fill_fields(sdr, kAnalyticConstants, clip)
                                                                : analytic_fields(sdr.height(), sdr.width()),
                                   false, false};
            // Playing: the model on a smaller copy, its fields brought back to
            // the frame's size (a proxy, inferred again where the playhead stops).
            const int side = play->side.load();
            const int h = sdr.height(), w = sdr.width();
            if (play->playing.load() && side > 0 && std::max(h, w) > side) {
                const double k = double(side) / double(std::max(h, w));
                const int sh = std::max(1, int(std::lround(h * k))), sw = std::max(1, int(std::lround(w * k)));
                const SdrImage small(area_resize(sdr.buffer(), sh, sw));
                std::lock_guard lock(*mu);
                auto fr = infer_frame(*backend, small, TileConfig{0, 0});
                if (!fr) return fr.error();
                fr->fields = resize_fields(fr->fields, h, w);
                fr->proxy = true;
                return fr;
            }
            // One untiled pass, as the Studio's preview; the master job shares the backend.
            std::lock_guard lock(*mu);
            return infer_frame(*backend, sdr, TileConfig{0, 0});
        });
    QPointer<MainWindow> self(this);
    // A frame the engine before this one finished (another model's, or
    // frames since closed) is not shown.
    const int gen = ++engine_gen_;
    engine_->on_ready([self, model, gen](const ReadyFrame& f) {
        QMetaObject::invokeMethod(qApp, [self, f, model, gen] {
            if (self && self->engine_gen_ == gen) self->frame_ready(f, model);
        });
    });
    engine_->set_sequence(int(frames_.size()));
    scrub_->set_count(int(frames_.size()));
    first_thumb_ = QImage();
    refresh_library();
    refresh_enabled();
    sync_ui();
    step_to(at);
}

void MainWindow::step_to(int i) {
    if (headroom_routed()) {
        headroom_request(i, headroom_playing_);
        return;
    }
    if (headroom_) stop_headroom();   // another shot or policy: the legacy engine again
    if (!engine_ || frames_.empty()) return;
    const int n = int(frames_.size());
    current_ = ((i % n) + n) % n;
    waiting_ = true;
    frame_ready_ok_ = false;
    engine_->show(current_);
    sync_ui();
}

void MainWindow::toggle_play() {
    if (headroom_routed()) {
        // The headroom preview plays as fast as its frames render (no clock,
        // no skipping, no temporal hold): an inspection mode, not real time.
        if (headroom_playing_) {
            stop_playback();
            return;
        }
        if (frames_.size() < 2) return;
        headroom_playing_ = true;
        btn_play_->set_glyph(IconButton::Glyph::Pause);
        log(QStringLiteral("playing the headroom preview as frames render (not real time)"));
        headroom_request(current_, true);
        return;
    }
    if (play_.isActive()) {
        stop_playback();
        return;
    }
    if (frames_.size() < 2 || !engine_) return;
    btn_play_->set_glyph(IconButton::Glyph::Pause);
    play_.setInterval(std::max(1, int(std::lround(1000.0 / play_fps()))));
    play_state_->playing = true;
    engine_->set_playing(true);
    play_t0_ = std::chrono::steady_clock::now();
    play_from_ = current_;
    play_shown_ = play_skipped_ = play_shown_window_ = 0;
    play_hold_.configure(play_fps(), steady_ ? kShotHoldSeconds : 0.0);
    start_feeder(current_ + 1);
    const int side = play_state_->side.load();
    log(QStringLiteral("playing at %1 fps").arg(play_fps(), 0, 'f', 3) +
        (use_model_ ? (side > 0 ? QStringLiteral("; the model at %1 px while playing, full size where you stop").arg(side)
                                : QStringLiteral("; the model at full size"))
                    : QStringLiteral("; analytic")));
    play_.start();
    play_status_.start();
}

double MainWindow::play_fps() const {
    return shot_video_ && shot_video_->fps ? std::clamp(*shot_video_->fps, 1.0, 120.0) : 24.0;
}

void MainWindow::start_feeder(int at) {
    std::lock_guard lock(feeder_slot_->mu);
    if (feeder_slot_->feeder) feeder_slot_->feeder->stop();
    feeder_slot_->feeder.reset();
#ifdef RUDRA_HAVE_STILL_DECODE
    if (!shot_video_ || at >= int(frames_.size())) return;
    auto f = MovieFeeder::start(shot_video_->path, play_fps(), at, int(frames_.size()), preview_max_side_);
    if (f) feeder_slot_->feeder = std::move(*f);
    else log("playback decodes frame by frame: " + qs(f.error().message));
#else
    (void)at;
#endif
}

void MainWindow::halt_playback() {
    if (headroom_playing_) {
        headroom_playing_ = false;
        if (headroom_) headroom_->pause();   // the frame on screen stays
    }
    const bool was = play_.isActive();
    play_.stop();
    play_status_.stop();
    if (btn_play_) btn_play_->set_glyph(IconButton::Glyph::Play);
    play_state_->playing = false;
    if (engine_) engine_->set_playing(false);
    if (feeder_slot_) {
        std::lock_guard lock(feeder_slot_->mu);
        if (feeder_slot_->feeder) feeder_slot_->feeder->stop();   // releases a loader waiting in take()
        feeder_slot_->feeder.reset();
    }
    if (was) {
        statusBar()->clearMessage();
        show_own_values();
    }
}

void MainWindow::show_own_values() {
    // Stopped: the frame on screen with its own shadow weight and anchor hold,
    // not the ones held while playing (a stop that does not step to the frame,
    // as a master render's, would otherwise keep them).
    shown_hold_ = measure_ ? measure_->anchor_hold : 1.0;
    if (!current_frame_ || current_frame_->scalars.shadow_weight == current_frame_->own_shadow_weight) return;
    auto cur = std::make_shared<Current>(*current_frame_);
    cur->scalars.shadow_weight = cur->own_shadow_weight;
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->set_frame({cur->sdr, cur->fields, cur->scalars, cur->model});
#endif
    current_frame_ = std::move(cur);
    run_stats();
}

void MainWindow::stop_playback() {
    halt_playback();
    const auto st = playback_stats();
    if (st.seconds > 0.0)
        log(QStringLiteral("played %1 frames in %2 s: %3 fps of %4%5")
                .arg(st.shown)
                .arg(st.seconds, 0, 'f', 1)
                .arg(st.shown / st.seconds, 0, 'f', 1)
                .arg(play_fps(), 0, 'f', 3)
                .arg(st.skipped ? QStringLiteral(", %1 skipped to keep time").arg(st.skipped) : QString()));
    // Where it stopped: the frame again, at full size if it was a playback proxy.
    step_to(current_);
}

MainWindow::PlaybackStats MainWindow::playback_stats() const {
    PlaybackStats s;
    s.shown = play_shown_;
    s.skipped = play_skipped_;
    s.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - play_t0_).count();
    return s;
}

void MainWindow::play_tick() {
    if (waiting_ || frames_.empty()) return;   // the frame asked for is not here yet: held
    const int n = int(frames_.size());
    int next = current_ + 1;
    if (realtime_) {
        // Keep time: the frame the clock has reached, skipping those the model has not.
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - play_t0_).count();
        const int due = play_from_ + int(t * play_fps());
        if (due > next) {
            play_skipped_ += due - next;
            next = due;
        }
    }
    if (next >= n) {   // the loop: from the top, the clock and the decoder again
        next %= n;
        play_t0_ = std::chrono::steady_clock::now();
        play_from_ = next;
        start_feeder(next + 1);
    }
    step_to(next);
}

void MainWindow::set_playback_side(int side) {
    play_state_->side = std::max(0, side);
    QSettings().setValue("playback/side", play_state_->side.load());
    sync_checks();
}

void MainWindow::set_playback_realtime(bool on) {
    realtime_ = on;
    QSettings().setValue("playback/realtime", on);
    sync_checks();
}

void MainWindow::set_steady_shots(bool on) {
    steady_ = on;
    QSettings().setValue("video/steady", on);
    if (play_.isActive()) play_hold_.configure(play_fps(), on ? kShotHoldSeconds : 0.0);
    log(on ? QStringLiteral("held steady within a shot while playing and in movie exports")
           : QStringLiteral("each frame's own values while playing and in movie exports"));
    if (headroom_ || headroom_routed()) route_headroom();   // the preview requires the hold off
    sync_checks();
}

void MainWindow::frame_ready(const ReadyFrame& f, const ModelConstants& model) {
    if (headroom_routed()) return;   // legacy pixels never stand in for the headroom preview
    if (f.index != current_) return;   // the engine already dropped stale ones; this is the UI's own check
    waiting_ = false;
    if (play_.isActive()) {
        ++play_shown_;
        ++play_shown_window_;
    }
    const QString name = QString::fromStdString(frame_name(std::size_t(f.index)));
    if (f.error) {
        log(name + ": " + QString::fromStdString(f.error->message));
        inspect_source();
        return;
    }
    FrameHeader header;
    header.resolution = QStringLiteral("%1x%2").arg(f.sdr->width()).arg(f.sdr->height()).toStdString();
    header.source_resolution = header.resolution;
    if (source_sizes_) {
        std::lock_guard lock(source_sizes_->mu);
        const auto [sw, sh] = source_sizes_->size[std::size_t(f.index)];
        if (sw > 0) header.source_resolution = QStringLiteral("%1x%2").arg(sw).arg(sh).toStdString();
    }
    header.tiled = false;   // the preview is one untiled pass, as the Studio's
    if (!f.from_cache) header.elapsed_s = f.infer_ms / 1000.0;
    else if (current_frame_ && current_frame_->header.elapsed_s) header.elapsed_s = current_frame_->header.elapsed_s;
    frame_proxy_ = f.fields->proxy;
    FrameScalars scalars = f.fields->scalars;
    const float own = scalars.shadow_weight;
    // Playing: the shadow weight held across the shot, as the export holds it.
    // And the static hold: the viewer holds still pixels' gain across the shot.
    PlaybackHold::PixelStep pixel;
    if (play_.isActive()) {
        scalars.shadow_weight = static_cast<float>(play_hold_.weight(f.index, *f.sdr, scalars.shadow_weight));
        pixel = play_hold_.pixel_step();
    }
    present_frame(*f.sdr, f.fields->fields, scalars, model, header, own, pixel);
    inspect_source();
    const auto st = engine_->stats();
    frame_info_ = QStringLiteral("%1  %2/%3  ·  %4")
                      .arg(name)
                      .arg(f.index + 1)
                      .arg(frames_.size())
                      .arg(f.from_cache ? QStringLiteral("cached")
                                        : QStringLiteral("decode %1 ms, infer %2 ms")
                                              .arg(f.decode_ms, 0, 'f', 0)
                                              .arg(f.infer_ms, 0, 'f', 0));
    frame_info_ += QStringLiteral("  ·  %1 in cache").arg(st.cached);
    show_status();
}

void MainWindow::show_status() {
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        const ViewerStatus s = viewer_->status();
        zoom_val_->setText(QStringLiteral("%1%").arg(s.zoom_percent));
        zoom_seg_->set_on(viewer_->viewport().scale <= 0.0 ? "fit" : (s.zoom_percent == 100 ? "actual" : ""));
        const bool hdr = s.target.path != OutputPath::SdrPqSimulation;
        findChild<QLabel*>("statusTime")->setText(QString::fromStdString(s.backend) + " · " +
                                                  QString::fromStdString(s.swapchain) +
                                                  (hdr ? QStringLiteral(" · peak %1 nits").arg(std::lround(s.target.peak_nits))
                                                       : QString()));
    }
#endif
    src_info_->setText(frame_info_.isEmpty() ? QStringLiteral("—") : frame_info_);
    update_badges();
}

// The page's timecode(): non-drop, from 01:00:00:00.
static QString timecode(int frame, double fps) {
    const int rate = std::max(1, int(std::floor(fps + 0.5)));
    const long long f = frame + static_cast<long long>(rate) * 3600;
    auto pad = [](long long v) { return QString::number(v).rightJustified(2, '0'); };
    return pad((f / (rate * 3600LL)) % 24) + ":" + pad((f / (rate * 60LL)) % 60) + ":" + pad((f / rate) % 60) + ":" +
           pad(f % rate);
}

void MainWindow::inspect_source() {
    std::optional<StillColour> still;
    if (!shot_video_ && source_sizes_) {
        std::lock_guard lock(source_sizes_->mu);
        if (current_ >= 0 && std::size_t(current_) < source_sizes_->colour.size())
            still = source_sizes_->colour[std::size_t(current_)];
    }
    source_panel_->inspect_source(!frames_.empty(), shot_video_.has_value(),
                                  shot_video_ ? shot_video_->stream : std::nullopt, still);
}

void MainWindow::sync_ui() {
    const auto& g = session_.grade;
    mode_seg_->set_on(QString::fromStdString(g.mode));
    // The exposure the frames and masters are composed at (analytic or the package's).
    source_panel_->sync(session_.source_curve(), g.calibration, frame_constants().corpus_ev, g.reference);
    inspect_source();
    {
        const QSignalBlocker b1(strength_), b2(peak_), b3(exposure_);
        exposure_->setValue(int(std::lround(g.exposure_ev * 20.0)));
        strength_->setValue(int(std::lround(g.strength * 20.0)));
        peak_->setValue(int(std::lround(session_.peak_ev * 2.0)));
    }
    strength_val_->setText(QString::number(g.strength, 'f', 2));
    exposure_val_->setText((g.exposure_ev > 0.0 ? QStringLiteral("+") : QString()) + QString::number(g.exposure_ev, 'f', 2));
    peak_val_->setText(QLocale(QLocale::English).toString(qlonglong(std::floor(session_.display_nits() + 0.5))));
    preserve_->set_on(g.preserve);
    preserve_->set_hint(g.preserve ? "do-no-harm" : "raw prediction");
    anchor_->set_on(session_.anchor);
    anchor_->set_hint(session_.anchor ? "conform" : "raw ITM level");
    {
        const QSignalBlocker b(anchor_knee_);
        anchor_knee_->setValue(int(std::lround(session_.anchor_knee * 100.0)));
    }
    anchor_knee_val_->setText(QString::number(session_.anchor_knee, 'f', 2));
    anchor_knee_->setEnabled(session_.anchor);
    carry_chroma_->set_on(session_.carry_chroma);
    carry_chroma_->set_hint(session_.carry_chroma ? "below the clip" : "per-channel");
    const bool aces = session_.container == "aces";
    container_field_->setText(aces ? "OpenEXR — ACES 2065-1" : "OpenEXR — linear Rec.2020");
    primaries_field_->setText(aces ? "AP0 (ST 2065-4)" : "Rec.2020");
    region_count_->setText(QString::number(g.regions.size()));
    regions_->sync();
    mask_panel_->sync();
    view_mode_->set_on(session_.wipe ? "#wipeBtn" : QString::fromStdString(session_.show));
    view_layer_->set_on(QString::number(session_.view_layer));
    guide_btn_->setChecked(guides_on_);
    changes_btn_->setChecked(session_.show_changes);
    invented_btn_->setChecked(session_.view_layer == 3);
    // Window
    rail_left_->setVisible(session_.rail_left);
    rail_right_->setVisible(session_.rail_right);
    scopes_->setVisible(session_.scopes_open && workspace_ != "simple");
    i_media_->set_on(session_.rail_left);
    i_inspector_->set_on(session_.rail_right);
    i_scopes_->set_on(session_.scopes_open && workspace_ != "simple");
    ws_->set_on(workspace_);
    tabs_->set_on(tab_);
    const int page = tab_ == "grade" ? 1 : tab_ == "deliver" ? 2 : 0;
    for (int i = 0; i < int(pages_.size()); ++i) pages_[std::size_t(i)]->setVisible(i == page);
    for (auto* n : notes_) n->setVisible(workspace_ != "simple");
    // Frames (drawFrames)
    const int n = int(frames_.size());
    {
        auto* title = findChild<QLabel*>("shotTitle");
        auto* sub = findChild<QLabel*>("shotSub");
        if (n == 0) {
            title->setText("No shot open");
            sub->setText("Drop frames, or open a folder");
        } else {
            title->setText(shot_title());
            QString res;
            if (current_frame_ && current_frame_->header.source_resolution)
                res = QString::fromStdString(*current_frame_->header.source_resolution).replace('x', QChar(0xd7)) +
                      QStringLiteral(" \u00b7 ");
            const double fps = shot_video_ ? shot_video_->fps.value_or(24.0) : 24.0;
            sub->setText(res + QStringLiteral("%1 frame%2 \u00b7 %3 fps \u00b7 Rec.709")
                                   .arg(n)
                                   .arg(n == 1 ? "" : "s")
                                   .arg(QString::number(fps, 'g', 5)));
        }
    }
    shot_count_->setText(QString::number(n));
    frames_empty_->setVisible(n == 0);
    if (auto* dz = findChild<QWidget*>("dropzone")) dz->setVisible(n == 0);   // shell.js: tucked once frames open
    tc_->setText(timecode(n ? current_ : 0, 24.0));
    scrub_->set_position(n > 1 ? double(current_) / double(n - 1) : 0.0);
    for (auto* b : {btn_prev_, btn_play_, btn_next_}) b->setEnabled(n > 1);
    const bool ready = n > 0 && backend_ != nullptr;
    btn_master_->setEnabled((ready && pending_reason("master").isEmpty()) || mastering());
    btn_reprocess_->setEnabled(ready);
    if (viewer_stack_->count() > 1) viewer_stack_->setCurrentIndex(n > 0 ? 1 : 0);
}

void MainWindow::keyPressEvent(QKeyEvent* e) {
    const bool modified = e->modifiers() & (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier);
    const bool shift = e->modifiers() & Qt::ShiftModifier;
    std::string key;
    switch (e->key()) {
        case Qt::Key_Left: key = "ArrowLeft"; break;
        case Qt::Key_Right: key = "ArrowRight"; break;
        case Qt::Key_Escape: key = "Escape"; break;
        case Qt::Key_B: key = "b"; break;
        default: break;
    }
    if (key.empty() || modified) {
        QMainWindow::keyPressEvent(e);
        return;
    }
    session_.key_down(key, shift, false, e->isAutoRepeat());
}

void MainWindow::keyReleaseEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_B && !e->isAutoRepeat()) session_.key_up("b");
    QMainWindow::keyReleaseEvent(e);
}

void MainWindow::set_workspace(const QString& mode) {
    workspace_ = mode == "simple" ? "simple" : "full";
    sync_ui();
}

void MainWindow::show_tab(const QString& tab) {
    tab_ = tab;
    sync_ui();
}

void MainWindow::show_scopes(const ScopeData& s, std::optional<double> maxcll,
                             const std::vector<std::uint8_t>& vector_rgba) {
    wave_->set_data(s, maxcll);
    hist_->set_data(s, maxcll);
    vector_->set_image(vector_rgba);
}

void MainWindow::clear_scopes() {
    wave_->clear();
    hist_->clear();
    vector_->clear();
}

void MainWindow::present_frame(SdrImage sdr, Fields fields, FrameScalars scalars, ModelConstants model,
                               FrameHeader header, std::optional<float> own_shadow_weight,
                               [[maybe_unused]] PlaybackHold::PixelStep pixel) {
    auto cur = std::make_shared<Current>();
    cur->baseline_source = session_.source_curve();
    cur->baseline_calibration = session_.grade.calibration;
    cur->baseline_reference = session_.grade.reference;
    // Not while playing: the UI thread has a frame's time, and the measure
    // builds the baseline on its own thread when this one is missing.
    if (!play_.isActive())
        cur->baseline = std::make_shared<const NetworkLinearImage>(corrected_baseline(
            sdr, model.corpus_ev, baseline_curve_params(scalars, session_.composite_params(), model.corpus_ev)));
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        ViewerHold hold;
        hold.on = pixel.on;
        hold.strength = pixel.strength;
        hold.reset = pixel.reset;
        hold.frame = current_;
        viewer_->set_frame({sdr, fields, scalars, model, hold});
        viewer_->set_composite(session_.composite_params());
    }
#endif
    cur->sdr = std::move(sdr);
    cur->fields = std::move(fields);
    cur->scalars = std::move(scalars);
    cur->model = model;
    cur->header = std::move(header);
    cur->index = current_;
    frame_ready_ok_ = true;
    cur->own_shadow_weight = own_shadow_weight.value_or(cur->scalars.shadow_weight);
    cur->hold_shot = play_hold_.shot();
    current_frame_ = std::move(cur);
    note_frame_in_strip(current_);
    run_stats();   // adopt(): computeStats at once for a new frame
}

void MainWindow::schedule_stats() {
    if (current_frame_ || (headroom_routed() && headroom_shown_)) stats_timer_.start();
}

void MainWindow::run_stats() {
    if (headroom_routed()) {
        // E2: the headroom frame's own readouts and scopes, off the UI thread.
        if (!headroom_shown_) return;
        if (stats_running_) {
            stats_again_ = true;
            return;
        }
        stats_running_ = true;
        const int gen = ++stats_gen_;
        auto model = headroom_shown_->model, baseline = headroom_shown_->baseline;
        const int index = headroom_shown_->index;
        QPointer<MainWindow> self(this);
        if (stats_worker_.joinable()) stats_worker_.join();
        stats_worker_ = std::thread([self, model, baseline, gen, index] {
            auto m = std::make_shared<const FrameMeasure>(measure_headroom(*model, *baseline));
            QMetaObject::invokeMethod(qApp, [self, m, gen, index] {
                if (!self) return;
                self->stats_running_ = false;
                if (gen == self->stats_gen_ && self->headroom_routed() && self->headroom_shown_ &&
                    self->headroom_shown_->index == index)
                    self->apply_measure(m, -1, -1);
                if (self->stats_again_) {
                    self->stats_again_ = false;
                    self->run_stats();
                }
            });
        });
        return;
    }
    if (!current_frame_) return;
    if (stats_running_) {   // the page's statsPending: once more when this one is done
        stats_again_ = true;
        return;
    }
    stats_running_ = true;
    const int gen = ++stats_gen_;
    auto frame = current_frame_;
    const CompositeParams params = session_.composite_params();
    QPointer<MainWindow> self(this);
    // Joined, not detached: a detached measurement outlived its window and, at
    // exit, the libraries it uses (a destroyed mutex aborts on macOS).
    if (stats_worker_.joinable()) stats_worker_.join();   // done: stats_running_ was cleared after it posted
    // The cached baseline is the one of the source it was built with; after a
    // source change the measure rebuilds it from the params (3.1).
    const NetworkLinearImage* cached = frame->baseline_source == params.source &&
                                               frame->baseline_calibration == params.calibration &&
                                               frame->baseline_reference == params.reference
                                           ? frame->baseline.get() : nullptr;
    const double knee = session_.anchor_knee;
    stats_worker_ = std::thread([self, frame, params, gen, cached, knee] {
        auto m = std::make_shared<const FrameMeasure>(
            measure_frame(frame->sdr, frame->fields, frame->scalars, frame->model, params, cached, knee));
        const int index = frame->index, shot = frame->hold_shot;
        QMetaObject::invokeMethod(qApp, [self, m, gen, index, shot] {
            if (!self) return;
            self->stats_running_ = false;
            if (gen == self->stats_gen_) self->apply_measure(m, index, shot);
            if (self->stats_again_) {
                self->stats_again_ = false;
                self->run_stats();
            }
        });
    });
}

void MainWindow::measure_now() {
    if (headroom_routed()) {
        if (!headroom_shown_) return;
        ++stats_gen_;
        apply_measure(std::make_shared<const FrameMeasure>(measure_headroom(*headroom_shown_->model, *headroom_shown_->baseline)));
        return;
    }
    if (!current_frame_) return;
    const auto& f = *current_frame_;
    ++stats_gen_;   // anything in flight is now stale
    const CompositeParams params = session_.composite_params();
    apply_measure(std::make_shared<const FrameMeasure>(measure_frame(
        f.sdr, f.fields, f.scalars, f.model, params,
        f.baseline_source == params.source && f.baseline_calibration == params.calibration &&
                f.baseline_reference == params.reference
            ? f.baseline.get() : nullptr,
        session_.anchor_knee)), play_.isActive() ? f.index : -1, f.hold_shot);
}

void MainWindow::fill_rows(QWidget* ms, const std::vector<MetricRow>& rows) {
    auto* v = ms->layout();
    while (QLayoutItem* it = v->takeAt(0)) {
        delete it->widget();
        delete it;
    }
    for (const auto& r : rows) {
        auto* row = new QWidget(ms);
        row->setProperty("role", "ro");
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 2, 0, 2);
        h->setSpacing(3);
        auto* k = new QLabel(QString::fromStdString(r.k), row);
        k->setProperty("role", "key");
        auto* val = new QLabel(QString::fromStdString(r.v), row);
        val->setProperty("role", "value");
        if (r.warn) val->setProperty("state", "warn");
        val->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        auto* u = new QLabel(QString::fromStdString(r.u), row);
        u->setProperty("role", "unit");
        h->addWidget(k);
        h->addStretch(1);
        h->addWidget(val);
        h->addWidget(u);
        v->addWidget(row);
    }
}

void MainWindow::apply_measure(std::shared_ptr<const FrameMeasure> m, int index, int hold_shot) {
    measure_ = std::move(m);
    const auto& f = *measure_;
    // Playing: the anchor's hold held across the shot, as the export holds it.
    shown_hold_ = index >= 0 && play_.isActive() ? play_hold_.anchor(index, f.anchor_hold, hold_shot) : f.anchor_hold;
#ifdef RUDRA_APP_VIEWER
    if (viewer_ && session_.anchor) {
        auto v = viewer_->view();
        if (v.anchor_hold != shown_hold_ || v.anchor_knee != f.anchor_knee) {
            v.anchor = true;
            v.anchor_knee = f.anchor_knee;
            v.anchor_hold = shown_hold_;
            viewer_->set_view(v);
        }
    }
#endif
    const MetricsText t = metrics_text(f.frame_metrics(), current_frame_ ? std::optional(current_frame_->header)
                                                                          : std::nullopt,
                                       f.coverage.clipped_pct);
    fill_rows(findChild<QWidget*>("measA"), t.a);
    fill_rows(findChild<QWidget*>("measB"), t.b);
    findChild<QLabel*>("statusMask")->setText(QString::fromStdString(t.status_mask));
    findChild<QLabel*>("statusTime")->setText(QString::fromStdString(t.status_time));
    src_info_->setText(QString::fromStdString(t.src_info));
    // The headroom path has no SDR clip measure: no clip bar or lane rather than a false 0 %.
    if (headroom_routed()) {
        clip_bar_->hide();
    } else {
        clip_bar_->set(f.coverage.clipped_pct, f.coverage.highlight_pct);
        clip_bar_->show();
    }
    show_scopes(f.measured.scopes, f.measured.metrics.maxcll, f.vector_rgba);
    // The clipping lane: gold where the SDR clipped, violet where the network
    // found crushed shadows (its shadow mask over 1 % of the frame).
    if (!frames_.empty() && !headroom_routed()) {
        const auto lane = f.coverage.clipped_pct > 0.01 ? ScrubBar::Lane::Highlights
                          : f.coverage.shadow_pct > 1.0 ? ScrubBar::Lane::Shadows
                                                        : ScrubBar::Lane::None;
        scrub_->set_lane(current_, lane);
        int hi = 0, sh = 0;
        for (int i = 0; i < scrub_->count(); ++i) {
            hi += scrub_->lane(i) == ScrubBar::Lane::Highlights;
            sh += scrub_->lane(i) == ScrubBar::Lane::Shadows;
        }
        findChild<QLabel*>("colHighlights")->setText(QString::number(hi));
        findChild<QLabel*>("colShadows")->setText(QString::number(sh));
    }
    update_pipe();
    update_compare();
    update_badges();
    refresh_enabled();
}

void MainWindow::update_compare() {
    if (!measure_) {
        compare_line_->setText(" ");
        compare_fit_->hide();
        return;
    }
    // The SDR view exposes to the view peak; an HDR swapchain clips at the panel.
    double panel = std::numeric_limits<double>::infinity();
#ifdef RUDRA_APP_VIEWER
    if (viewer_ && viewer_->view().target.path != OutputPath::SdrPqSimulation) panel = viewer_->view().target.peak_nits;
#endif
    CompareText t = compare_text(measure_->compare, session_.display_nits(), panel, peak_->minimum() / 2.0,
                                 peak_->maximum() / 2.0);
    if (session_.view_layer == 3) {   // the Invented layer: the line is its legend
        t = CompareText{};
        t.line = support_text(measure_->support);
    }
    compare_line_->setText(QString::fromStdString(t.line));
    const char* state = t.warn ? "warn" : "";
    if (compare_line_->property("state").toString() != state) {
        compare_line_->setProperty("state", state);
        compare_line_->style()->unpolish(compare_line_);
        compare_line_->style()->polish(compare_line_);
    }
    compare_fit_ev_ = t.fit_peak_ev;
    compare_fit_->setText(QString::fromStdString(t.fit_label));
    compare_fit_->setVisible(t.fit_shown);
}

void MainWindow::update_pipe() {
    std::optional<double> maxcll;
    if (measure_) maxcll = measure_->measured.metrics.maxcll;
    const PipeText t = pipe_text(session_.container, session_.display_nits(), maxcll,
                                 current_frame_ ? std::optional(current_frame_->header) : std::nullopt);
    findChild<QLabel*>("pipeIn")->setText(QString::fromStdString(t.in));
    findChild<QLabel*>("pipeWorking")->setText(QString::fromStdString(t.working));
    view_transform_->setText(QString::fromStdString(t.view));
    findChild<QLabel*>("pipeMaster")->setText(QString::fromStdString(t.master).replace("EXR, half", "EXR, float32"));
    auto* warn = findChild<QLabel*>("pipeWarn");
    warn->setText(QString::fromStdString(t.warn));
    warn->setVisible(t.warn_shown);
}

void MainWindow::probe_pixel(std::optional<std::pair<double, double>> px, QPoint global) {
    std::optional<ProbeInput> p;
    if (px && measure_) p = measure_->probe_at(px->first, px->second);
    const ProbePanelText t = probe_panel(p);
    auto set = [this](const char* id, const std::string& text, const char* state) {
        auto* l = findChild<QLabel*>(id);
        l->setText(QString::fromStdString(text));
        if (l->property("state").toString() != state) {
            l->setProperty("state", state);
            l->style()->unpolish(l);
            l->style()->polish(l);
        }
    };
    set("probeXY", t.xy, "");
    set("probeNits", t.nits, t.idle ? "idle" : "");
    findChild<QLabel*>("probeNitsUnit")->setVisible(!t.idle);
    set("probeDelta", t.delta, t.delta_class.find("idle") != std::string::npos ? "idle" : "");
    set("probeSrc", t.src, t.src_class.c_str());
    set("probeBase", t.base, "");
    set("probeModel", t.model, t.model_class.c_str());
    set("probeMask", t.mask, "");
    // The floating box.
    if (!p) {
        probe_box_->hide();
        return;
    }
    auto* grid = static_cast<QGridLayout*>(probe_box_->layout());
    while (QLayoutItem* it = grid->takeAt(0)) {
        delete it->widget();
        delete it;
    }
    int r = 0;
    for (const auto& row : rudra::probe_box(*p)) {
        auto* k = new QLabel(QString::fromStdString(row.k), probe_box_);
        k->setProperty("role", "key");
        auto* v = new QLabel(QString::fromStdString(row.v), probe_box_);
        v->setProperty("role", "value");
        if (!row.cls.empty()) v->setProperty("state", QString::fromStdString(row.cls));
        grid->addWidget(k, r, 0);
        grid->addWidget(v, r, 1);
        ++r;
    }
    probe_box_->adjustSize();
    // showProbe's placing: 16 px right of and below the cursor, flipped to the
    // other side at the viewer's edges, at least 4 px in.
    QWidget* area = viewer_stack_;
    const QPoint origin = area->mapToGlobal(QPoint(0, 0));
    const int bw = std::max(196, probe_box_->width()), bh = probe_box_->height();
    int lx = global.x() - origin.x() + 16, ly = global.y() - origin.y() + 16;
    if (lx + bw > area->width()) lx = global.x() - origin.x() - bw - 16;
    if (ly + bh > area->height()) ly = global.y() - origin.y() - bh - 16;
    probe_box_->move(origin + QPoint(std::max(4, lx), std::max(4, ly)));
    probe_box_->show();
}

MasterRequest MainWindow::master_request() const {
    // The settings of the moment: params(), the Deliver checks, the container.
    MasterRequest q;
    q.analytic = !use_model_;
    q.checkpoint = use_model_ && manifest_ ? manifest_->name : std::string();
    q.checkpoint_sha256 = use_model_ && manifest_ ? manifest_->source_sha256 : std::string();
    q.clip_fill = session_.grade.clip_fill;
    q.clip_peaks = session_.grade.clip_peaks;
    q.preserve_outside = session_.grade.preserve;
    q.recovery_mode = session_.grade.mode;
    q.strength = session_.grade.strength;
    q.exposure_ev = session_.grade.exposure_ev;
    for (const auto& r : session_.grade.regions) q.regions.push_back({r.label, r.low_nits, r.high_nits, r.ev});
    q.anchor = session_.anchor;
    q.anchor_knee = session_.anchor_knee;
    q.carry_chroma = session_.carry_chroma;
    q.container = session_.container;
    q.exr_half = false;  // Grading masters retain float32 precision and range.
    q.source_curve = session_.grade.source;
    q.calibration = session_.calibration_points();
    q.reference = session_.grade.reference;
    q.masks = session_.composite_params().masks;
    return q;
}

const SdrImage* MainWindow::frame_sdr() const { return current_frame_ ? &current_frame_->sdr : nullptr; }

void MainWindow::master(PrepareMasterFrame prepare, std::size_t count) {
    auto* status = findChild<QLabel*>("renderStatus");
    auto say = [status](const QString& t) {
        status->setText(t);
        status->setVisible(!t.isEmpty());
    };
    const bool injected = bool(prepare);
    if (refuse_headroom_export(QStringLiteral("Master EXR"))) return;
    if (!injected && (needs_model() || frames_.empty())) return;   // !state.live || !current()
    if (mastering()) return;                                    // state.busy
    auto* dir = findChild<QLineEdit*>("renderDir");
    const bool to_movie = !injected && frames_movie_.has_value();   // export_frames_to_movie's frames
    const QString folder = to_movie ? from_path(frames_movie_->frames_dir) : dir->text().trimmed();
    if (folder.isEmpty()) {
        say("Choose a render folder first.");
        dir->setFocus();
        return;
    }
    const bool sequence = to_movie || findChild<QComboBox*>("renderMode")->currentData().toString() == "sequence";
    std::vector<std::filesystem::path> sources;
    if (!injected) sources = sequence ? frames_ : std::vector<std::filesystem::path>{frames_[std::size_t(current_)]};
    const std::size_t n = injected ? (sequence ? count : 1) : sources.size();
    RenderPlan plan;
    plan.render_dir = folder.toStdString();
    plan.render_name = to_movie ? frames_movie_->name : findChild<QLineEdit*>("renderName")->text().trimmed().toStdString();
    plan.render_count = int(n);
    plan.frame_start = to_movie ? 0 : findChild<QSpinBox*>("renderStart")->value();
    plan.sequence = sequence;
    auto targets = master_targets(plan);
    if (!targets) {
        say(QStringLiteral("Stopped after 0 / %1: %2").arg(n).arg(QString::fromStdString(targets.error().message)));
        log("Render stopped: " + QString::fromStdString(targets.error().message));
        return;
    }
    log("Render destination: " + QString::fromStdString(targets->front().string()) +
        (targets->size() > 1 ? " \u2026 " + QString::fromStdString(targets->back().string()) : QString()));

    MasterRequest q = master_request();
    if (to_movie) q.container = "linear";   // Rec.2020 scene linear, what deliver reads
    const ModelConstants model = frame_constants();
    if (!prepare) {
#ifdef RUDRA_HAVE_STILL_DECODE
        prepare = [sources, backend = use_model_ ? backend_.get() : nullptr, mu = backend_mutex_, q](std::size_t i) -> Result<MasterFrame> {
            auto file = ensure_frame_file(sources[i]);
            if (!file) return file.error();
            auto d = decode_sdr_file(*file);
            if (!d) return d.error();
            if (!backend) {   // the analytic reconstruction: no network, the clip fill unless it is off
                Fields fields = analytic_fields_for(d->rgb, q);
                return MasterFrame{std::move(d->rgb), d->bits, std::move(fields), FrameScalars{}, *file, false,
                                   primaries_id(d->colour.primaries)};
            }
            std::lock_guard lock(*mu);
            auto fr = infer_frame(*backend, d->rgb, TileConfig{0, 0});   // full resolution, untiled
            if (!fr) return fr.error();
            return MasterFrame{std::move(d->rgb), d->bits, std::move(fr->fields), std::move(fr->scalars), *file, fr->tiled,
                               primaries_id(d->colour.primaries)};
        };
#else
        say("This build has no still decode (RUDRA_WITH_OPENCV=OFF).");
        return;
#endif
    }
    halt_playback();   // stopPlay()
    btn_master_->setText("Rendering\u2026");
    btn_master_->setToolTip("Click to stop after the frame being written");
    QPointer<MainWindow> self(this);
    master_job_ = std::make_unique<MasterJob>(model, q, *targets, std::move(prepare));
    master_job_->start(
        [self, n](std::size_t i) {
            QMetaObject::invokeMethod(qApp, [self, i, n] {
                if (self) self->findChild<QLabel*>("renderStatus")->setText(QStringLiteral("Rendering %1 / %2").arg(i + 1).arg(n));
                if (self) self->findChild<QLabel*>("renderStatus")->show();
            });
        },
        [self](const MasterProgress& p) {
            const QString line = "Saved " + QString::fromStdString(p.last.exr.string()) +
                                 QStringLiteral(" \u00b7 %1x%2").arg(p.last.width).arg(p.last.height);
            QMetaObject::invokeMethod(qApp, [self, line] {
                if (self) self->log(line);
            });
        },
        [self, folder](const MasterOutcome& o) {
            QMetaObject::invokeMethod(qApp, [self, o, folder] {
                if (self) self->master_finished(o, folder);
            });
        });
    refresh_enabled();
    sync_ui();   // the button stays live: a click stops the render
}

void MainWindow::master_finished(const MasterOutcome& o, const QString& folder) {
    auto* status = findChild<QLabel*>("renderStatus");
    if (o.error || o.cancelled) {
        const QString why = o.error ? QString::fromStdString(o.error->message) : QStringLiteral("cancelled");
        status->setText(QStringLiteral("Stopped after %1 / %2: %3").arg(o.completed).arg(o.total).arg(why));
        log("Render stopped: " + why);
    } else {
        status->setText(QStringLiteral("Rendered %1 frame(s) to %2").arg(o.completed).arg(folder));
    }
    status->show();
    std::optional<FramesMovie> movie;
    if (frames_movie_) {
        if (!o.error && !o.cancelled) movie = std::move(*frames_movie_);
        frames_movie_.reset();
    }
    delivered_ += int(o.completed);
    findChild<QLabel*>("colDelivered")->setText(QString::number(delivered_));
    btn_master_->setText("Master EXR");   // busy(false)
    btn_master_->setToolTip({});
    refresh_enabled();
    sync_ui();
    // The movie's encode goes last, so its "Encoding…" button (a click stops
    // the movie) is what shows while it runs, not "Master EXR".
    if (movie) encode_frames_movie(std::move(*movie));
}

// ---------------------------------------------------------------------------
// Step 11: copies, drop to open, recent shots, settings
// ---------------------------------------------------------------------------

void MainWindow::copy_text(const QString& label, const std::string& text) {
    QGuiApplication::clipboard()->setText(QString::fromStdString(text));
    log("copied " + label + " to the clipboard");
}

std::string MainWindow::delivery_text() const {
    DeliveryRecord d;
    if (current_frame_) {
        // state.header: what came back with the frame on screen.
        if (manifest_) d.checkpoint = manifest_->name;
        d.resolution = current_frame_->header.resolution;
    }
    if (!frames_.empty()) d.frame = frame_name(std::size_t(current_));
    d.aces = session_.container == "aces";
    d.mode = session_.grade.mode;
    d.strength = session_.grade.strength;
    d.preserve = session_.grade.preserve;
    for (const auto& r : session_.grade.regions) d.regions.push_back({r.label, r.low_nits, r.high_nits, r.ev});
    if (measure_) d.metrics = std::pair{measure_->measured.metrics, measure_->compose_ms};
    return delivery_value(d).stringify();
}

void MainWindow::add_files(const std::vector<std::filesystem::path>& files) {
    if (files.empty()) return;
    if (needs_model()) {
        log("Open a model package first, or turn off Reconstruct > Use the model.");
        return;
    }
    auto all = frames_;
    const std::size_t start = all.size();
    all.insert(all.end(), files.begin(), files.end());
    log(QStringLiteral("added %1 frame%2").arg(files.size()).arg(files.size() == 1 ? "" : "s"));
    for (const auto& f : files) remember_source(QString::fromStdString(f.string()));
    // The project's shot: stills are kept by name. A movie stays the shot
    // (stills added to one are not saved with it); a folder becomes its frames.
    if (shot_kind_ != "movie") {
        if (shot_kind_ != "files") shot_sources_ = frames_;
        shot_kind_ = "files";
        for (const auto& f : files) shot_sources_.push_back(std::filesystem::absolute(f));
    }
    mark_dirty();
    // select(state.frames.length === files.length ? 0 : start)
    start_engine(std::move(all), all.size() == files.size() ? 0 : int(start));
}

void MainWindow::open_paths(const QStringList& paths) {
    std::vector<std::filesystem::path> files;
    for (const auto& p : paths) {
        const QFileInfo fi(p);
        if (fi.isDir()) {
            if (QFileInfo(QDir(p).filePath("manifest.json")).isFile()) use_model(p.toStdString(), backend_choice_);
            else open_source(p, true);
            return;   // one folder is one shot, or one model
        }
        if (fi.isFile() && fi.suffix().compare("rudra", Qt::CaseInsensitive) == 0) {
            // A project is a session of its own.
            if (confirm_replace(false) && !open_project(to_path(p)) && isVisible())
                QMessageBox::warning(this, "RUDRA", problem_);
            return;
        }
        if (fi.isFile()) {
            const std::string suf = fi.suffix().toLower().prepend('.').toStdString();
            if (std::find(video_suffixes().begin(), video_suffixes().end(), suf) != video_suffixes().end()) {
                open_source(p);   // a movie is a shot of its own
                return;
            }
            files.emplace_back(p.toStdString());
        }
    }
    add_files(files);
}

void MainWindow::set_drop_hot(bool hot) {
    if (auto* dz = findChild<QWidget*>("dropzone")) {
        dz->setProperty("state", hot ? "hot" : "");
        dz->style()->unpolish(dz);
        dz->style()->polish(dz);
        for (auto* c : dz->findChildren<QWidget*>()) {
            c->style()->unpolish(c);
            c->style()->polish(c);
        }
    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent* e) {
    if (!e->mimeData()->hasUrls()) return;
    e->acceptProposedAction();
    set_drop_hot(true);
}

void MainWindow::dragLeaveEvent(QDragLeaveEvent*) { set_drop_hot(false); }

void MainWindow::dropEvent(QDropEvent* e) {
    set_drop_hot(false);
    QStringList paths;
    for (const auto& u : e->mimeData()->urls())
        if (u.isLocalFile()) paths << u.toLocalFile();
    if (paths.isEmpty()) return;
    e->acceptProposedAction();
    open_paths(paths);
}

namespace {
constexpr int kRecentMax = 10;
}

QStringList MainWindow::recent_sources() const { return QSettings().value("recent/sources").toStringList(); }

void MainWindow::remember_source(const QString& path) {
    QSettings st;
    QStringList r = st.value("recent/sources").toStringList();
    const QString p = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    r.removeAll(p);
    r.prepend(p);
    while (r.size() > kRecentMax) r.removeLast();
    st.setValue("recent/sources", r);
}

void MainWindow::fill_recent() {
    if (!recent_menu_) return;
    for (QAction* a : recent_menu_->actions())
        if (a->objectName().startsWith("recent:")) {
            recent_menu_->removeAction(a);
            a->deleteLater();
        }
    QAction* clear = action("recent-clear");
    const QStringList r = recent_sources();
    int i = 0;
    for (const auto& p : r) {
        const QFileInfo fi(p);
        auto* a = new QAction(QStringLiteral("%1  ·  %2").arg(fi.fileName(), fi.absolutePath()), recent_menu_);
        a->setObjectName(QStringLiteral("recent:%1").arg(i++));
        a->setData(p);
        // A project brings its own model; frames need one open.
        a->setEnabled(fi.exists() && (!needs_model() || fi.suffix().compare("rudra", Qt::CaseInsensitive) == 0));
        // As a drop: a folder or a movie opens as a shot, a still is added, a project opens.
        connect(a, &QAction::triggered, this, [this, p] { open_paths({p}); });
        recent_menu_->insertAction(clear, a);
    }
    if (clear) clear->setEnabled(!r.isEmpty());
}

void MainWindow::save_settings() const {
    QSettings st;
    st.setValue("window/geometry", saveGeometry());
    st.setValue("window/workspace", workspace_);
    st.setValue("window/tab", tab_);
    st.setValue("window/railLeft", session_.rail_left);
    st.setValue("window/railRight", session_.rail_right);
    st.setValue("window/scopes", session_.scopes_open);
    st.setValue("deliver/container", QString::fromStdString(session_.container));
    if (auto* e = findChild<QLineEdit*>("renderDir")) st.setValue("deliver/renderDir", e->text());
    if (auto* e = findChild<QLineEdit*>("renderName")) st.setValue("deliver/renderName", e->text());
    if (auto* c = findChild<QComboBox*>("renderMode")) st.setValue("deliver/renderMode", c->currentIndex());
    if (auto* sb = findChild<QSpinBox*>("renderStart")) st.setValue("deliver/renderStart", sb->value());
    save_queues();
}

void MainWindow::save_queues() const {
    QStringList q;
    for (const auto& e : queues_->entries())
        q << QString::fromStdString(e.queue.string()) + "\t" + QString::fromStdString(e.title);
    QSettings().setValue("queue/entries", q);
}

void MainWindow::restore_settings() {
    QSettings st;
    if (st.contains("window/geometry")) restoreGeometry(st.value("window/geometry").toByteArray());
    if (st.contains("window/workspace")) set_workspace(st.value("window/workspace").toString());
    if (st.contains("window/tab")) show_tab(st.value("window/tab").toString());
    const bool l = st.value("window/railLeft", session_.rail_left).toBool(),
               r = st.value("window/railRight", session_.rail_right).toBool(),
               sc = st.value("window/scopes", session_.scopes_open).toBool();
    if (l != session_.rail_left) run("rail-left");
    if (r != session_.rail_right) run("rail-right");
    if (sc != session_.scopes_open) run("scopes");
    const QString c = st.value("deliver/container").toString();
    if (c == "aces" || c == "linear") session_.set_container(c.toStdString());
    if (auto* e = findChild<QLineEdit*>("renderDir"); e && st.contains("deliver/renderDir"))
        e->setText(st.value("deliver/renderDir").toString());
    if (auto* e = findChild<QLineEdit*>("renderName"); e && st.contains("deliver/renderName"))
        e->setText(st.value("deliver/renderName").toString());
    if (auto* cb = findChild<QComboBox*>("renderMode"); cb && st.contains("deliver/renderMode"))
        cb->setCurrentIndex(st.value("deliver/renderMode").toInt());
    if (auto* sb = findChild<QSpinBox*>("renderStart"); sb && st.contains("deliver/renderStart"))
        sb->setValue(st.value("deliver/renderStart").toInt());
    {   // the exports queued before: listed again, their state read from their files
        std::vector<QueueEntry> entries;
        for (const QString& line : st.value("queue/entries").toStringList()) {
            const auto parts = line.split('\t');
            if (parts.size() == 2 && QFileInfo(parts[0]).isFile())
                entries.push_back({parts[0].toStdString(), parts[1].toStdString()});
        }
        queues_->set_entries(std::move(entries));
    }
    sync_checks();
    sync_ui();
}

void MainWindow::closeEvent(QCloseEvent* e) {
    if (!confirm_replace(true)) {
        e->ignore();
        return;
    }
    save_settings();
    if (autosave_timer_.isActive()) autosave_now();
    QMainWindow::closeEvent(e);
}


// ---------------------------------------------------------------------------
// The Pro chrome
// ---------------------------------------------------------------------------

namespace {

// A small picture of an SDR frame: its display-encoded codes as they are.
QImage sdr_thumb(const SdrImage& sdr, int height) {
    const auto& b = sdr.buffer();
    if (b.width() <= 0 || b.height() <= 0) return {};
    const int h = std::max(1, std::min(height, b.height()));
    const int w = std::max(1, int(std::lround(double(b.width()) * h / b.height())));
    QImage img(w, h, QImage::Format_RGB888);
    for (int y = 0; y < h; ++y) {
        uchar* row = img.scanLine(y);
        const int sy = std::min(b.height() - 1, int((y + 0.5) * b.height() / h));
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(b.width() - 1, int((x + 0.5) * b.width() / w));
            for (int c = 0; c < 3; ++c)
                row[x * 3 + c] = uchar(std::clamp(int(std::lround(b.at(c, sy, sx) * 255.0f)), 0, 255));
        }
    }
    return img;
}

}  // namespace

void MainWindow::note_frame_in_strip(int index) {
    if (!current_frame_ || index < 0 || index >= scrub_->count()) return;
    const QImage t = sdr_thumb(current_frame_->sdr, 88);
    scrub_->set_thumb(index, t);
    if (index == 0 || first_thumb_.isNull()) first_thumb_ = t;
    if (index == 0 || scrub_->count() == 1) refresh_library();   // the shot row's picture
}

void MainWindow::refresh_library() {
    auto* shots = findChild<QWidget*>("shots");
    if (!shots) return;
    for (auto* c : shots->findChildren<QWidget*>(Qt::FindDirectChildrenOnly)) c->deleteLater();
    auto* v = static_cast<QVBoxLayout*>(shots->layout());
    auto add_row = [&](const QString& name, const QString& sub, const QImage& thumb, bool sel, bool hdr,
                       const QString& open) {
        auto* r = new QWidget(shots);
        r->setProperty("role", "shot");
        r->setProperty("sel", sel);
        r->setAttribute(Qt::WA_StyledBackground, true);
        auto* h = new QHBoxLayout(r);
        h->setContentsMargins(8, 6, 8, 6);
        h->setSpacing(10);
        auto* pic = new QLabel(r);
        pic->setFixedSize(46, 30);
        if (!thumb.isNull())
            pic->setPixmap(QPixmap::fromImage(thumb.scaled(46, 30, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation)
                                                  .copy(0, 0, 46, 30)));
        else pic->setStyleSheet({});
        h->addWidget(pic);
        auto* text = new QWidget(r);
        auto* tv = new QVBoxLayout(text);
        tv->setContentsMargins(0, 0, 0, 0);
        tv->setSpacing(0);
        auto* n = new QLabel(name, text);
        n->setProperty("role", "shot-name");
        auto* s = new QLabel(sub, text);
        s->setProperty("role", "shot-sub");
        tv->addWidget(n);
        tv->addWidget(s);
        h->addWidget(text, 1);
        if (hdr) {
            auto* tag = new QLabel("HDR", r);
            tag->setProperty("role", "hdr-tag");
            h->addWidget(tag);
        }
        if (!open.isEmpty()) {
            r->setCursor(Qt::PointingHandCursor);
            r->setToolTip(open);
            r->installEventFilter(this);
            r->setProperty("open", open);
        }
        v->addWidget(r);
    };
    int rows = 0;
    QString current;
    if (!frames_.empty()) {
        const bool folder = frames_.size() > 1;
        const auto& first = frames_.front();
        current = QString::fromStdString(shot_video_ ? shot_video_->path.string() : (folder ? first.parent_path() : first).string());
        QString res;
        if (current_frame_ && current_frame_->header.source_resolution)
            res = QString::fromStdString(*current_frame_->header.source_resolution).replace('x', QChar(0xd7)) + " · ";
        const bool hdr = measure_ && measure_->measured.metrics.maxcll > 203.0;
        add_row(shot_title(),
                res + QStringLiteral("%1 f").arg(frames_.size()), scrub_->count() ? thumb_of_first() : QImage(), true,
                hdr, {});
        ++rows;
    }
    // Then the shots opened before (Open recent), to reopen with a click.
    for (const auto& p : recent_sources()) {
        if (rows >= 6) break;
        if (QDir::cleanPath(p) == QDir::cleanPath(current)) continue;
        const QFileInfo fi(p);
        if (!fi.exists()) continue;
        add_row(fi.fileName(), fi.isDir() ? QStringLiteral("folder") : fi.suffix().toUpper(), {}, false, false, p);
        ++rows;
    }
    shots->setVisible(rows > 0);
}

QImage MainWindow::thumb_of_first() const {
    return first_thumb_;
}

void MainWindow::update_badges() {
    auto* text = findChild<QLabel*>("hdrBadgeText");
    auto* sub = findChild<QLabel*>("hdrBadgeSub");
    auto* dot = findChild<QLabel*>("hdrDot");
    bool hdr = false;
    QString what = "display not measured";
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        const ViewerStatus s = viewer_->status();
        hdr = s.target.path != OutputPath::SdrPqSimulation;
        what = hdr ? QLocale(QLocale::English).toString(qlonglong(std::lround(s.target.peak_nits))) + " nits headroom"
                   : QStringLiteral("simulated at 203 nits");
    }
#endif
    text->setText(hdr ? "HDR" : "SDR");
    sub->setText(what);
    if (dot->property("state").toString() != (hdr ? "hdr" : "sdr")) {
        dot->setProperty("state", hdr ? "hdr" : "sdr");
        dot->style()->unpolish(dot);
        dot->style()->polish(dot);
    }
    auto* cll = findChild<QLabel*>("cllBadge");
    if (measure_) {
        const double maxcll = measure_->measured.metrics.maxcll;
        const bool clipped = maxcll > session_.display_nits() + 0.5;
        cll->setText(QStringLiteral("MaxCLL ") + QLocale(QLocale::English).toString(qlonglong(std::lround(maxcll))) +
                     (clipped ? QStringLiteral(" · clipped on screen") : QString()));
    } else {
        cll->setText(QStringLiteral("MaxCLL —"));
    }
}

bool MainWindow::eventFilter(QObject* o, QEvent* e) {
    if (o == exposure_ && e->type() == QEvent::MouseButtonDblClick) {   // the exposure back to 0
        session_.reset_exposure();
        return true;
    }
    if (e->type() == QEvent::MouseButtonRelease) {
        const QString p = o->property("open").toString();
        if (!p.isEmpty()) {
            QTimer::singleShot(0, this, [this, p] {
                if (QFileInfo(p).isDir()) open_source(p, true);
                else add_files({std::filesystem::path(p.toStdString())});
            });
            return true;
        }
    }
    return QMainWindow::eventFilter(o, e);
}

void MainWindow::log(const QString& line) {
    if (log_) log_->appendPlainText(line);
    app_log_line(line);   // and the log file, when the app installed one (startup.hpp)
}

}  // namespace rudra::app

// ---------------------------------------------------------------------------
// Phase 4 step 11: movies and their exports
// ---------------------------------------------------------------------------

namespace rudra::app {

QString MainWindow::shot_title() const {
    if (frames_.empty()) return {};
    if (shot_video_) return QString::fromStdString(shot_video_->path.filename().string());
    const auto& first = frames_.front();
    return QString::fromStdString(frames_.size() > 1 ? first.parent_path().filename().string() : first.filename().string());
}

std::string MainWindow::frame_name(std::size_t i) const {
    if (shot_video_ && i < std::size_t(shot_video_->count())) return shot_video_->name_of(int(i));
    return i < frames_.size() ? frames_[i].filename().string() : std::string();
}

VideoQueueOptions MainWindow::queue_options() const {
    VideoQueueOptions o;
    if (backend_choice_) {
        o.runtime = to_string(backend_choice_->runtime);
        o.device = backend_choice_->device;
    }
    return o;
}

bool MainWindow::queue_video_export(const QString& format) {
    auto say = [this](const QString& m) {
        log(m);
        if (auto* status = findChild<QLabel*>("renderStatus")) {
            status->setText(m);
            status->setVisible(!m.isEmpty());
        }
    };
    if (session_.grade.headroom_policy != HeadroomPolicy::Disabled) return start_headroom_export(format);
    if (!shot_video_) {
        say("Open a movie to export it to HDR10, HLG or ProRes.");
        return false;
    }
    if (needs_model() || (use_model_ && !manifest_)) {
        say("Open a model package first, or turn off Reconstruct > Use the model.");
        return false;
    }
    const QString dir = findChild<QLineEdit*>("renderDir")->text().trimmed();
    if (dir.isEmpty()) {
        say("Choose a render folder first.");
        return false;
    }
    QString name = findChild<QLineEdit*>("renderName")->text().trimmed();
    if (name.isEmpty()) name = QString::fromStdString(shot_video_->path.stem().string()) + "_" + format;
    // A log master never takes the graded file's name: the curve goes on the end.
    else if (format.endsWith("_acescct")) name += "_acescct";
    else if (format.endsWith("_logc4")) name += "_logc4";
    const bool prores = format.startsWith("prores");
    const std::filesystem::path output =
        std::filesystem::path(dir.toStdString()) / (name.toStdString() + (format == "ffv1rgb16" ? ".mkv" : prores ? ".mov" : ".mp4"));
    std::error_code ec;
    std::filesystem::path sidecar = output;
    sidecar += ".json";
    if (std::filesystem::exists(output, ec) || std::filesystem::exists(sidecar, ec)) {
        say(QString::fromStdString(output.filename().string()) + " is already there; nothing is replaced. Change the name.");
        return false;
    }
    // What this ffmpeg can do, before anything is written.
    auto caps = probe_ffmpeg();
    if (!caps) {
        say(QString::fromStdString(caps.error().message));
        return false;
    }
    const bool can = caps->zscale && (format == "ffv1rgb16" ? caps->ffv1 : prores ? caps->prores_ks : caps->libx265);
    if (!can) {
        say(QString::fromStdString(std::string("This FFmpeg build needs ") + (format == "ffv1rgb16" ? "ffv1" : prores ? "prores_ks" : "libx265") +
                                   " and zscale support (" + caps->version + ")"));
        return false;
    }
    // The export carries what the view and Master EXR use: the reconstruction
    // (analytic or the model), recovery, source curve, calibration or
    // reference, painted masks, anchor and Region EV, in a settings file
    // beside the queue that the export reads every frame through.
    // A name of its own: an earlier queue for this output (one that failed,
    // say) keeps the settings it was made with.
    std::filesystem::path grade = output, masks = output;
    grade += ".settings.json";
    masks += ".masks.png";
    for (int k = 2; std::filesystem::exists(grade, ec) || std::filesystem::exists(masks, ec); ++k) {
        grade = output;
        grade += ".settings " + std::to_string(k) + ".json";
        masks = output;
        masks += ".masks " + std::to_string(k) + ".png";
    }
    MasterRequest q = master_request();
    if (q.masks) {
        if (auto w = write_mask_set(masks, *q.masks); !w) {
            say(QString::fromStdString(w.error().message));
            return false;
        }
        q.masks_file = masks.string();
    }
    {
        std::ofstream gf(grade, std::ios::binary);
        gf << master_request_json(q) << "\n";
        if (!gf) {
            say("Could not write the export's settings: " + from_path(grade));
            return false;
        }
    }
    auto queue = write_export_queue(shot_video_->path, output,
                                    use_model_ ? manifest_->root / "manifest.json" : std::filesystem::path(),
                                    format.toStdString(), grade, steady_ ? kShotHoldSeconds : 0.0);
    if (!queue) {
        say(QString::fromStdString(queue.error().message));
        return false;
    }
    QString label = format == "hdr10" ? "HDR10"
                    : format == "hlg" ? "HLG"
                    : format == "ffv1rgb16" ? "FFV1 RGB16"
                    : format.startsWith("prores4444") ? "ProRes 4444"
                                                      : "ProRes 422 HQ";
    if (format.endsWith("_acescct")) label += " ACEScct";
    if (format.endsWith("_logc4")) label += " LogC4";
    const std::string title = shot_title().toStdString() + " \xe2\x86\x92 " + label.toStdString() + ", " + output.filename().string();
    queues_->run({*queue, title}, queue_options());
    save_queues();
    say("Queued " + label + ": " + QString::fromStdString(output.string()));
    return true;
}

QString MainWindow::frames_movie_base(const QString& format) const {
    QString name = findChild<QLineEdit*>("renderName")->text().trimmed();
    if (name.isEmpty()) name = shot_title();
    name += "_" + format;
    for (QChar& c : name)
        if (!(c.isLetterOrNumber() && c.unicode() < 128) && c != '_' && c != '.' && c != '-') c = '_';
    return name;
}

bool MainWindow::export_frames_to_movie(const QString& format, double fps) {
    if (refuse_headroom_export(format.toUpper() + QStringLiteral(" export"))) return false;
    auto* status = findChild<QLabel*>("renderStatus");
    auto say = [this, status](const QString& t) {
        status->setText(t);
        status->setVisible(!t.isEmpty());
        log(t);
    };
    const SequenceTarget* target = find_sequence_target(format.toStdString());
    if (!target) {
        say("Unknown movie format: " + format);
        return false;
    }
    if (!shot_is_frame_folder()) {
        say("Open a folder of frames first: a still goes out as EXR.");
        return false;
    }
    if (mastering() || encoding_movie_) {
        say("A render is running: wait for it, or stop it with the Master EXR button.");
        return false;
    }
    if (needs_model()) {
        say("Open a model package first, or turn off Reconstruct > Use the model.");
        return false;
    }
    if (!(fps > 0.0 && fps <= 240.0)) {
        say("The frame rate must be between 0 and 240 fps.");
        return false;
    }
    const QString dir = findChild<QLineEdit*>("renderDir")->text().trimmed();
    if (dir.isEmpty()) {
        say("Choose a render folder first.");
        return false;
    }
    const QString name = frames_movie_base(format);
    const std::filesystem::path out_dir = to_path(dir);
    std::filesystem::path output = out_dir / (name.toStdString() + target->suffix);
    const std::filesystem::path frames_dir = out_dir / (name.toStdString() + "_frames");
    std::error_code ec;
    if (std::filesystem::exists(output, ec)) {
        say(QString::fromStdString(output.filename().string()) + " is already there; nothing is replaced. Change the name.");
        return false;
    }
    if (std::filesystem::exists(frames_dir, ec) && !std::filesystem::is_empty(frames_dir, ec)) {
        say(QString::fromStdString(frames_dir.filename().string()) +
            " is already there; nothing is replaced. Change the name, or move that folder.");
        return false;
    }
    // What this ffmpeg can do, before anything is written.
    auto caps = probe_ffmpeg();
    if (!caps) {
        say(QString::fromStdString(caps.error().message));
        return false;
    }
    const bool prores = target->suffix == ".mov";
    if (!(format == "ffv1rgb16" ? caps->ffv1 : prores ? caps->prores_ks : caps->libx265)) {
        say(QString::fromStdString(std::string("This FFmpeg build has no ") + (format == "ffv1rgb16" ? "ffv1" : prores ? "prores_ks" : "libx265") + " (" +
                                   caps->version + ")"));
        return false;
    }
    std::filesystem::create_directories(frames_dir, ec);
    if (ec) {
        say("Could not make " + from_path(frames_dir) + ": " + QString::fromStdString(ec.message()));
        return false;
    }
    frames_movie_ = FramesMovie{format, frames_dir, output, name.toStdString(), fps};
    master();   // the frames first; master_finished hands them to the encoder
    if (!mastering()) {
        frames_movie_.reset();
        return false;
    }
    log("Movie export: mastering " + QString::number(frames_.size()) + " frames to " + from_path(frames_dir) +
        ", then " + QString::fromStdString(target->note));
    return true;
}

void MainWindow::encode_frames_movie(FramesMovie job) {
    auto* status = findChild<QLabel*>("renderStatus");
    auto paths = list_linear_frames(job.frames_dir);
    if (!paths) {
        status->setText("Movie export stopped: " + QString::fromStdString(paths.error().message));
        status->show();
        return;
    }
    if (encode_worker_.joinable()) encode_worker_.join();
    SequenceEncodeOptions o;
    o.target = job.format.toStdString();
    o.fps = job.fps;
    o.peak_nits = 1000.0;   // the video export's peak (video/convert.hpp)
    o.source = Primaries::Rec2020;
    encode_cancel_ = false;
    encoding_movie_ = true;
    btn_master_->setText("Encoding…");
    btn_master_->setToolTip("Click to stop the movie; the mastered frames stay");
    status->setText("Measuring " + QString::number(paths->size()) + " frames for the movie…");
    status->show();
    QPointer<MainWindow> self(this);
    encode_worker_ = std::thread([self, job, o, paths = std::move(*paths), cancel = &encode_cancel_]() mutable {
        auto progress = [self](const LinearDeliveryProgress& p) {
            const QString t = QStringLiteral("%1 %2 / %3").arg(p.encoding ? "Encoding" : "Measuring").arg(p.done).arg(p.total);
            QMetaObject::invokeMethod(qApp, [self, t] {
                if (self) self->findChild<QLabel*>("renderStatus")->setText(t);
            });
        };
        o.note = [self](const std::string& n) {
            const QString t = QString::fromStdString(n);
            QMetaObject::invokeMethod(qApp, [self, t] {
                if (self) self->log(t);
            });
        };
        auto r = deliver_linear_frames(paths, job.output, o, 203.0 /* the linear master: 1.0 = 203 nits */, progress, cancel);
        QString line;
        if (r)
            line = QStringLiteral("Movie written: %1 · %2 frames · MaxCLL %3 · MaxFALL %4")
                       .arg(QString::fromStdString(r->file.string()))
                       .arg(r->frames)
                       .arg(r->metadata.maxcll)
                       .arg(r->metadata.maxfall);
        else
            line = "Movie export stopped: " + QString::fromStdString(r.error().message) + ". The mastered frames are in " +
                   QString::fromStdString(job.frames_dir.string());
        QMetaObject::invokeMethod(qApp, [self, line] {
            if (!self) return;
            self->encoding_movie_ = false;
            self->btn_master_->setText("Master EXR");
            self->btn_master_->setToolTip({});
            auto* st = self->findChild<QLabel*>("renderStatus");
            st->setText(line);
            st->show();
            self->log(line);
            self->refresh_enabled();
            self->sync_ui();
        });
    });
    refresh_enabled();
    sync_ui();
}

void MainWindow::resume_queue(const std::filesystem::path& queue) {
    for (const auto& e : queues_->entries())
        if (e.queue == queue) {
            queues_->run(e, queue_options(), true);   // a failed job is tried again, an interrupted one restarts
            return;
        }
}

void MainWindow::forget_queue(const std::filesystem::path& queue) {
    queues_->forget(queue);
    save_queues();
}

}  // namespace rudra::app

// ---- Calibrate (roadmap 3.2) ---------------------------------------------

namespace rudra::app {

void MainWindow::arm_calibration(int slot) {
    const bool on = slot >= 0 && current_frame_;
    if (on) arm_mask(-1);   // one pointer mode at a time (3.3)
    source_panel_->set_armed(on ? slot : -1);
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->set_pick_mode(on);
#endif
}

std::optional<int> MainWindow::sdr_code_at(double x, double y) const {
    if (!current_frame_) return std::nullopt;
    const SdrImage& sdr = current_frame_->sdr;
    const int w = sdr.width(), h = sdr.height();
    const int cx = std::clamp(int(std::floor(x)), 0, w - 1), cy = std::clamp(int(std::floor(y)), 0, h - 1);
    // A 5x5 mean of the three channels: a neutral patch, not a single pixel's noise.
    double sum = 0.0;
    int n = 0;
    const auto& b = sdr.buffer();
    for (int yy = std::max(0, cy - 2); yy <= std::min(h - 1, cy + 2); ++yy)
        for (int xx = std::max(0, cx - 2); xx <= std::min(w - 1, cx + 2); ++xx) {
            const std::size_t i = std::size_t(yy) * std::size_t(w) + std::size_t(xx);
            for (int c = 0; c < 3; ++c) sum += b.plane(c)[i];
            n += 3;
        }
    return std::clamp(int(std::lround(sum / std::max(n, 1) * 255.0)), 0, 255);
}

bool MainWindow::load_reference(const std::filesystem::path& path) {
    if (!current_frame_) return false;
    std::filesystem::path file = path;
    if (file.empty()) {
        const QString chosen = QFileDialog::getOpenFileName(this, "Open the graded HDR of this frame", {},
                                                            "HDR frames (*.exr *.png *.tif *.tiff)");
        if (chosen.isEmpty()) return false;
        file = chosen.toStdString();
    }
    auto img = read_reference(file);
    if (!img) {
        QMessageBox::warning(this, "RUDRA", QString::fromStdString(img.error().message + "\n" + img.error().detail));
        return false;
    }
    // The frame on screen is the preview (fit_max_side); a full-size reference
    // is brought to it the same way, INTER_AREA in float, so the codes line up.
    const SdrImage& sdr = current_frame_->sdr;
    PlanarBuffer linear = std::move(img->linear709);
#ifdef RUDRA_HAVE_STILL_DECODE
    if ((linear.width() != sdr.width() || linear.height() != sdr.height()) && preview_max_side_ > 0 &&
        std::max(linear.width(), linear.height()) > preview_max_side_)
        linear = fit_max_side(SdrImage(std::move(linear)), preview_max_side_).buffer();
#endif
    auto fit = fit_reference(sdr, linear, file.filename().string());
    if (!fit) {
        QMessageBox::warning(this, "RUDRA", QString::fromStdString(fit.error().message + "\n" + fit.error().detail));
        return false;
    }
    session_.set_reference(std::move(*fit));
    return true;
}

// ---- painted masks (roadmap 3.3) -------------------------------------------

void MainWindow::arm_mask(int band) {
    const bool on = band >= 0 && current_frame_;
    session_.arm_mask(on ? band : -1);
    if (on) arm_calibration(-1);   // one pointer mode at a time
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->set_paint_mode(on);
#endif
}

void MainWindow::paint_stroke(double x0, double y0, double x1, double y1, bool erase) {
    if (!current_frame_ || session_.painted_band() < 0) return;
    const SdrImage& sdr = current_frame_->sdr;
    session_.stroke(sdr.width(), sdr.height(), x0, y0, x1, y1, erase);
}

// ---- the clip fill (12 Oct 2026) -------------------------------------------

ClipFillOptions MainWindow::session_clip_options() const {
    ClipFillOptions o;
    o.enabled = session_.grade.clip_fill;
    o.overrides = session_.grade.clip_peaks;
    return o;
}

void MainWindow::arm_clip_peak(bool on) {
    on = on && current_frame_ && !use_model_;
    if (on) {
        arm_mask(-1);          // one pointer mode at a time
        arm_calibration(-1);
        log("click a clipped region (a lamp, the sun, a blown sky) to set its peak");
    }
    clip_pick_armed_ = on;
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->set_pick_mode(on);
#endif
}

void MainWindow::clip_peak_picked(double x, double y) {
    arm_clip_peak(false);
    if (!current_frame_) return;
    const SdrImage& sdr = current_frame_->sdr;
    const int w = sdr.width(), h = sdr.height();
    const int cx = std::clamp(int(std::floor(x)), 0, w - 1), cy = std::clamp(int(std::floor(y)), 0, h - 1);
    const std::size_t i = std::size_t(cy) * std::size_t(w) + std::size_t(cx);
    const auto& b = sdr.buffer();
    const float top = std::max({b.plane(0)[i], b.plane(1)[i], b.plane(2)[i]});
    if (top < 0.97f) {
        log("that point is not clipped: pick inside a region at white");
        return;
    }
    bool ok = false;
    const double nits = QInputDialog::getDouble(this, "Clipped peak",
                                                "Peak of this clipped region, in nits (0 removes a peak set here):",
                                                2000.0, 0.0, 10000.0, 0, &ok);
    if (!ok) return;
    session_.set_clip_peak((cx + 0.5) / w, (cy + 0.5) / h, nits);
}

void MainWindow::calibration_picked(double x, double y) {
    const int slot = source_panel_->armed();
    source_panel_->set_armed(-1);
    if (slot < 0) return;
    if (const auto code = sdr_code_at(x, y)) source_panel_->set_picked(slot, *code);
}

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
    QPointer<MainWindow> self(this);
    QTimer::singleShot(600, this, [self] {
        if (self) self->show_getting_started();
    });
}

// ---- The reconstruction: analytic, or with the model (10 Oct 2026 review) ----

ModelConstants MainWindow::frame_constants() const {
    if (use_model_ && manifest_) return ModelConstants{manifest_->log_scale, manifest_->max_hdr, manifest_->corpus_ev};
    return kAnalyticConstants;
}

void MainWindow::show_mode() {
    if (!mode_label_) return;
    if (!use_model_) {
        mode_label_->setText("Analytic reconstruction");
        mode_label_->setToolTip("The inverse of the source curve, with your calibration, reference and grade. No model "
                                "runs, for the view or for a master. Reconstruct > Use the model turns it on.");
    } else {
        mode_label_->setText("AI-assisted: " + (manifest_ ? QString::fromStdString(manifest_->name)
                                                          : (loading_model_ ? QStringLiteral("loading…")
                                                                            : QStringLiteral("no model package"))));
        mode_label_->setToolTip("The model's reconstruction on top of the analytic inverse; its name is recorded in "
                                "every master's sidecar.");
    }
}

void MainWindow::set_use_model(bool on, bool remember) {
    if (on == use_model_) {
        show_mode();
        return;
    }
    use_model_ = on;
    if (remember) QSettings().setValue("model/use", on);
    log(on ? QStringLiteral("reconstruction: the model's (AI-assisted)")
           : QStringLiteral("reconstruction: analytic, no model runs"));
    // On with nothing loaded: the package used last, or the catalog's pick.
    if (on && !backend_ && !loading_model_) {
        rescan_models();
        const std::filesystem::path last = QSettings().value("model/package").toString().toStdString();
        std::error_code ec;
        if (!last.empty() && std::filesystem::is_regular_file(last / "manifest.json", ec)) use_model(last, backend_choice_);
        else if (const auto i = catalog_.pick()) use_model(catalog_.entries[*i].package, backend_choice_);
        else log("no model package found; File > Model packages… adds one");
    }
    // The frames on screen are composed again, with or without the network.
    if (!frames_.empty() && !needs_model()) start_engine(frames_, current_);
    // Off: a shot that was waiting for a model opens now.
    if (!needs_model()) {
        if (pending_source_) {
            const auto [path, folder] = *pending_source_;
            pending_source_.reset();
            QTimer::singleShot(0, this, [this, path = path, folder = folder] { open_source(path, folder); });
        }
        if (pending_shot_) {
            auto shot = std::move(*pending_shot_);
            pending_shot_.reset();
            QTimer::singleShot(0, this, [this, shot] {
                const bool was = restoring_;
                restoring_ = true;
                open_shot(shot.kind, shot.sources, shot.frame);
                restoring_ = was;
            });
        }
    }
    if (headroom_ || headroom_routed()) route_headroom();   // the preview requires the model off
    mark_dirty();
    show_mode();
    sync_checks();
    sync_ui();
    refresh_enabled();
}

// ---- Headroom preview (E1) -------------------------------------------------

bool MainWindow::headroom_routed() const {
    return headroom_capable_ && session_.grade.headroom_policy != HeadroomPolicy::Disabled && shot_video_ &&
           !frames_.empty();
}

Result<void> MainWindow::set_headroom_preview(bool on) {
    if (on && !headroom_capable_)
        return make_error(ErrorCode::Unsupported, "This window does not support the headroom preview");
    // A grade change (one undo step); session_changed routes the frames.
    return session_.set_headroom_policy(on ? HeadroomPolicy::Bt1886V1 : HeadroomPolicy::Disabled);
}

void MainWindow::toggle_super_whites() {
    if (session_.grade.headroom_policy != HeadroomPolicy::Disabled) {
        (void)set_headroom_preview(false);   // never refused
        log(QStringLiteral("super-whites: off, the movie is read as usual"));
        return;
    }
    // What bt1886-headroom-v1 accepts (engine/headroom_presentation.cpp): said
    // before anything changes, and nothing the artist set is cleared for them.
    QString why;
    if (frames_.empty() || !shot_video_)
        why = QStringLiteral("Open a movie first. Super-whites are kept for limited-range BT.709 movies, "
                             "not for stills or folders of frames.");
    else if (session_.grade.source != "bt1886")
        why = QStringLiteral("Super-whites are kept with Reconstruct > Source > BT.1886 only. Choose it, then try again.");
    else if (!session_.grade.clip_peaks.empty() || !session_.grade.calibration.empty() ||
             !session_.grade.reference.empty())
        why = QStringLiteral("Clear the clipped peaks, the calibration and the reference match first: "
                             "they are not available while super-whites are kept.");
    if (!why.isEmpty()) {
        problem_ = why;
        log(QStringLiteral("super-whites: ") + why);
        if (isVisible()) QMessageBox::information(this, QStringLiteral("Keep super-whites"), why);
        return;
    }
    QStringList off;
    if (use_model_) off << QStringLiteral("the model");
    if (steady_) off << QStringLiteral("Hold steady within a shot");
    if (session_.grade.clip_fill) off << QStringLiteral("Fill clipped highlights");
    if (session_.anchor) off << QStringLiteral("Anchor to source exposure");
    if (session_.carry_chroma) off << QStringLiteral("Carry source chroma");
    if (isVisible()) {
        QString text = QStringLiteral(
            "Keep super-whites is experimental. RUDRA reads the movie with the detail recorded above white and "
            "carries it into the HDR, instead of clipping it and rebuilding the highlights.\n\n"
            "It works on limited-range BT.709 movies; others are refused with the reason.");
        if (!off.isEmpty()) text += QStringLiteral("\n\nIt also turns off: ") + off.join(QStringLiteral(", ")) + '.';
        if (QMessageBox::question(this, QStringLiteral("Keep super-whites"), text + QStringLiteral("\n\nTurn it on?")) !=
            QMessageBox::Yes)
            return;
    }
    if (!headroom_capable_) {
        // Remembered, so a project saved with super-whites opens at the next start.
        headroom_capable_ = true;
        session_.allow_headroom();
        QSettings().setValue("experimental/superWhites", true);
    }
    if (use_model_) set_use_model(false, false);   // the default for the next start is unchanged
    if (steady_) set_steady_shots(false);
    if (session_.anchor) session_.toggle_anchor();
    if (session_.carry_chroma) session_.toggle_carry_chroma();
    if (session_.grade.clip_fill) session_.toggle_clip_fill();
    if (auto on = set_headroom_preview(true); !on) {
        problem_ = qs(on.error().message);
        log(QStringLiteral("super-whites: ") + problem_);
        if (isVisible()) QMessageBox::warning(this, QStringLiteral("Keep super-whites"), problem_);
        return;
    }
    problem_.clear();
    log(QStringLiteral("super-whites: kept (bt1886-headroom-v1, experimental)") +
        (off.isEmpty() ? QString() : QStringLiteral("; turned off: ") + off.join(QStringLiteral(", "))));
}

HeadroomPreviewStats MainWindow::headroom_stats() const { return headroom_ ? headroom_->stats() : HeadroomPreviewStats{}; }

void MainWindow::clear_legacy_frame() {
    // What the measurements, scopes and probe read: legacy SDR state only.
    current_frame_.reset();
    frame_ready_ok_ = false;
    measure_.reset();
    ++stats_gen_;
    clear_scopes();
    if (clip_bar_) clip_bar_->hide();
    probe_pixel(std::nullopt);
}

void MainWindow::stop_headroom() {
    if (!headroom_ && !headroom_shown_ && headroom_problem_.isEmpty()) return;
    const bool shown = headroom_shown_.has_value();
    headroom_.reset();   // joins; frames already queued for this window are dropped
    headroom_shown_.reset();
    headroom_problem_.clear();
    if (headroom_playing_) {
        headroom_playing_ = false;
        if (btn_play_) btn_play_->set_glyph(IconButton::Glyph::Play);
    }
#ifdef RUDRA_APP_VIEWER
    if (viewer_ && shown) viewer_->clear_frame();
#else
    (void)shown;
#endif
}

void MainWindow::route_headroom() {
    if (!headroom_routed()) {
        if (!headroom_ && !headroom_shown_ && headroom_problem_.isEmpty()) return;
        stop_headroom();
        // Turned off (the grade, undo, another shot): the legacy engine shows the frame again.
        if (engine_ && !frames_.empty()) step_to(current_);
        return;
    }
    if (!headroom_) {
        auto port = headroom_factory_ ? headroom_factory_() : make_headroom_presentation_port();
        if (!port) {
            headroom_failed(port.error());
            return;
        }
        QPointer<MainWindow> self(this);
        HeadroomPreviewDriver::Callbacks cb;
        cb.post = [self](std::function<void()> task) {
            QMetaObject::invokeMethod(qApp, [self, task = std::move(task)] {
                if (self) task();
            });
        };
        cb.shown = [this](const HeadroomShown& f) { headroom_present(f); };
        cb.failed = [this](const Error& e) { headroom_failed(e); };
        cb.finished = [this] { headroom_finished(); };
        auto driver = HeadroomPreviewDriver::create(std::move(*port), std::move(cb));
        if (!driver) {
            headroom_failed(driver.error());
            return;
        }
        headroom_ = std::move(*driver);
        // The legacy engine keeps its frames but nothing of it is shown now.
        halt_playback();
        clear_legacy_frame();
#ifdef RUDRA_APP_VIEWER
        if (viewer_) viewer_->clear_frame();
#endif
    }
    if (auto c = headroom_->configure(session_, false, use_model_, steady_); !c) {
        headroom_failed(c.error());
        return;
    }
    headroom_request(current_, headroom_playing_);
}

void MainWindow::headroom_request(int index, bool play) {
    if (frames_.empty()) return;
    const int n = int(frames_.size());
    current_ = ((index % n) + n) % n;
    if (!headroom_) {
        route_headroom();   // creates and configures, then requests current_
        return;
    }
    if (!headroom_->configured()) {
        // Still refused: the reason stays on screen; nothing older is shown.
        sync_ui();
        return;
    }
    if (auto r = headroom_->show(shot_video_->path, current_, n, play); !r) {
        headroom_failed(r.error());
        return;
    }
    headroom_problem_.clear();
    waiting_ = true;
    sync_ui();
}

void MainWindow::headroom_present(const HeadroomShown& f) {
    // Inside the bridge's final presentation (its lock is held): no driver calls.
    headroom_shown_ = f;
    current_ = f.index;
    waiting_ = false;
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->set_composited(*f.model, *f.baseline);
#endif
    // The rest (labels, strip) after this returns, still before any other request.
    QMetaObject::invokeMethod(this, [this] { headroom_after_present(); }, Qt::QueuedConnection);
}

void MainWindow::headroom_after_present() {
    if (!headroom_shown_) return;
    const auto& f = *headroom_shown_;
    frame_info_ = QStringLiteral("%1  %2/%3  \u00b7  headroom preview (%4), %5x%6")
                      .arg(QString::fromStdString(frame_name(std::size_t(f.index))))
                      .arg(f.index + 1)
                      .arg(frames_.size())
                      .arg(qs(headroom_policy_id(session_.grade.headroom_policy)))
                      .arg(f.width)
                      .arg(f.height);
    // E4: measured super-white detail is told apart from clipping at the code ceiling.
    frame_info_ += QStringLiteral("  \u00b7  above white: %1% detail, %2% at the code ceiling (clipped), %3% at white")
                       .arg(f.source_whites.detail_pct, 0, 'f', 2)
                       .arg(f.source_whites.ceiling_pct, 0, 'f', 2)
                       .arg(f.source_whites.at_white_pct, 0, 'f', 2);
    note_frame_in_strip(f.index);
    run_stats();   // E2: its readouts and scopes follow on their own thread
    show_status();
    sync_ui();
}

void MainWindow::headroom_failed(const Error& e) {
    headroom_problem_ = qs(e.message);
    log(QStringLiteral("headroom preview: ") + headroom_problem_ +
        (e.detail.empty() ? QString() : QStringLiteral(" (") + qs(e.detail) + QStringLiteral(")")));
    // Nothing older and no legacy picture stands in for the refused preview.
    headroom_shown_.reset();
    clear_legacy_frame();
    waiting_ = false;
    if (headroom_playing_) {
        headroom_playing_ = false;
        if (btn_play_) btn_play_->set_glyph(IconButton::Glyph::Play);
    }
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->clear_frame();
#endif
    frame_info_ = QStringLiteral("Headroom preview stopped: ") + headroom_problem_;
    statusBar()->showMessage(frame_info_);
    sync_ui();
}

void MainWindow::headroom_finished() {
    if (!headroom_playing_) return;   // a paused frame's range ends after its one frame
    headroom_playing_ = false;
    if (btn_play_) btn_play_->set_glyph(IconButton::Glyph::Play);
    log(QStringLiteral("the headroom preview reached the end of the shot"));
    sync_ui();
}

bool MainWindow::refuse_headroom_export(const QString& what) {
    if (session_.grade.headroom_policy == HeadroomPolicy::Disabled) return false;
    const QString m = what + QStringLiteral(" does not support the headroom policy yet; turn the headroom preview off to export.");
    log(m);
    if (auto* status = findChild<QLabel*>("renderStatus")) {
        status->setText(m);
        status->setVisible(true);
    }
    return true;
}

bool MainWindow::start_headroom_export(const QString& format) {
    auto say = [this](const QString& m) {
        log(m);
        if (auto* status = findChild<QLabel*>("renderStatus")) {
            status->setText(m);
            status->setVisible(!m.isEmpty());
        }
    };
    if (headroom_export_running_) {   // a second click cancels, as Master EXR's does
        headroom_export_cancel_ = true;
        say(QStringLiteral("Cancelling the headroom export\u2026"));
        return false;
    }
    static const QStringList kFormats = {"hdr10", "hlg", "prores422hq", "prores4444", "ffv1rgb16"};
    if (!kFormats.contains(format)) {
        say(format.toUpper() + QStringLiteral(" does not support the headroom policy yet; HDR10, HLG, ProRes and FFV1 RGB16 do."));
        return false;
    }
    if (!headroom_routed() || !shot_video_) {
        say(QStringLiteral("Open a movie with the headroom preview to export it."));
        return false;
    }
    if (use_model_) {
        say(QStringLiteral("The headroom export is analytic; turn off Reconstruct > Use the model."));
        return false;
    }
    const QString dir = findChild<QLineEdit*>("renderDir")->text().trimmed();
    if (dir.isEmpty()) {
        say("Choose a render folder first.");
        return false;
    }
    QString name = findChild<QLineEdit*>("renderName")->text().trimmed();
    if (name.isEmpty()) name = QString::fromStdString(shot_video_->path.stem().string()) + "_" + format + "_headroom";
    HeadroomExportRequest r;
    r.source = shot_video_->path;
    r.output = std::filesystem::path(dir.toStdString()) / name.toStdString();
    r.target = format.toStdString();
    r.fps = play_fps();
    r.grade = master_request();
    r.grade.settle_grain = false;   // the window has no grain control; v1 never settles grain (see current_project)
    r.grade.source_file = shot_video_->path;
    r.app_version = QCoreApplication::applicationVersion().toStdString();
    if (auto valid = validate_headroom_grade(r.grade); !valid) {
        say(qs(valid.error().message));
        return false;
    }
    if (headroom_export_worker_.joinable()) headroom_export_worker_.join();   // the last one finished
    headroom_export_cancel_ = false;
    headroom_export_running_ = true;
    headroom_export_result_.clear();
    say(QStringLiteral("Exporting %1 with the headroom policy (decode, render, encode, check, then publish)").arg(format.toUpper()));
    QPointer<MainWindow> self(this);
    headroom_export_worker_ = std::thread([self, r, cancel = &headroom_export_cancel_] {
        auto out = export_headroom_movie(r, [self](const HeadroomExportProgress& p) {
            QMetaObject::invokeMethod(qApp, [self, p] {
                if (!self) return;
                if (auto* status = self->findChild<QLabel*>("renderStatus")) {
                    status->setText(QStringLiteral("Headroom export: %1 %2/%3").arg(qs(p.phase)).arg(p.done).arg(p.total));
                    status->setVisible(true);
                }
            });
        }, cancel);
        QMetaObject::invokeMethod(qApp, [self, out = std::move(out)] {
            if (!self) return;
            self->headroom_export_running_ = false;
            QString m;
            if (out) {
                self->headroom_export_result_ = qs(out->file.string());
                m = QStringLiteral("Published %1 (%2 frames, MaxCLL %3, MaxFALL %4) and its provenance sidecar")
                        .arg(qs(out->file.string())).arg(out->frames).arg(out->metadata.maxcll).arg(out->metadata.maxfall);
            } else {
                self->headroom_export_result_ = qs(out.error().message);
                m = QStringLiteral("The headroom export did not publish anything: ") + qs(out.error().message) +
                    (out.error().detail.empty() ? QString() : QStringLiteral(" (") + qs(out.error().detail) + ")");
            }
            self->log(m);
            if (auto* status = self->findChild<QLabel*>("renderStatus")) {
                status->setText(m);
                status->setVisible(true);
            }
            self->refresh_enabled();
        });
    });
    refresh_enabled();
    return true;
}

}  // namespace rudra::app
