// The headroom preview (split from main_window.cpp, 28 Oct 2026 review: one file per part of the window).
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
    // Only the letters a file name can safely carry ("A/B" or "x:y" wrote elsewhere; 28 Oct 2026 review).
    QString name = batch_clean_name(findChild<QLineEdit*>("renderName")->text());
    if (name.isEmpty()) name = from_path(shot_video_->path.stem()) + "_" + format + "_headroom";
    HeadroomExportRequest r;
    r.source = shot_video_->path;
    r.output = to_path(dir) / to_path(name);
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
