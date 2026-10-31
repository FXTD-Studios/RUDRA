// Movies and their exports (split from main_window.cpp, 28 Oct 2026 review: one file per part of the window).
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


// ---------------------------------------------------------------------------
// Phase 4 step 11: movies and their exports
// ---------------------------------------------------------------------------


QString MainWindow::shot_title() const {
    if (frames_.empty()) return {};
    if (shot_video_) return from_path(shot_video_->path.filename());
    const auto& first = frames_.front();
    return from_path(frames_.size() > 1 ? first.parent_path().filename() : first.filename());
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
    // Only the letters a file name can safely carry ("A/B" or "x:y" wrote elsewhere; 28 Oct 2026 review).
    QString name = batch_clean_name(findChild<QLineEdit*>("renderName")->text());
    if (name.isEmpty()) name = from_path(shot_video_->path.stem()) + "_" + format;
    // A log master never takes the graded file's name: the curve goes on the end.
    else if (format.endsWith("_acescct")) name += "_acescct";
    else if (format.endsWith("_logc4")) name += "_logc4";
    const bool prores = format.startsWith("prores");
    const bool mov16 = format == "rgb16mov";   // uncompressed 16-bit RGB QuickTime: rawvideo, in every ffmpeg
    const std::filesystem::path output =
        to_path(dir) /
        to_path(name + (format == "ffv1rgb16" ? ".mkv" : prores || mov16 ? ".mov" : ".mp4"));
    std::error_code ec;
    std::filesystem::path sidecar = output;
    sidecar += ".json";
    if (std::filesystem::exists(output, ec) || std::filesystem::exists(sidecar, ec)) {
        say(from_path(output.filename()) + " is already there; nothing is replaced. Change the name.");
        return false;
    }
    // What this ffmpeg can do, before anything is written.
    auto caps = probe_ffmpeg();
    if (!caps) {
        say(QString::fromStdString(caps.error().message));
        return false;
    }
    const bool can = caps->zscale && (mov16 || (format == "ffv1rgb16" ? caps->ffv1 : prores ? caps->prores_ks : caps->libx265));
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
                    : format == "rgb16mov" ? "16-bit RGB MOV"
                    : format.startsWith("prores4444") ? "ProRes 4444"
                                                      : "ProRes 422 HQ";
    if (format.endsWith("_acescct")) label += " ACEScct";
    if (format.endsWith("_logc4")) label += " LogC4";
    const std::string title = shot_title().toStdString() + " \xe2\x86\x92 " + label.toStdString() + ", " + output.filename().string();
    queues_->run({*queue, title}, queue_options());
    save_queues();
    say("Queued " + label + ": " + from_path(output));
    return true;
}

QString MainWindow::frames_movie_base(const QString& format) const {
    // Only the letters a file name can safely carry ("A/B" or "x:y" wrote elsewhere; 28 Oct 2026 review).
    QString name = batch_clean_name(findChild<QLineEdit*>("renderName")->text());
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
        say(from_path(output.filename()) + " is already there; nothing is replaced. Change the name.");
        return false;
    }
    if (std::filesystem::exists(frames_dir, ec) && !std::filesystem::is_empty(frames_dir, ec)) {
        say(from_path(frames_dir.filename()) +
            " is already there; nothing is replaced. Change the name, or move that folder.");
        return false;
    }
    // What this ffmpeg can do, before anything is written.
    auto caps = probe_ffmpeg();
    if (!caps) {
        say(QString::fromStdString(caps.error().message));
        return false;
    }
    const bool prores = target->suffix == ".mov" && format != "rgb16mov";
    if (format != "rgb16mov" && !(format == "ffv1rgb16" ? caps->ffv1 : prores ? caps->prores_ks : caps->libx265)) {
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
                       .arg(from_path(r->file))
                       .arg(r->frames)
                       .arg(r->metadata.maxcll)
                       .arg(r->metadata.maxfall);
        else
            line = "Movie export stopped: " + QString::fromStdString(r.error().message) + ". The mastered frames are in " +
                   from_path(job.frames_dir);
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
