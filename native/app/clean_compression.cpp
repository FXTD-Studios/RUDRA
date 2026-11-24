// RUDRA 1.3 (24 Nov 2026): Reconstruct > Clean compression (clean_compression.hpp).
#include "clean_compression.hpp"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QGuiApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPointer>
#include <QProcess>
#include <QStandardPaths>
#include <QStatusBar>
#include <QStorageInfo>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

#include "main_window.hpp"
#include "paths.hpp"
#include "rudra/core/block_grid.hpp"
#include "rudra/core/clip_fill.hpp"
#include "source_panel.hpp"

namespace rudra::app {

QString clean_filter(int qp) {
    return qp > 0 ? QStringLiteral("spp=quality=3:qp=%1").arg(qp) : QString();
}

QStringList clean_probe_arguments(const QString& source, double seek_seconds) {
    QStringList a{"-hide_banner", "-v", "error", "-nostdin"};
    if (seek_seconds >= 0.0) a << "-ss" << QString::number(seek_seconds, 'f', 3);
    const int n = int(std::size(kCleanQps));
    QString g = QStringLiteral("[0:v]split=%1").arg(n);
    for (int i = 0; i < n; ++i) g += QStringLiteral("[s%1]").arg(i);
    QString stack;
    for (int i = 0; i < n; ++i) {
        const QString f = clean_filter(kCleanQps[i]);
        g += QStringLiteral(";[s%1]%2[c%1]").arg(i).arg(f.isEmpty() ? QStringLiteral("null") : f);
        stack += QStringLiteral("[c%1]").arg(i);
    }
    g += QStringLiteral(";%1vstack=inputs=%2,format=gray").arg(stack).arg(n);
    a << "-i" << source << "-frames:v" << "1" << "-filter_complex" << g << "-f" << "rawvideo" << "-pix_fmt" << "gray" << "-";
    return a;
}

std::vector<double> clean_stack_grids(const QByteArray& raw, int width, int height, int count) {
    std::vector<double> out;
    const qsizetype one = qsizetype(width) * height;
    if (width <= 0 || height <= 0 || count <= 0 || raw.size() < one * count) return out;
    const auto* p = reinterpret_cast<const std::uint8_t*>(raw.constData());
    for (int i = 0; i < count; ++i) out.push_back(block_grid(p + one * i, width, height, width));
    return out;
}

QStringList clean_movie_arguments(const QString& source, const QString& out, int qp) {
    // FFV1 (lossless) keeps the cleaned pictures exactly; the source's pixel format, colour tags and rate pass
    // through, the audio is copied, the metadata (timecode) kept.
    return {"-hide_banner", "-v", "error", "-nostdin", "-y", "-i", source, "-map", "0:v:0", "-map", "0:a?",
            "-vf", clean_filter(qp), "-c:v", "ffv1", "-level", "3", "-slices", "16", "-slicecrc", "1",
            "-c:a", "copy", "-map_metadata", "0", "-progress", "pipe:1", "-nostats", out};
}

QStringList clean_still_arguments(const QString& source, const QString& out, int qp) {
    return {"-hide_banner", "-v", "error", "-nostdin", "-y", "-i", source, "-frames:v", "1",
            "-vf", clean_filter(qp) + QStringLiteral(",format=rgb24"), out};
}

QString clean_cache_dir(const QString& source, int qp) {
    const QFileInfo fi(source);
    const QByteArray key = (fi.absoluteFilePath() + "|" + QString::number(fi.size()) + "|" +
                            QString::number(fi.lastModified().toMSecsSinceEpoch()) + "|" + clean_filter(qp))
                               .toUtf8();
    const QString root = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    return QDir(root).filePath(QStringLiteral("clean/") +
                               QString::fromLatin1(QCryptographicHash::hash(key, QCryptographicHash::Sha1).toHex().left(16)));
}

QString clean_sidecar_path(const QString& copy) {
    const QFileInfo fi(copy);
    return fi.isDir() ? QDir(copy).filePath(QStringLiteral("rudra_clean.json")) : copy + QStringLiteral(".rudra_clean.json");
}

std::optional<CleanSidecar> read_clean_sidecar(const QString& copy) {
    if (copy.isEmpty()) return std::nullopt;
    QFile f(clean_sidecar_path(copy));
    if (!f.open(QIODevice::ReadOnly)) return std::nullopt;
    const QJsonObject o = QJsonDocument::fromJson(f.readAll()).object();
    if (o.value("schema").toString() != QLatin1String("rudra_clean_compression_v1")) return std::nullopt;
    CleanSidecar s;
    s.original = o.value("original").toString();
    s.filter = o.value("filter").toString();
    s.qp = o.value("qp").toInt();
    s.grid_before = o.value("block_grid_before").toDouble(1.0);
    s.grid_after = o.value("block_grid_after").toDouble(1.0);
    s.original_lossy = o.value("original_lossy").toBool(true);
    s.frames = o.value("frames").toInt();
    s.made = o.value("made").toString();
    return s;
}

bool write_clean_sidecar(const QString& copy, const CleanSidecar& s) {
    QFile f(clean_sidecar_path(copy));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
    const QJsonObject o{{"schema", "rudra_clean_compression_v1"}, {"original", s.original}, {"filter", s.filter},
                        {"qp", s.qp}, {"block_grid_before", s.grid_before}, {"block_grid_after", s.grid_after},
                        {"original_lossy", s.original_lossy}, {"frames", s.frames}, {"made", s.made},
                        {"why", "Reconstruct > Clean compression: the codec's block grid cleaned with ffmpeg's spp at the "
                                "smallest strength that took it to a clean frame's (RUDRA 1.3)"}};
    return f.write(QJsonDocument(o).toJson()) > 0;
}

// ---- the action -------------------------------------------------------------------------------------------

namespace {
// The shot's own file or folder: a movie, a single still, or the folder of a sequence of stills.
QString shot_source(const std::optional<FrameSequence>& video, const std::vector<std::filesystem::path>& frames) {
    if (video) return from_path(video->path);
    if (frames.empty()) return {};
    if (frames.size() == 1) return from_path(frames.front());
    return from_path(frames.front().parent_path());
}
}  // namespace

std::optional<CleanSidecar> MainWindow::shot_clean_sidecar() const {
    return read_clean_sidecar(shot_source(shot_video_, frames_));
}

namespace {
QString same_path(const QString& p) { return p.isEmpty() ? p : QFileInfo(p).absoluteFilePath(); }
}  // namespace

void MainWindow::compression_status(const QString& text, const QString& state, const QString& button) {
    compression_status_shot_ = frames_.empty() ? QString() : same_path(shot_source(shot_video_, frames_));
    if (source_panel_) source_panel_->set_compression_status(text, state, button);
}

void MainWindow::sync_compression_status() {
    if (!source_panel_) return;
    source_panel_->set_compression_busy(cleaning_);
    if (cleaning_) return;   // the running clean writes its own line
    const QString shot = frames_.empty() ? QString() : same_path(shot_source(shot_video_, frames_));
    if (shot == compression_status_shot_) return;   // still the shot the line speaks of
    compression_status_shot_ = shot;
    if (const auto side = shot.isEmpty() ? std::nullopt : read_clean_sidecar(shot)) {   // a cleaned copy says so
        source_panel_->set_compression_status(
            QStringLiteral("Cleaned: block grid %1 \u2192 %2 (a clean picture is 1.00). The shot is now a lossless cleaned "
                           "copy; the original file is untouched")
                .arg(side->grid_before, 0, 'f', 2).arg(side->grid_after, 0, 'f', 2),
            QStringLiteral("ok"), QStringLiteral("Use the original"));
        return;
    }
    source_panel_->set_compression_status({}, {}, QStringLiteral("Check compression"));   // a new shot: not checked
}

void MainWindow::clean_compression() {
    if (frames_.empty()) {
        statusBar()->showMessage(QStringLiteral("Clean compression: open a shot first"), 5000);
        return;
    }
    if (cleaning_) {
        statusBar()->showMessage(QStringLiteral("Clean compression: already cleaning this shot"), 5000);
        return;
    }
    if (const auto side = shot_clean_sidecar()) {   // a cleaned copy on screen: back to the original
        if (!QFileInfo::exists(side->original)) {
            log(QStringLiteral("Clean compression: the original %1 is not there any more").arg(QDir::toNativeSeparators(side->original)));
            return;
        }
        log(QStringLiteral("Clean compression: back to the original %1").arg(QDir::toNativeSeparators(side->original)));
        const double before = side->grid_before;
        const QString original = side->original;
        open_media(original);
        // Said of the original, whenever its frames land (open_media may finish later).
        compression_status_shot_ = same_path(original);
        if (source_panel_)
            source_panel_->set_compression_status(
                QStringLiteral("The original: compression artifacts (block grid %1; a clean picture is 1.00). Check "
                               "compression uses the cleaned copy again").arg(before, 0, 'f', 2),
                QStringLiteral("warn"), QStringLiteral("Check compression"));
        return;
    }
    const QString source = shot_source(shot_video_, frames_);
    const bool movie = shot_video_.has_value();
    if (!source_is_lossy()) {
        statusBar()->showMessage(QStringLiteral("Clean compression: %1 is not JPEG or MP4-compressed: nothing to clean")
                                     .arg(QFileInfo(source).fileName()), 6000);
        compression_status(QStringLiteral("OK \u00b7 no compression artifacts: not JPEG or MP4-compressed"), QStringLiteral("ok"));
        return;
    }
    const QString ffmpeg = QStandardPaths::findExecutable(QStringLiteral("ffmpeg"));
    if (ffmpeg.isEmpty()) {
        log(QStringLiteral("Clean compression: ffmpeg is not on the PATH"));
        compression_status(QStringLiteral("Not checked: ffmpeg is not on the PATH"), QStringLiteral("warn"));
        return;
    }
    // The stills to clean (a movie is one file) and the frames to measure.
    std::vector<QString> stills;
    if (!movie)
        for (const auto& f : frames_)
            if (is_lossy_still_path(f.string())) stills.push_back(from_path(f));
    int w = 0, h = 0;
    if (movie && shot_video_->stream) w = shot_video_->stream->width, h = shot_video_->stream->height;
    if (const SdrImage* s = frame_sdr(); s && (w <= 0 || h <= 0)) w = s->width(), h = s->height();
    if (w < 16 || h < 16) {
        log(QStringLiteral("Clean compression: the shot's size is not known yet (show a frame first)"));
        compression_status(QStringLiteral("Not checked yet: show a frame first"), QStringLiteral("warn"));
        return;
    }
    QGuiApplication::setOverrideCursor(Qt::WaitCursor);
    statusBar()->showMessage(QStringLiteral("Clean compression: measuring the block grid on %1 frames…").arg(kCleanSamples));
    compression_status(QStringLiteral("Checking %1 frames for compression artifacts\u2026").arg(kCleanSamples), QStringLiteral("busy"));
    if (source_panel_) {
        source_panel_->set_compression_busy(true);
        source_panel_->repaint();   // the measuring below holds the event loop for a few seconds
    }
    const int n = int(std::size(kCleanQps));
    std::vector<double> sum(std::size_t(n), 0.0);
    int measured = 0;
    QString last_error;   // ffmpeg's own words when a frame could not be measured
    bool no_spp = false;  // an ffmpeg without the GPL filters (an LGPL build on the PATH)
    for (int k = 0; k < kCleanSamples; ++k) {
        QString src = source;
        double seek = -1.0;
        if (movie) {
            const double fps = shot_video_->fps.value_or(24.0);
            const int count = std::max(1, int(frames_.size()));
            seek = (count * (0.1 + 0.2 * k)) / std::max(fps, 1.0);
        } else {
            if (stills.empty()) break;
            src = stills[std::size_t(std::llround(double(k) * double(stills.size() - 1) / std::max(1, kCleanSamples - 1)))];
            if (k > 0 && stills.size() == 1) break;
        }
        QProcess p;
        p.start(ffmpeg, clean_probe_arguments(src, seek));
        if (!p.waitForFinished(120000) || p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0) {
            const QString err = QString::fromUtf8(p.readAllStandardError()).trimmed();
            no_spp = no_spp || err.contains(QStringLiteral("No such filter: 'spp'"));
            last_error = err.section('\n', -1);
            continue;
        }
        const auto grids = clean_stack_grids(p.readAllStandardOutput(), w, h, n);
        if (grids.size() != std::size_t(n)) continue;
        for (int i = 0; i < n; ++i) sum[std::size_t(i)] += grids[std::size_t(i)];
        ++measured;
    }
    QGuiApplication::restoreOverrideCursor();
    if (source_panel_) source_panel_->set_compression_busy(false);
    if (!measured) {
        log(QStringLiteral("Clean compression: ffmpeg could not measure %1%2").arg(QDir::toNativeSeparators(source))
                .arg(last_error.isEmpty() ? QString() : QStringLiteral(": ") + last_error));
        statusBar()->clearMessage();
        // An ffmpeg built without the GPL filters (an LGPL build on the PATH instead of RUDRA's own) has no spp.
        compression_status(no_spp
                               ? QStringLiteral("Not checked: this ffmpeg has no spp filter (an LGPL build); RUDRA's own ffmpeg has it")
                               : QStringLiteral("Not checked: ffmpeg could not read the shot (see the log)"),
                           QStringLiteral("warn"));
        return;
    }
    std::vector<double> mean(sum.size());
    for (std::size_t i = 0; i < sum.size(); ++i) mean[i] = sum[i] / measured;
    const int qp = choose_cleanup_qp(kCleanQps, mean, kCleanTarget);
    const auto at = std::size_t(std::find(std::begin(kCleanQps), std::end(kCleanQps), qp) - std::begin(kCleanQps));
    if (qp == 0) {
        const QString t = QStringLiteral("Clean compression: no block grid to clean in %1 (%2, a clean frame 1.00; cleaned "
                                         "from %3)").arg(QFileInfo(source).fileName()).arg(mean[0], 0, 'f', 2).arg(kCleanTarget, 0, 'f', 2);
        log(t);
        statusBar()->showMessage(t, 8000);
        compression_status(QStringLiteral("OK \u00b7 no compression artifacts (block grid %1; a clean picture is 1.00)")
                               .arg(mean[0], 0, 'f', 2), QStringLiteral("ok"));
        return;
    }
    // The copy: a folder in the app's cache. The disk: FFV1 keeps about 60 % of the raw 4:2:0 picture.
    const QString dir = clean_cache_dir(source, qp);
    QDir().mkpath(dir);
    const QString copy = movie ? QDir(dir).filePath(QFileInfo(source).completeBaseName() + QStringLiteral("_clean.mkv"))
                               : stills.size() == 1 ? QDir(dir).filePath(QFileInfo(stills.front()).completeBaseName() + ".png")
                                                    : dir;
    CleanSidecar side{QFileInfo(source).absoluteFilePath(), clean_filter(qp), qp, mean[0], mean[at], true,
                      movie ? int(frames_.size()) : int(stills.size()), QDateTime::currentDateTime().toString(Qt::ISODate)};
    if (read_clean_sidecar(copy)) {   // made before for this file and strength
        log(QStringLiteral("Clean compression: the cleaned copy made earlier (%1)").arg(QDir::toNativeSeparators(copy)));
        open_media(copy);
        return;
    }
    const double need = movie ? double(w) * h * 1.5 * 0.6 * double(std::max<std::size_t>(frames_.size(), 1))
                              : double(w) * h * 3.0 * 0.6 * double(stills.size());
    const QStorageInfo disk(dir);
    if (disk.isValid() && double(disk.bytesAvailable()) < need + 1e9) {
        log(QStringLiteral("Clean compression: about %1 GB is needed in %2 and %3 GB is free")
                .arg(need / 1e9, 0, 'f', 1).arg(QDir::toNativeSeparators(dir)).arg(double(disk.bytesAvailable()) / 1e9, 0, 'f', 1));
        compression_status(QStringLiteral("Compression artifacts found (block grid %1), but there is not enough disk space to "
                                          "clean them (see the log)").arg(mean[0], 0, 'f', 2), QStringLiteral("warn"));
        return;
    }
    log(QStringLiteral("Clean compression: %1, block grid %2 (a clean frame 1.00): %3 takes it to %4; a cleaned copy into %5")
            .arg(QFileInfo(source).fileName()).arg(mean[0], 0, 'f', 2).arg(clean_filter(qp)).arg(mean[at], 0, 'f', 2)
            .arg(QDir::toNativeSeparators(dir)));
    compression_status(QStringLiteral("Compression artifacts found (block grid %1): cleaning them\u2026").arg(mean[0], 0, 'f', 2),
                       QStringLiteral("busy"));
    cleaning_ = true;
    sync_checks();
    // One ffmpeg after another: the movie, or each still.
    auto jobs = std::make_shared<std::vector<QStringList>>();
    if (movie) jobs->push_back(clean_movie_arguments(source, copy, qp));
    else
        for (const auto& s : stills)
            jobs->push_back(clean_still_arguments(s, QDir(dir).filePath(QFileInfo(s).completeBaseName() + ".png"), qp));
    const int total = movie ? std::max(1, int(frames_.size())) : int(jobs->size());
    auto next = std::make_shared<std::function<void(int)>>();
    const std::weak_ptr<std::function<void(int)>> weak_next = next;
    QPointer<MainWindow> self(this);
    *next = [self, ffmpeg, jobs, weak_next, total, movie, copy, side](int i) {
        if (!self) return;
        if (i >= int(jobs->size())) {
            self->cleaning_ = false;
            write_clean_sidecar(copy, side);
            self->log(QStringLiteral("Clean compression: done; the cleaned copy is the shot now (Reconstruct > Clean "
                                     "compression again goes back to the original)"));
            self->open_media(copy);
            self->statusBar()->showMessage(QStringLiteral("Clean compression: using a cleaned copy (%1; block grid %2 -> %3)")
                                               .arg(side.filter).arg(side.grid_before, 0, 'f', 2).arg(side.grid_after, 0, 'f', 2), 10000);
            self->sync_checks();
            return;
        }
        const auto next = weak_next.lock();   // held by the running ffmpeg until it ends
        if (!next) return;
        auto* p = new QProcess(self);
        QObject::connect(p, &QProcess::readyReadStandardOutput, p, [self, p, total, movie, side] {
            if (!self || !movie) return;
            const QString out = QString::fromUtf8(p->readAllStandardOutput());
            const qsizetype at = out.lastIndexOf(QStringLiteral("frame="));
            if (at < 0) return;
            const int f = out.mid(at + 6).section('\n', 0, 0).trimmed().toInt();
            self->statusBar()->showMessage(QStringLiteral("Clean compression: frame %1 of %2").arg(f).arg(total));
            self->compression_status(QStringLiteral("Compression artifacts found (block grid %1): cleaning them, frame %2 of %3")
                                         .arg(side.grid_before, 0, 'f', 2).arg(f).arg(total), QStringLiteral("busy"));
        });
        QObject::connect(p, &QProcess::finished, p, [self, p, i, next, total, movie](int code, QProcess::ExitStatus st) {
            p->deleteLater();
            if (!self) return;
            if (st != QProcess::NormalExit || code != 0) {
                self->cleaning_ = false;
                self->log(QStringLiteral("Clean compression: ffmpeg failed: %1").arg(QString::fromUtf8(p->readAllStandardError()).trimmed()));
                self->statusBar()->showMessage(QStringLiteral("Clean compression failed (see the log)"), 8000);
                self->compression_status(QStringLiteral("Cleaning failed (see the log); the shot is the original"), QStringLiteral("warn"));
                self->sync_checks();
                return;
            }
            if (!movie) self->statusBar()->showMessage(QStringLiteral("Clean compression: still %1 of %2").arg(i + 1).arg(total));
            (*next)(i + 1);
        });
        p->start(ffmpeg, (*jobs)[std::size_t(i)]);
    };
    (*next)(0);
}

}  // namespace rudra::app
