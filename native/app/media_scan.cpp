#include "media_scan.hpp"

#include <QDir>
#include <QFileInfo>
#include <QImageReader>
#include <QProcess>

#include <algorithm>
#include <vector>

#include "media_library.hpp"
#include "rudra/media/sequence.hpp"

namespace rudra::app {

FrameCodes frame_codes(const QImage& in) {
    FrameCodes r;
    if (in.isNull()) return r;
    const QImage img = in.convertToFormat(QImage::Format_RGB888);
    const int w = img.width(), h = img.height();
    std::vector<std::uint8_t> top(std::size_t(w) * std::size_t(h));
    for (int y = 0; y < h; ++y) {
        const uchar* p = img.constScanLine(y);
        for (int x = 0; x < w; ++x) top[std::size_t(y) * w + x] = std::max({p[3 * x], p[3 * x + 1], p[3 * x + 2]});
    }
    // Black bars: whole rows / columns at code 4 or less are not picture.
    std::vector<char> bar_row(std::size_t(h), 1), bar_col(std::size_t(w), 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (top[std::size_t(y) * w + x] > 4) bar_row[std::size_t(y)] = bar_col[std::size_t(x)] = 0;
    std::size_t clipped = 0, crushed = 0, picture = 0;
    for (int y = 0; y < h; ++y) {
        if (bar_row[std::size_t(y)]) continue;
        for (int x = 0; x < w; ++x) {
            if (bar_col[std::size_t(x)]) continue;
            const std::uint8_t c = top[std::size_t(y) * w + x];
            ++picture;
            clipped += c >= 254;
            crushed += c <= 1;
        }
    }
    const std::size_t all = std::size_t(w) * std::size_t(h);
    r.clipped_pct = all ? 100.0 * double(clipped) / double(all) : 0.0;
    r.crushed_pct = picture ? 100.0 * double(crushed) / double(picture) : 0.0;
    return r;
}

namespace {

QImage read_still(const QString& path) {
    QImageReader r(path);
    r.setAutoTransform(true);
    QImage img = r.read();
    // Big frames are looked at with every 2nd (4th) pixel: the codes stay what they are.
    while (!img.isNull() && img.width() > 2400) img = img.scaled(img.width() / 2, img.height() / 2, Qt::IgnoreAspectRatio, Qt::FastTransformation);
    return img;
}

QImage movie_frame(const QString& ffmpeg, const QString& path, double at) {
    QProcess p;
    // Nearest-neighbour down to 960 wide at most, so every code is one the file holds.
    p.start(ffmpeg, {"-v", "error", "-ss", QString::number(at, 'f', 3), "-i", path, "-frames:v", "1", "-vf",
                     "scale='min(960,iw)':-2:flags=neighbor", "-f", "image2pipe", "-vcodec", "png", "-pix_fmt", "rgb24", "-"});
    if (!p.waitForFinished(30000) || p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0) {
        p.kill();
        return {};
    }
    return QImage::fromData(p.readAllStandardOutput(), "PNG");
}

double movie_seconds(const QString& ffprobe, const QString& path, QSize* size) {
    QProcess p;
    p.start(ffprobe, {"-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration:format=duration", "-of",
                      "default=noprint_wrappers=1", path});
    if (!p.waitForFinished(10000) || p.exitCode() != 0) {
        p.kill();
        return -1.0;
    }
    // The video stream's own length first: the file's can be the audio's, longer (a seek past the
    // last frame reads nothing).
    double d = -1.0;
    int w = 0, h = 0;
    for (const QString& line : QString::fromUtf8(p.readAllStandardOutput()).split('\n')) {
        const QString l = line.trimmed();
        bool ok = false;
        if (l.startsWith("duration=")) {
            const double v = l.mid(9).toDouble(&ok);
            if (ok && v > 0 && (d < 0 || v < d)) d = v;
        }
        else if (l.startsWith("width=")) w = l.mid(6).toInt();
        else if (l.startsWith("height=")) h = l.mid(7).toInt();
    }
    if (size) *size = QSize(w, h);
    return d;
}

}  // namespace

MediaScan scan_media_file(const QString& path, const QString& ffmpeg, const QString& ffprobe) {
    MediaScan s;
    s.path = path;
    const QString kind = MediaLibrary::kind_of(path);
    std::vector<QImage> frames;
    if (kind == "still") {
        frames.push_back(read_still(path));
    } else if (kind == "folder") {
        QStringList filters;
        for (const auto& suf : frame_suffixes()) filters << QStringLiteral("*") + QString::fromStdString(suf);
        const QStringList all = QDir(path).entryList(filters, QDir::Files, QDir::Name);
        if (!all.isEmpty())
            for (double f : {0.1, 0.5, 0.9}) {
                const int i = std::clamp(int(f * (all.size() - 1) + 0.5), 0, int(all.size()) - 1);
                frames.push_back(read_still(QDir(path).filePath(all[i])));
            }
    } else {
        QSize size;
        const double d = movie_seconds(ffprobe, path, &size);
        if (d < 0) {
            s.error = QStringLiteral("ffprobe could not read it");
            return s;
        }
        if (size.isValid() && size.width() > 0) s.info = QStringLiteral("%1×%2").arg(size.width()).arg(size.height());
        for (double f : {0.1, 0.5, 0.9}) {
            QImage img = movie_frame(ffmpeg, path, std::max(0.0, d * f));
            if (img.isNull()) img = movie_frame(ffmpeg, path, std::max(0.0, d * f * 0.5));   // past the last frame: earlier
            frames.push_back(img);
        }
    }
    frames.erase(std::remove_if(frames.begin(), frames.end(), [](const QImage& i) { return i.isNull(); }), frames.end());
    if (frames.empty()) {
        if (s.error.isEmpty()) s.error = QStringLiteral("no frame could be read");
        return s;
    }
    for (const QImage& f : frames) {
        const FrameCodes c = frame_codes(f);
        s.clipped_pct = std::max(s.clipped_pct, c.clipped_pct);
        s.crushed_pct = std::max(s.crushed_pct, c.crushed_pct);
    }
    s.frames = int(frames.size());
    const QImage& mid = frames[frames.size() / 2];
    s.thumb = mid.scaled(352, 200, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
    if (s.info.isEmpty() && kind != "movie") {
        const QSize sz = kind == "still" ? QImageReader(path).size() : QSize();
        s.info = sz.isValid() ? QStringLiteral("%1×%2").arg(sz.width()).arg(sz.height())
                              : QStringLiteral("%1×%2").arg(mid.width()).arg(mid.height());
    }
    s.ok = true;
    return s;
}

}  // namespace rudra::app
