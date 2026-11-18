#pragma once
// The smart bins on import (1.2.6, 18 Nov 2026, user feedback: "smart bins, what are they for? they
// do nothing"). They were set only when a clip had been opened and measured, so a fresh import left
// them empty. Each imported clip is now looked at in the background: a still once, a folder of
// frames and a movie at three places (10 %, 50 %, 90 %), and the share of its pixels that are
// clipped (max(R, G, B) code 254 or more) or crushed (code 1 or less, black bars left out) decides
// the bins. A movie gets its thumbnail from the same frames.

#include <QImage>
#include <QString>
#include <QStringList>

namespace rudra::app {

// A clip is in "Clipped highlights" from this share of clipped pixels in its worst frame, and in
// "Crushed shadows" from this share of crushed ones (letterbox bars are not counted).
inline constexpr double kScanClippedPct = 0.05;
inline constexpr double kScanCrushedPct = 0.5;

struct FrameCodes {
    double clipped_pct = 0.0, crushed_pct = 0.0;
};
// One 8-bit frame's shares. Rows and columns that are black all along (bars) are left out of the crushed count.
FrameCodes frame_codes(const QImage& frame);

struct MediaScan {
    QString path;
    bool ok = false;
    double clipped_pct = 0.0, crushed_pct = 0.0;   // the worst frame's
    int frames = 0;                                 // frames looked at
    QImage thumb;                                   // the middle frame, small
    QString info;                                   // "1920×1080"
    QString error;
};
// Blocking (run it off the UI thread): a still, a folder of frames or a movie (through ffprobe and
// ffmpeg, found on the PATH or given).
MediaScan scan_media_file(const QString& path, const QString& ffmpeg = QStringLiteral("ffmpeg"),
                          const QString& ffprobe = QStringLiteral("ffprobe"));

}  // namespace rudra::app
