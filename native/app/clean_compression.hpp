#pragma once
// RUDRA 1.3 (24 Nov 2026): Reconstruct > Clean compression. A JPEG still or an H.264/HEVC movie carries its
// codec's block grid and ringing, which the HDR expansion makes stronger. RUDRA measures the grid on a few
// frames, cleans it with ffmpeg's spp filter at the smallest strength that takes the grid down to a clean
// picture's (core/block_grid.hpp), and opens a cleaned copy (lossless FFV1 or PNG, in the app's cache) as the
// shot: the viewer, both routes, the AI fill and the QC report all read the same pictures. The original is not
// touched; the action again goes back to it. A sidecar beside the copy says what was done.
#include <QByteArray>
#include <QString>
#include <QStringList>

#include <optional>
#include <vector>

namespace rudra::app {

inline constexpr int kCleanQps[] = {0, 2, 4, 6, 8};   // spp's qp, tried in this order (0: none)
inline constexpr double kCleanTarget = 1.10;          // the block grid a clean frame shows, with a margin
inline constexpr int kCleanSamples = 5;               // frames measured across the shot

// "spp=quality=3:qp=4" (0: no filter).
QString clean_filter(int qp);
// ffmpeg's arguments for one measured frame: the frame at `seek_seconds` (a movie; < 0 for a still) cleaned at
// every kCleanQps strength, stacked top to bottom as 8-bit luma on stdout.
QStringList clean_probe_arguments(const QString& source, double seek_seconds);
// The block grid of each picture of such a stack (width x height each, `count` of them).
std::vector<double> clean_stack_grids(const QByteArray& raw, int width, int height, int count);
// ffmpeg's arguments for the cleaned copy: a movie into FFV1 in Matroska (its own pixel format, colour tags, audio,
// timecode and rate kept), a still into an 8-bit PNG.
QStringList clean_movie_arguments(const QString& source, const QString& out, int qp);
QStringList clean_still_arguments(const QString& source, const QString& out, int qp);
// The cleaned copies of `source` live here (the app's cache, one folder per source file and strength).
QString clean_cache_dir(const QString& source, int qp);

struct CleanSidecar {
    QString original;        // the source as it came in (a movie, a still, or a folder of stills)
    QString filter;          // "spp=quality=3:qp=4"
    int qp = 0;
    double grid_before = 1.0, grid_after = 1.0;   // block_grid, the mean of the measured frames
    bool original_lossy = true;
    int frames = 0;
    QString made;            // ISO 8601
};
// The sidecar of a cleaned copy: <movie or still>.rudra_clean.json, or rudra_clean.json in a folder of stills.
QString clean_sidecar_path(const QString& copy);
std::optional<CleanSidecar> read_clean_sidecar(const QString& copy);
bool write_clean_sidecar(const QString& copy, const CleanSidecar& s);

}  // namespace rudra::app
