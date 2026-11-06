#pragma once
// AI fill fixes the master in place (4 Nov 2026, Ahmed: option a). After a fill has
// finished without error, each filled frame replaces its master frame, and the
// original moves to <master>_before_aifill (a new folder, never overwritten: _2, _3 ...),
// so nothing is lost and moving them back restores the master. The fill's own folder
// keeps its review pictures and report. A failure part-way puts that frame back and
// stops: every frame is either the original or the filled one, never neither.
#include <QString>
#include <QStringList>

namespace rudra::app {

struct GenFillSwap {
    int replaced = 0;     // master frames now holding the fill
    int skipped = 0;      // filled frames with no master frame of that name (left in the fill folder)
    QString backup;       // where the originals went
    QString error;        // empty when every frame was swapped
};

GenFillSwap genfill_replace_in_master(const QString& master_dir, const QString& fill_dir);

// The filled frames, in frame order, for the viewer's AI fill: the fill folder's EXRs
// while they are there, else, after a fill was moved into the master in place (a
// <master>_before_aifill* folder beside it holds the originals), the master's EXRs.
// Empty when neither holds a fill.
QStringList genfill_filled_frames(const QString& fill_dir, const QString& master_dir);

}  // namespace rudra::app
