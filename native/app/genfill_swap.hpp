#pragma once
// AI fill fixes the master in place (4 Nov 2026, Ahmed: option a). After a fill has
// finished without error, each filled frame replaces its master frame, and the
// original moves to <master>_before_aifill (a new folder, never overwritten: _2, _3 ...),
// so nothing is lost and moving them back restores the master. The fill's own folder
// keeps its review pictures and report. A failure part-way puts that frame back and
// stops: every frame is either the original or the filled one, never neither.
#include <QString>

namespace rudra::app {

struct GenFillSwap {
    int replaced = 0;     // master frames now holding the fill
    int skipped = 0;      // filled frames with no master frame of that name (left in the fill folder)
    QString backup;       // where the originals went
    QString error;        // empty when every frame was swapped
};

GenFillSwap genfill_replace_in_master(const QString& master_dir, const QString& fill_dir);

}  // namespace rudra::app
