#pragma once
// RUDRA --headroom-check report.json --movie CLIP [--timeout-ms N] (E1/E3, 17 Oct 2026):
// the experimental headroom preview in the real window with the real decoder,
// bridge and viewer. Developer acceptance only; it opts the window into the
// capability that every public entry point keeps closed. Steps: open the movie,
// turn the refused controls off, enable bt1886-headroom-v1, then paused frame,
// seeks, a grade change and its undo (bit-identical pixels), playback to the end
// (timed), a transactional HDR10 export, the viewer's upload/present/readback, and the
// switch back to the legacy engine. Every step is timed into a JSON report;
// existing report paths are refused. Requires Windows source leases.

#include <QString>

namespace rudra::app {

class MainWindow;

struct HeadroomCheckArgs {
    QString report;
    QString movie;             // limited-range Rec.709, 8/10-bit YUV 4:4:4 (the v1 contract)
    int timeout_ms = 120000;   // per asynchronous step
};

int run_headroom_check(MainWindow& w, const HeadroomCheckArgs& a);

}  // namespace rudra::app
