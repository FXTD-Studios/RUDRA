#pragma once
// RUDRA --workflow-check (Phase 3 step 12): the Studio workflow, scripted, in
// the real window: optionally open a package, open a folder of frames, scrub every frame,
// grade and undo, compare, probe and measure, and master three frames; with
// --movie, a movie opened as a shot, every frame scrubbed and its HDR10 export
// run through the queue to a master that passed QC. Every
// step is timed and checked and the whole is written to a JSON report; the
// masters are then held to the Studio-held CLI path by
// `rudra-native master-compare` (scripts/NATIVE_PHASE3_EXIT.ps1 runs both).
// --analytic skips model loading and checks analytic master provenance. Existing
// report/output paths are refused. Unexpected message dialogs are cancelled and
// recorded as failures; asynchronous waits are bounded by timeout_ms.

#include <QString>

namespace rudra::app {

class MainWindow;

struct WorkflowArgs {
    QString report;           // where the JSON goes
    QString package;          // the model package
    QString frames;           // a folder of frames
    QString backend;          // a BackendChoice key, or "" for automatic
    QString out;              // where the masters go (a fresh folder)
    int check_every = 24;     // a delivered frame is compared with its own decode every N frames
    QString movie;            // Phase 4: a movie, opened, scrubbed and exported to HDR10 through the queue
    bool analytic = false;   // do not load or execute a model package
    int timeout_ms = 120000; // maximum wait per asynchronous step, also used by failure-path tests
};

// Runs in the event loop of `w`; returns the process exit code (0: every step passed).
int run_workflow_check(MainWindow& w, const WorkflowArgs& a);

}  // namespace rudra::app
