# RUDRA 1.4 plan: the Batch tab

Written 24 Nov 2026, after the mockup was approved (Design canvas "RUDRA Batch tab mockup",
https://claude.ai/artifact/9snKd3H2xDvVK4JPdV5JJf: the tab and the sheet shown before a batch starts).
Status: **moved into 1.3.0 (24 Nov 2026, Ahmed)**. Built in 1.3: the tab with both queues in one list (steps 3 and 6
in part; see "In 1.3.0" at the end). Steps 1, 2, 4, 5 (one runner for both routes, presets, the sheet before
starting) stay for 1.3.x / 1.4.

## What it is

A third tab, **Reconstruct · Deliver · Batch**, kept minimal: one list of shots (thumbnail, name, one line
for route, AI fill and preset, a status), one panel for the selected shot (status in plain words, one
action, then Route, Preset, AI fill; the rest behind **Details**), and a footer with one progress line and
one button (**Pause**). A short sheet before the start shows only what needs a decision ("SH030 is on the
learned route", "v001 already exists") and folds the rest into "Everything else is ready".

It replaces both queues there are today: Deliver › Batch queue… (`batch_window.cpp`, learned route only)
and the analytic route's Queue window (`queue_window.cpp`, movie exports). One queue, both routes.

## Nothing is lost: where each current feature goes

Each row becomes an `AppBatch.*` test, so "nothing lost" is checked, not promised.

| Today | In the Batch tab |
|---|---|
| Add files / folder, numbered frames grouped as sequences, drop, Add current shot (Ctrl+B) | The **+** menu, drop on the list, Ctrl+B |
| Import EDL (event, reel, record in/out kept as provenance) | **+** › Import EDL…; the event under Details |
| Per-shot frame range and name (double-click to edit) | Details: Frames and Name, editable (double-click still works) |
| Start frame (1001), timecode / reel | Details |
| Source display (BT.1886 / sRGB / Gamma 2.2) | Details, beside Route (Auto uses it) |
| Learned-route trims (strength, saturation, peak, exposure, contrast) | Details, shown when the route is Learned |
| Apply Deliver settings to the selected shots | Preset menu › "Current Deliver settings" |
| Analytic export formats: PQ, ACEScct, LogC4 (Export sheet) | Inside presets |
| AI fill on/off for many shots | The switch acts on a multi-selection |
| Up / Down, Remove, Retry, Clear done | Drag to reorder; right-click menu |
| Stop now / stop after the shot | Pause = after this shot; Alt+Pause = Stop now |
| MaxCLL / MaxFALL, flagged above 2x the batch median | Shown on finished shots; the flag counts as Needs attention |
| Per-job Stop / Resume (Queue window) | Right-click on the row |
| Crash-safe queue file (an interrupted job comes back pending) | Kept: the same file |
| `rudra batch status` (CLI) | Kept: the tab reads the same state files |

## Two things that are new work, not moves

1. **One runner for both routes.** The analytic queue has only ever done movie exports. Merged, every job
   goes through one runner with two lanes: a GPU lane (learned master, AI fill: one at a time; waits while
   another app holds the GPU, as `tmp/RUN_V5_ANALYTIC_123.ps1` does: over 6 GB in use) and a CPU lane
   (analytic jobs beside it).
2. **Old queue files.** 1.3 users may have a saved batch queue or export jobs in the old formats: read and
   migrate them, never drop them silently (a test with a 1.3 queue file of each kind).

## Steps

1. Queue model: `BatchJob` covers both routes and the analytic export jobs; migration of 1.3 queue files;
   tests first.
2. Runner: GPU lane and CPU lane, the GPU-busy wait, the per-shot disk check (AI fill about 45 MB a frame).
3. The tab: list, panel, footer, Details, right-click and **+** menus (replaces the dialog).
4. Presets: output formats + size + AI fill; "Current Deliver settings" is one of them.
5. The sheet before a batch: route against source curve, names already there, disk, GPU, time (from the
   learned seconds per frame).
6. Retire Deliver › Batch queue… and the Queue window (both point to the tab); the CLI stays.
7. Tests: the table above, the migration, the two lanes never running two GPU jobs at once.

## Worth adding while there (from the 23 Nov learned-route check)

The app's own per-frame flicker flag (`temporal_excess_stops`, flag at 0.25) flagged none of 03V's frames
on the learned route, where the T gate measured 0.32 EV at frames 27, 51, 53, 74 (the region mask's
one-frame lag on newly clipped pixels; `docs/HANDOFF_1_3_2026-11-23.md`). A batch's Needs review should
use a measure that catches that, such as the T gate's still-clipped p95, run after each AI fill.

## In 1.3.0 (24 Nov 2026)

- `native/app/batch_page.cpp/.hpp`: the Batch tab (Reconstruct · Deliver · Batch; `tabBatch`), shown in place of the
  rails and viewer (`bodyStack`). One list: the batch's shots (thumbnail, name, output, AI fill, frames; status) and
  the video exports (the Queue window's, with Stop / Resume / Remove). Filters All / Needs attention (failed, stopped,
  an EDL warning, a MaxCLL above twice the batch median, a failed export) / Done. The panel: status, one action (Try
  again, Show in folder, Resume, Stop), output with "Use the Deliver tab's settings", the AI fill switch, Details
  (frames, name, folder, start frame, timecode and reel, the SDR's display, trims, MaxCLL / MaxFALL, the EDL event,
  where it was written). The + menu: Add files, Add folder, Add the shot on screen (Ctrl+B), Import EDL, the Queue
  window. Right-click: earlier / later (Ctrl+Up / Down), AI fill on / off, try again, remove (Delete), clear the
  finished. Footer: progress, one line, Start / Pause, Stop now while running.
- Deliver › Batch queue… and Ctrl+Shift+B open the tab; the dialog (`batch_window.cpp`) is no longer built.
- Not yet: the learned and analytic routes in one runner (exports are still made from Deliver's Export sheet),
  presets, drag to reorder, the sheet before starting, the GPU lane wait.
- Tests: `AppBatch.TheBatchTabTakesTheBodyAndGivesItBack`, `AppBatch.EveryControlOfTheOldDialogHasAHomeInTheTab`,
  `Batch.RendersEveryShotThroughTheLoadedModel` (now through the tab; skipped on Windows).
