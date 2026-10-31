# Production continuation — 17 October 2026

The owner requested a remaining-work file first, then implementation. The active
ledger is [REMAINING_WORK.md](REMAINING_WORK.md). The full roadmap is **not complete**.
This increment completes several bounded foundations and workflow improvements;
public headroom selection, commercial clearance and release gates remain open.

## Session and preview foundation

- Headroom policy now lives in `GradeSnapshot`, giving session undo/redo and
  project storage one authoritative value. The project JSON layout is unchanged:
  enabled policy still uses version 2 `decode.headroom_policy`; legacy files stay
  version 1. C++ callers now use `project.grade.headroom_policy`.
- Default `Session` and project readers reject headroom. Explicitly capable
  sessions accept the policy; unsupported loads fail before mutating the grade,
  delivery settings or undo stacks. Parameter JSON carries enabled policy so
  legacy master parsing refuses it. Existing legacy/session goldens still pass.
- Added `HeadroomPresentation`, an isolated session-to-preview bridge over the
  existing asynchronous decoder and CPU baseline/grade. Paused and continuous
  ranges share the same path. It snapshots controls without silently changing
  them; unsupported controls revoke old output. Model/temporal hold must be off.
- Grade revisions and decoder generations reject old/foreign/out-of-order frames,
  including after undo, seek, cancel or a late decode failure. Final synchronous
  display callbacks serialize with source/seek/grade changes. No conversion to
  canonical SDR, export receipt, persistent cache or UI enablement was added.

Integration contract: snapshot Session on its owner thread after each change;
run `render_next` off the UI thread; call `present` synchronously on the display
thread. Its callback must not re-enter the bridge or queue naked pixels for later
presentation. Join external callers before destruction. A paused grade change
needs another seek. Source tokens retain their shared pixel charge; the baseline
and graded CPU images are **additional memory outside that allowance**. MainWindow
routing, scopes, GPU upload/display lifetime and export transactions remain work.

## Application improvements

- The Source card displays actual per-frame still profile/primaries, explicit
  Rec.709 fallback and applied EXIF orientation. Metadata comes from the existing
  decoder cache, with no additional file read. Switching frames or sources updates
  the card. Profile names are plain text, including markup-looking names.
- `--workflow-check --analytic` exercises the real app without loading weights.
  It checks frame identity, grading/undo/redo, comparison, measurement, three EXR
  masters and optional movie HDR10 delivery. EXR sidecars must identify analytic
  reconstruction. Existing model-backed checks continue to work.
- Unexpected message dialogs are cancelled and recorded as failure. Waits default
  to 120 seconds per asynchronous step, configurable with `--timeout-ms` up to one
  hour. This does not bound synchronous file opening or an OS storage stall.
  Existing output/report paths are refused; report-write failure returns failure.
  Model-load callback state survives a timeout safely. Requested and presented
  frame indices are checked separately so stale pixels cannot satisfy a wait.
- Windows builds stage the configured ONNX DLLs beside app, CLI and test binaries.
  The native tests had been finding System32's 1.17 runtime despite a 1.22 SDK on
  PATH. Staged DLL hashes now match the configured 1.22 package; the full native
  run no longer emits the API 22/17 warning. No neural-quality acceptance implied.
- Fixed the layer-check tool's existing `KeyError: bench`: standalone diagnostic
  targets are treated like tests; production layer rules remain unchanged.

## Validation

- **358/358 native tests pass**, 66.584 s; eight new session/presentation cases.
  Policy identity, byte-compatible legacy state, restore refusal, unsupported
  controls, pixel parity, foreign/stale/out-of-order frames, serialized final
  presentation and late failures are covered.
- **65/65 app tests pass**, 55.162 s; five new profile/workflow cases. Existing
  image/folder/movie, export format matrix, playback, projects and F shortcut pass.
- **10/10 Python compatibility tests pass**, 10.18 s (console result).
- **40/40 shuffled presentation executions pass**, four new decoder/presentation
  lifecycle cases repeated ten times. This is not exhaustive race/sanitizer proof.
- Real Windows app, rebuilt executable: standalone analytic check **PASS**, three
  stills and the existing six-frame movie, all nine recorded steps. No unexpected
  dialog or stderr. This is developer-machine acceptance, not a clean install or
  calibrated HDR judgment.
- Release builds, layer check and tracked diff whitespace checks pass. Initial
  native test compile errors (const buffer accessor and a stats member name) were
  fixed; the failed build log is retained. No tolerance was relaxed.

Evidence: `reports/production_continuation_native_20261017.{json,log}`,
`production_continuation_app_20261017.{json,log}`, final build log,
`headroom_session_focused_20261017.{json,log}` (23),
`analytic_workflow_focused_20261017.{json,log}` (4), and
`reports/production_continuation_20261017/standalone.json` with exports/logs.
Stress evidence: `reports/headroom_presentation_stress_20261017.log`.
Deliberate dialog/argument failures in app-test logs are expected passing tests.

## Preview benchmark and remaining performance gate

Eight completed runs: HD/UHD, eight frames each, direct CPU renderer and the new
session bridge, two repetitions with reversed order. Both render baseline and
+1 EV grade, validate source identity, and hash both images. Same protected source
preparation is reused, timed separately. All output hashes agree and each run
uses one decoder. Inputs are existing synthetic FFV1 clips, unchanged after runs.

| Measurement | HD | UHD |
|---|---:|---:|
| Initial preparation | 1.405 s | 5.609 s |
| Direct CPU feed/render/hash, 8 frames (median of 2) | 2.829 s | 14.970 s |
| Session bridge feed/render/hash, 8 frames (median of 2) | 4.212 s | 14.183 s |
| Bridge first rendered frame, after preparation (median) | 0.670 s | 2.198 s |
| Bridge source-pixel peak | 94.92 MiB | 189.84 MiB |

**The new CPU preview is not ready for real-time use.** UHD is approximately
0.56 fps for this decode/render/hash workload, before GPU presentation. This is
not the prior decode-only 10.52 fps benchmark or the existing GPU application's
performance. HD runs vary considerably; no general speedup claim is justified.
Rendered images and FFmpeg/GPU memory are not included in the source-pixel peak.
Do not enable the mode based on correctness results alone.

Artifacts: `reports/production_continuation_20261017/presentation_{hd,uhd}.json`.
The combined validation record and source/binary hashes are saved beside them.
Reproduce the opt-in diagnostic target `rudra_headroom_presentation_bench` with
`<source.mkv> <fresh-report.json>`. It limits each run to the first eight frames;
it is not installed or included in default builds. No neural or competitor
quality scores were generated by this benchmark.

## Continue here

1. Connect the guarded bridge to MainWindow's source lifecycle, paused/playing
   display, grade notifications and project restore under explicit capability
   gates. Keep public selection closed. Budget rendered CPU/GPU copies as well
   as source tokens; never hold a decoder/presentation lock across queued UI work.
2. Profile the CPU render cost and establish a colour-correct efficient preview
   path; preserve the frozen pixel-reference results. Then scopes, supported
   controls, provenance and transactional exports must all pass before enablement.
3. Remaining source overrides, trial/expanded delivery, readable QC/recovery,
   production-footage/GPU/platform/package tests and full AI work remain in the
   ledger. Existing evaluation ZIP/installer need another refresh before use.

External gates persist: calibrated owner HDR review, licensed competitor outputs,
commercial/legal decisions, signing credentials, clean Windows environment and
Mac acceptance. No training, downloads, G: dataset writes, model promotion,
package publication, commit or push occurred.
