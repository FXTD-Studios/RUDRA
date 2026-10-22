# Getting Started and load-to-export review — 16 October 2026

Built and launched the updated native application:
`build/native_phase3/app/Release/RUDRA.exe`. Left it open for user inspection.
Review combines source inspection, automated Qt interaction/exports and a
native-rendered Getting Started screenshot. It is not an exhaustive manual
mouse-through of every setting on the physical display.

## Implemented fixes

- **F fits the view** through the existing zoom-fit action. The menu and
  keyboard sheet expose it; Getting Started and USER_GUIDE describe it.
  Regression verifies F triggers Fit and still types `f` in a text field.
- Getting Started export copy no longer claims all movie formats receive
  HDR10+/Dolby Vision metadata. Metadata depends on the delivery format.
- **ProRes ACEScct/LogC4 export fixed.** Actual movie export exposed a QC
  failure on the pinned FFmpeg: frames carried linear/BT.2020 tags from the
  conversion filter despite output options requesting unknown log tags.
  For log outputs only, `setparams` now clears transfer/primaries metadata
  after conversion. It does not transform pixels. QC remains strict, and
  ordinary PQ/HLG encoder commands are unchanged.

## Verification

**59/59 app tests pass** (32.605 s), **264/264 native tests pass** (36.922 s),
using pinned FFmpeg 7.1.1. Build and diff whitespace checks pass.

| Workflow | Evidence / scope |
| --- | --- |
| Getting Started | Five-step navigation, reopen, close/Escape and click behavior pass; rendered first page inspected without clipping |
| Fit shortcut | Actual Qt keyboard dispatch and text-entry non-interference pass |
| Image/folder input | Existing open/drop, decode, orientation, profiles, frame navigation and cache tests pass |
| Video input | Movie open/scrub/playback, queues and analytic grading tests pass |
| Still/sequence EXR | Existing master-target, sidecar, cancellation and error-path tests pass |
| Folder to HDR10 | Actual output checked for four frames, 25 fps, BT.2020/PQ tags and no-overwrite behavior |
| Folder to HLG/ProRes HQ/4444 | Newly exercised actual exports; codec, frame count and 24 fps checked |
| Movie delivery | New app test completes all eight queue formats: HDR10, HLG, ProRes HQ/4444 PQ, and HQ/4444 ACEScct/LogC4; each traverses delivery QC |
| Other controls | Existing action, undo, project save/reopen, comparison, scopes, startup and playback tests pass |

This covers the offered format/encoding matrix, not every Cartesian combination
of codecs, frame rates, audio modes, resolutions, model choices and artist
settings. Long clips, arbitrary user media and physically calibrated HDR
judgment remain outside this run.

Evidence:
- `reports/workflow_review_app_pass_20261016.{json,log}`
- `reports/workflow_review_native_pass_20261016.{json,log}`
- `reports/getting_started_review_20261016.png`
- `reports/workflow_review_fix_build_20261016.log`

Earlier failed/interrupted review runs are retained. A shortcut-sheet count
and export wait/path expectations were updated for new coverage. The log
export QC failure was a product bug, not waived as a test expectation.

## Remaining improvements / limits

1. **Refresh distribution artifacts:** previously created ZIP/installer do not
   contain these new fixes. Current verified result is the build executable.
2. **Model-free workflow checker:** the standalone `--workflow-check` still
   loads a package first; add an analytic mode so a full no-model UI workflow
   can be reproduced outside the unit-test harness.
3. **Source-dependent export capabilities:** a single still offers EXR only;
   frame folders offer HDR10/HLG/ProRes but no log variants or audio. Movies
   expose ProRes PQ/ACEScct/LogC4. These are current limits, not failed options.
   Extending them requires an explicit product choice and matching tests.
4. **Failure reporting in unattended UI tests:** failed export notifications
   can enter a modal dialog. Add bounded dialog-aware failure handling before
   expanding unattended testing to deliberately failing media.
5. **Manual/production validation:** clean-machine install/uninstall, calibrated
   shadow/black-level review, larger mixed-media jobs, signing and legal
   clearance remain pending. Owner is not ready for visual review and has no
   clean environment yet; do not count local test success as either gate.

No training, dataset edits, release, commit or push occurred. Existing user
artifacts and previous evaluation packages were preserved.
