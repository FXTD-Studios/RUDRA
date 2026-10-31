# Headroom project compatibility and shared decoding — 16 October 2026

Completed the storage/compatibility increment and the common indexed decoder
needed by paused frames, playback and export. **302 native, 60 app and 20 Python
tests pass.** Headroom remains unavailable in the app and CLI; these changes do
not claim their pixel pipelines are connected yet.

## Saved settings and compatibility

- Moved the versioned policy identifier into `core/headroom_policy.hpp` so
  storage and rendering share one definition. No reconstruction math changed.
- Legacy projects still write **version 1**, with their existing fields and
  defaults. Headroom projects write **version 2**, with
  `decode.headroom_policy: "bt1886-headroom-v1"` and explicit grain-settling state.
- Project readers default to `ProjectReadSupport::LegacyOnly`. A caller must
  explicitly declare headroom support to read version 2. The current app keeps
  the default and refuses before changing the open shot, grade, dirty state or
  project path. Older native readers' existing maximum-version check also
  rejects version 2; this is a format guarantee, not an old-binary runtime test.
- Storage validates the supported analytic/movie/grade combination. Unsupported
  settings fail instead of being clamped into the app's range. Project exposure
  remains −3 to +3 EV and region EV −4 to +4, matching existing UI controls;
  the isolated renderer's wider limits were not changed.
- Missing/corrupt painted masks fail headroom loading. Saving a parsed project
  whose referenced masks have not been loaded also fails before writes. Legacy
  missing-mask behavior is unchanged. Invalid saves preserve the existing file.
- The native master-params reader refuses project JSON and non-disabled headroom
  policies. Python inference, field preview, master preflight and master writing
  refuse them before decoding, inference, staging or output. Neither silently
  drops the policy into the legacy SDR path. Python has no native-project loader;
  old Python installations need the new guard for parameter-level protection.

## Common frame selection and cache identity

- `FloatVideoDecoder` accepts an optional `FloatFrameRange{first, count}` with
  strict bounds. Both paused reads (`count=1`) and playback ranges now use its
  exact same float conversion. Omitted range keeps full-export behavior.
- Selection uses decoded-frame indices through FFmpeg `trim`, with output count
  limited explicitly. It does not approximate frame indices with timestamp seeks.
  A seven-frame, 10-bit BT.709 FFV1 test at **24000/1001 fps**, starting at **5 s**,
  verifies bit-identical samples for random single-frame reads, playback from the
  middle and full sequential decode. Source tags and signed/headroom samples stay
  in the existing `ExtendedVideoFrame` contract.
- The initial generated fixture omitted a required input colour tag and was
  correctly rejected. Fixed the fixture's input tags; no eligibility relaxation.
  Initial failure evidence is retained separately.
- `headroom_source_cache_id` creates a dedicated `headroom-float-v1-<sha256>`
  namespace from full source and decoder-file hashes, resolved paths, policy,
  contract, dimensions, clock and decoding arguments. Tests change source bytes
  while preserving path, size and modification time and verify a different key.
  Decoder bytes, dimensions, frame count and matrix also change the key.
- This is **cache identity only**: no disk cache or eviction owner is implemented,
  and no legacy PNG cache is read or rewritten. Each identity calculation reads
  the source and decoder executable in full; future callers should compute it
  at source-open, not per frame. They must guard active-file changes and reject
  invalid/incomplete cached payloads. Indexed trim traverses preceding frames;
  optimized seeking, bounded caching/prefetch and their costs remain to measure.
- Decoder cancellation/EOF behavior remains intact and its existing tests pass.
  Its threading contract still requires joining read/interrupt operations before
  finish/close/destruction. No new asynchronous cache or feeder is claimed.

## Evidence

| Check | Result | Report prefix under `reports/` |
| --- | --- | --- |
| Focused native | 45/45, 2.197 s | `headroom_storage_focused_20261016` |
| Complete native | 302/302, 34.678 s | `headroom_storage_native_20261016` |
| Complete offscreen app | 60/60, 48.742 s | `headroom_storage_app_20261016` |
| Python compatibility + master-render regression | 20/20, 2.14 s | `headroom_storage_python_initial_20261016.log` |

Native/app reports have `.json` and `.log` files. No failed, disabled or skipped
tests in the final native/app runs. Release build logs:
`headroom_storage_build_20261016.log`, `headroom_storage_final_build_20261016.log`.
Both builds succeeded; final build has no compiler errors/warnings. Initial
focused run (`headroom_storage_focused_initial_20261016`) records the fixture
failure above. Whitespace checks pass. Prior highlight-preservation and legacy
color/delivery tests are included in the complete native run.

## Next integration acceptance criteria

1. Build the bounded float frame cache/source owner and playback feeder on the
   range API. Include source/decoder invalidation, cancellation, errors, byte
   budget/eviction, cache payload verification and full-resolution identity.
   Keep float data separate from the legacy `SdrImage`/PNG paths.
2. Connect session policy/undo and the preview/display/scopes/probes to the shared
   render output. Storage support alone is not permission to remove the reader
   guard. Do not run the old SDR GPU composite on these pixels.
3. Connect EXR/movie export jobs and write actual policy, source and grade
   provenance. Refuse unsupported temporal/artist combinations explicitly.
4. Prove paused/playback/export parity at the app level, encoded-output quality,
   memory/throughput, GPU parity and calibrated display behavior before UI
   enablement. Eligibility remains tagged 8/10-bit YUV444; 4:2:0 needs validation.

No source dataset writes, new training, downloads, model promotion, app restart,
installer refresh, commit or release. The live application executable was not
rebuilt; test binaries were. Commercial rights, signing, clean-machine acceptance
and visual HDR review remain unresolved. Distribution packages remain stale.
