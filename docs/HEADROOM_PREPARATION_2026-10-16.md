# Reusable headroom preparation — 16–17 October 2026

Status: implementation, native validation and bounded performance checks complete.
This is an isolated native prototype increment. App headroom remains
disabled and the default project-reader guards remain active.

## Change and ownership contract

`PreparedHeadroomSource::prepare` performs the existing complete stream/frame
probe, timing/dimension/source-contract checks and full source/decoder SHA-256
once. Source and decoder read leases span preparation and all sharing owners.
The resolved probe executable is additionally leased during both probe passes.
The prepared handle cannot be constructed from caller-supplied metadata/hashes.
The content identity schema, decode arguments, colour science and delivery math
are unchanged. No persistent cache or source dataset writes were introduced.

`HeadroomStream::open(prepared)` and `HeadroomFrameSource::open(prepared)` share
the immutable video/identity without another probe or hash. Existing path-based
opens delegate to preparation. Hold the prepared handle across seeks and create
a new immutable stream range owner for each seek. The final prepared handle or
decoder owner releases the source/decoder leases; frames alone do not retain them.

Preview tokens and completion receipts remain specific to one range owner.
Cancelling one range cannot cancel another range sharing the source, and a stop
request made after preparation completes cannot revoke the prepared handle.
Exact frame bytes, finite values, EOF/child exit and validated completion are
still required. A decode receipt still does not authorize export publication.

An observed source/decoder identity change now latches in the shared preparation.
Every sharing owner refuses subsequent validation/publication, even after the
timestamp is restored. **Retry requires fresh preparation**; the old batch-only
behaviour of restoring metadata and issuing `request()` is intentionally removed.
Tests now enforce this stronger shared-source contract. Detection is checked at
owner validation boundaries; it is not a broadcast that instantly interrupts
every other decoder. Stop/join those owners before discarding a generation.

## Cancellation

Preparation accepts `HeadroomPrepareOptions` with a `std::stop_token` and an
optional explicit probe path (empty resolves ffprobe on PATH). Probing propagates
the token into both captured child processes. A stop kills and reaps the direct
leaf child, including waits before stdout EOF and after both output pipes close.
Existing stdout/stderr capture and error reporting remain available. Hashing
checks the token between 1 MiB read/hash chunks and never returns a partial digest.
Cancellation is also checked between preparation phases and before publication.

This synchronous API belongs on a worker. File opens/OS reads and JSON parsing/
timing validation are not preemptible; cancellation is observed at their next
boundary. The measured latency is local Windows evidence, not a general deadline
for slow storage or very large metadata. Captured process cancellation is intended
for leaf tools such as ffprobe, not process trees launched through a shell. The
POSIX callback/reaping implementation was reviewed but not executed here; enforced
headroom leases still fail closed outside Windows.

## Validation

- Corrected build passes with no compiler warnings/errors. An initial new-test
  compile failure used a mutable-only buffer accessor on immutable pixels; it
  was replaced with a span comparison. The failed build log is retained.
- **36/36 focused tests pass** in 12.802 s.
- **336/336 native tests pass** in 53.473 s; zero failures/errors/disabled tests.
  Eight new tests cover shared lease lifetime/single probing, unchanged content
  identity, independent cancellation and receipt identity, latched shared
  invalidation, cancellation in both probe passes before/after EOF, cancelled/
  null input rejection, full timing/dimension rejection, hash interruption, and
  real FFV1 seeks matching sequential pixels. Existing analytic/colour/delivery
  goldens, budget, decoder late-failure and backpressure tests continue to pass.
- Controlled probe-stop latencies: **0.774–1.060 ms**; in-progress 64 MiB hash
  cancellation: **1.903 ms**. These are single-suite observations, not guarantees.
- Full results: `reports/headroom_preparation_native_20261016.{json,log}`;
  focused results: `reports/headroom_preparation_focused_20261016.{json,log}`.
  Frozen diagnostic executable and source fingerprints:
  `reports/headroom_preparation_20261016/`.
- Pre-existing ORT API 22/17 warning persists. No neural-runtime acceptance is
  claimed. App/Python/GPU tests and app/packages were not rebuilt/rerun here.

## Performance evidence

**12 completed cases pass; one timed-out case is retained separately.** The
combined evidence is `reports/headroom_preparation_20261016/comparison.json`,
reconciled by its neighbouring `summarize.py`. Initial runs are in `paired/`;
the UHD retry is in `uhd_retry_20261017/`. Native code and the frozen executable
were unchanged for the retry; only the runner gained a bounded timeout option.

The existing 240-frame HD/UHD fixtures are 10.01 s, 24000/1001 fps, offset-PTS,
10-bit limited Rec.709 FFV1. Three seeks create independent one-frame owners at
start/end/middle. `reopen` repeats full preparation; `reuse` shares it. Per-owner
pixel limits and all decoding/receipt checks are identical. HD has two pairs in
alternating order; UHD has one retry pair. Three full sequential controls match
**every frame hash from the previous frozen increment**, and all **18 seek frames**
match the relevant sequential frame. Content IDs also match the previous FFV1
candidate. Source hashes are unchanged after all runs. No external competitor,
rendering, encoding or display timing is claimed.

| Measurement | HD reopen → reuse | UHD reopen → reuse |
|---|---:|---:|
| Later preparation + owner creation | 10,961.34 → **0.050 ms** | 84,018.04 → **0.138 ms** |
| Later first frame, including preparation | 11.342 → **0.342 s** | 85.229 → **1.183 s** |
| Complete three-seek sequence | 32.410 → **11.314 s** | 246.651 → **95.486 s** |
| Sampled process-tree peak | 107.35 → 106.28 MiB | 429.85 → 470.80 MiB |

Later measurements aggregate end/middle observations; HD values are medians over
two pairs, UHD over its single pair. Real preparation cancellations after 100 ms
completed in **1.856, 3.001 and 4.337 ms**. Full initial preparation is still
expensive: the reused-source runs begin with **10.432 s HD / 92.535 s UHD**.
Reuse removes repeated scans/hashes; it does not accelerate the first scan or
change the decoder. Retained native pixels stay below 256 MiB for every seek.

The initial UHD three-reopen case exceeded its 180 s runner limit and produced
no accepted performance result. `initial_timeout.json` records that observation;
its log and the partial initial summary are preserved. The retry raised only
that per-case guard to 300 s and completed all four UHD cases. It did not relax
cancellation, pixel parity, lease, EOF/exit, identity or budget checks. Preparation
time drifted substantially between runs/days; a background CPU-load snapshot
before retry was 34%. The cause of the initial timeout is not established. Small
samples under varying load do not support a general speed guarantee. Memory is
sampled working-set sums, not a hard cap or unique physical RAM. No build/tests
ran alongside timed cases.

Reproduce after building `rudra_headroom_frames_bench` with pinned FFmpeg/ffprobe
on PATH; use fresh output directories. The original two-resolution run used
`--repeats 2` and the default 180 s case limit. The completed UHD retry was:

```powershell
.venv/Scripts/python.exe tools/benchmark_headroom_preparation.py `
  --binary reports/headroom_preparation_20261016/candidate.exe `
  --sources reports/headroom_memory_20261016/fixtures_240/uhd_240frames_ffv1.mkv `
  --out reports/headroom_preparation_uhd_repeat --repeats 1 --timeout 300
```

Final native/benchmark source fingerprints are in `source_hashes_final.json`;
`source_hashes.json` preserves the pre-retry runner fingerprint. Whitespace,
source/binary fingerprints and all result assertions pass. The original failed
test build, timeout and successful corrected evidence are retained.

## Remaining gates and next increment

1. An asynchronous playback adapter with explicit generation replacement,
   cancellation/join ordering, stale-result suppression and a **combined retained
   frame budget across old/new generations**. Current budgets remain per owner;
   shared metadata does not impose a global pixel or FFmpeg memory limit.
   Acceptance: rapid seek/cancel races cannot publish old frames or receipts;
   opening, read, completion and budget waits all cancel; borrowed frames from
   retired owners still consume the combined budget until released; late failures
   never promote provisional pixels. Compare paused/continuous/seek samples and
   analytic masters, then measure latency and memory before app wiring.
2. Long production footage and simultaneous consumer validation. Initial full
   preparation remains expensive; reuse only removes repeats for a live handle.
   Decoding itself is unchanged, and previous UHD feed remains below real time.
3. Session, undo, preview/scopes, provenance and transactional export integration
   before enabling headroom selection or loosening reader guards.
4. Existing commercial model/quality, dependency/codec rights, calibrated HDR
   review, signing, clean-machine and macOS release gates remain.

No training, model promotion, downloads, source dataset edits, commit or release.
