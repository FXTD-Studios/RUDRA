# Continuous headroom decoder: isolated prototype

## Decision and scope

The verified batch feeder waits for EOF and a successful decoder exit before
publishing each batch. Its 256 MiB budget holds only two UHD float frames, and
consumer retention led to 23 decoder starts for the 24-frame benchmark. Removing
the terminal checks would allow a late failure to masquerade as successful data.

Added `HeadroomStream`, a synchronous, bounded pull decoder with one immutable
range per owner. It starts FFmpeg once and reads sequentially. It uses the same
float conversion, guarded range selection, source/decoder content identity and
Windows read leases as the batch feeder. The lease implementation was moved into
a private shared header without changing its behavior. Existing batch semantics,
colour science, production readers and UI defaults are unchanged.

This is a CPU prototype, not app integration or production playback acceptance.
Call `next()` and `finish()` on a worker thread. A new seek/retry requires a new
owner; no old-generation frame or receipt is accepted by that owner, even when
both owners have the same content cache ID. This intentionally avoids adding a
second asynchronous cache/state machine before measuring the simpler design.

## Trust and completion contract

| Data or operation | Meaning | Required consumer behavior |
| --- | --- | --- |
| `HeadroomPreviewFrame` | Exact complete frame with finite samples and the source contract; terminal decoder success remains unknown | May preview or stage temporarily. Validate with the originating live owner before presentation/use. No implicit conversion to the verified frame type. |
| `finish()` / `HeadroomDecodeReceipt` | Every expected frame was returned, exact EOF reached, child exited successfully, source/decoder identity still matches and no cancellation was observed | Receipt belongs only to this owner/range. Revalidate before cache reuse or accepting decode completion. |
| `verified_frame(frame, receipt)` | Explicit promotion of retained pixels after successful terminal validation | Check completion again on later reuse. This prototype has no persistent cache. |
| Late failure, identity change or cancellation | Terminal invalidation, including previously returned previews and receipts | Stop presentation; discard provisional cache/staging. No retry in the failed owner. |
| Export publication | **Not implemented by this receipt** | An export transaction must additionally account for all rendered frames, finish/validate its encoder, validate the decode receipt and publish its temporary output only on success. |

Reading the expected number of frames is insufficient for completion. Tests
include a producer returning every finite frame and then exiting with an error,
as well as extra trailing output, truncation, nonfinite output and waits both
before EOF and after stdout closes. Successful preview bytes cannot bypass these
checks. A decode receipt does not certify rendering, metadata or encoding.

Already-returned shared memory cannot physically be revoked. Pixels remain safe
to read after owner destruction but cannot then be validated for reuse. A future
UI/export adapter must observe errors and stop using invalidated data; this API
does not promise automatic screen clearing or rollback of a published file.
Validation is a point-in-time check, not an atomic filesystem export transaction.

## Resource and concurrency contract

- Pixel allocation includes the in-progress decoder frame and all live consumer
  handles. `Busy` under memory pressure consumes no input and starts no new
  decoder; releasing a handle permits the next read. No hidden frame copy/cache.
- Budget is per owner. FFmpeg/probe memory, metadata, rendering, copies explicitly
  made by callers and other owners are excluded, as for the existing feeder.
- `start`, `next` and `finish` serialize; concurrent decoding calls return `Busy`.
  `cancel` and token validation may run concurrently with a blocked read/finish.
  Cancellation is terminal and interrupts the child; the first error stays latched.
- Join callers before destroying the owner. Source probing/hashing in `open`
  remains synchronous and not cancellable. Windows supplies enforced read leases;
  other platforms continue to fail closed.

## Validation and measurements

**326/326 native tests pass**, 42.776 seconds, with no failures, errors, disabled
or skipped tests. The 22 focused source/lifecycle tests also pass. Nine new tests
cover continuous real-range decode and +1 EV grade parity, bounded retention,
late failures, cross-owner token rejection, concurrent cancellation, terminal
waits, source/decoder leases, metadata invalidation and invalid inputs. Existing
batch tests and analytic/colour/delivery goldens still pass.

Evidence: `reports/headroom_stream_native_20261016.{json,log}` and
`headroom_stream_focused_20261016.{json,log}`. The first build found a const-buffer
access error in a new test; the test now compares immutable spans. Initial and
corrected build logs are retained; the corrected build has no compiler warnings
or errors. Tracked and changed-source whitespace checks pass.

The pre-existing ONNX Runtime API 22/17 test-loader warning persists. These
results do not establish neural/backend readiness. App, Python and GPU suites
were not rerun. The application and release packages were not rebuilt; the CLI
test dependency was rebuilt. No dataset changes, training, downloads, commit,
model promotion or release occurred.

**42 benchmark runs passed**, three repetitions on each existing 24-frame
1920x1080 and 3840x2160 synthetic FFV1 fixture (10-bit YUV444, 24000/1001 fps,
five-second timestamp offset). Baseline is the previous increment's frozen
optimized batch binary, not the earlier unoptimized feeder. Both binaries run
feeder/sequential/legacy controls; only the candidate has the new stream mode.
Every float frame hash matches across all variants and repetitions. All legacy
control hashes also match. Protected-source cache IDs match, while stream tokens
remain owner-specific. Every stream run validates its completion receipt.

Median feed times, including consumer hashing and final completion but excluding
source preparation, rendering, inference, display and encoding:

| Measurement | HD | UHD / 4K |
| --- | ---: | ---: |
| Frozen batch feeder | 3.501 s | 22.741 s |
| Current batch control | 3.463 s | 22.684 s |
| Continuous prototype | 0.585 s | 2.155 s |
| Current direct sequential reference | 0.560 s | 2.151 s |
| Continuous speedup over frozen batch | 5.99x | 10.55x |
| Continuous feed rate | 41.05 fps | 11.14 fps |
| Decoder starts, batch to continuous | 12 to 1 | 23 to 1 |

Stream feed ranges are 568.1–634.4 ms HD and 2141.0–2175.4 ms UHD. These are local
diagnostics with three repetitions, not production throughput guarantees. The
unchanged batch/legacy controls help reveal run-to-run variation. No external
competitor or new HDR quality comparison is claimed for this decode-only change.
UHD remains below real-time throughput even before rendering or display.

Source preparation still costs a median **1.418 s HD / 4.123 s UHD** for the
stream. Comparing medians of preparation plus feed reduces the speedup to
**2.46x HD / 4.26x UHD**. First frame after preparation improves from 204 to 140 ms
HD and 671 to 444 ms UHD. Full-source probing/hashing is not accelerated here.

Memory is a material tradeoff:

| Median sampled process-tree working-set sum | Frozen batch | Continuous |
| --- | ---: | ---: |
| HD | 405.5 MiB | 323.4 MiB |
| UHD | 760.6 MiB | 1234.7 MiB |

UHD total sampled memory increases **62%**, close to the direct sequential path's
1243.7 MiB. This does not identify which FFmpeg stage holds those allocations;
that requires profiling. It is not a memory improvement at UHD. The native
owned-pixel peaks are separately bounded: exactly **24,883,200 bytes HD** and
**99,532,800 bytes UHD**, one frame in these streaming benchmark consumers, within
the unchanged 256 MiB budget. Native tests also exercise retained-frame pressure.
Sampled tree sums can double-count shared pages and miss peaks between samples.

Evidence in `reports/headroom_stream_20261016/`:

- `paired/summary.json` and individual reports/logs: all 42 measurements,
  executable/FFmpeg/fixture hashes and per-frame hashes.
- `comparison.json` and `summarize.py`: reproducible medians/ranges, speedups,
  legacy parity, shared content identity, receipt and pixel-budget checks.
- `batch_baseline.exe`, `stream_candidate.exe`: frozen comparison binaries.
- `source_hashes.json`: fingerprints of relevant implementation/test/benchmark
  inputs. Machine details are in the preceding increment's
  `reports/headroom_perf_20261016/hardware.json` (Windows 11, i9-14900KF, 128 GiB).

Reproduce with pinned FFmpeg and the required native DLLs on PATH, using a new
output directory (the runner refuses an existing one):

```powershell
.venv/Scripts/python.exe tools/benchmark_headroom_frames.py --binary reports/headroom_stream_20261016/stream_candidate.exe --compare-binary reports/headroom_stream_20261016/batch_baseline.exe --fixtures reports/headroom_perf_20261016/fixtures_tagged --out reports/headroom_stream_recheck --repeats 3 --stream
```

Short synthetic clips do not establish long-footage, additional-codec,
Unicode/network-path, portable-platform or production export acceptance.

## Remaining integration work

Next profile and bound the external decoder's memory and check longer sequences;
the frame budget does not bound FFmpeg's own buffering. Then implement a shared
protected-source owner and cancellation for source preparation, followed by an
isolated asynchronous playback adapter with generation
changes and a combined allocation budget across retained old/new frames. Reuse
the verified batch path for paused reads where appropriate; avoid repeatedly
probing/hashing the entire source on each seek. Stress cancellation, scrubbing,
completion failure propagation and longer clips before wiring app consumers.

Session/undo, preview, scopes, export transactions and provenance must all support
the policy before exposing it. Keep the default rejection of version-2 headroom
projects and leave headroom selection disabled. No new model/training approval is inferred.
Commercial rights, clean-machine acceptance, calibrated HDR review, signing,
portable-platform validation and production quality gates remain unresolved.
