# Visual review and packaging preflight — 16 October 2026

## Visual review: awaiting owner assessment

Opened the existing local review movie in the default player. Verified it with
the pinned FFmpeg 7.1.1 ffprobe: HEVC, 1920 × 360, 10-bit YUV420, PQ transfer,
BT.2020 primaries/matrix, duration 31.958 seconds. Correct file tags do not
prove the player's HDR presentation or display calibration.

File: `reports/shadow_review_20261016/shadow_review_hdr10.mp4`.
Panel order: reference HDR | default BT.1886 | optional BT.1886 shadows.
The first four shots (0–16 seconds) are the dark Sparks risk cases. Compare
black level, shadow detail, banding and tint against the reference, then
confirm the setting in RUDRA itself. The +4 EV SDR preview PNGs are not for
grading. Primary PA279CRV HDR output was verified in the preceding probes;
calibration status and the owner's visual judgment are not established.

Record before sign-off:

- Display/preset and calibration date or method: pending.
- Player and confirmation of HDR output: pending.
- Dark shots: better/same/worse, black lift/tint/banding notes: pending.
- Remaining shots: better/same/worse and notes: pending.
- Native-app comparison: pending.

No visual pass is claimed. Keep the shadows curve opt-in and BT.1886 default.

## Packaging preflight

- Pinned FFmpeg 7.1.1 available, hash verified, full native/app tests passed.
- Inno Setup is present at the per-user installation path. No install needed.
- `RUDRA_SIGN_PFX` is not configured in this session. This does not prove the
  owner lacks a certificate; signing is simply not ready here.
- Current `scripts/PACKAGE_WINDOWS.ps1` requires a model manifest, copies
  `dist/models` and checks a model after packaging. The local package is v8;
  its weights are explicitly non-commercial. Do not run this packaging path
  as a commercially cleared build.
- The script also removes existing versioned output folders/ZIPs. It was not
  executed: existing artifacts remain untouched.

Next engineering task: add an explicitly model-free packaging mode with
isolated output and appropriate startup checks; preserve the existing research
beta path. Validate the staged app with a restricted PATH before installer
testing. Removing weights alone does not clear the shadow-table rights,
Netflix notice conflicts, code/commercial licence, FFmpeg obligations or
other questions in `docs/LEGAL_QUESTIONS.md`.

Clean-machine installation, signing, macOS validation and remaining external
comparisons are still open. No release, deployment, signing, training or
dataset writes were performed during this preflight.
