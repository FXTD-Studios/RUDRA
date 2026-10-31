# Windows validation — 16 October 2026

Validated local `5efc693` plus the test-only changes below. No training, dataset writes, commit, push or release deployment.

## Results

| Configuration | Build | Native tests | App tests |
| --- | --- | --- | --- |
| `build/native_gate_b`, Release, OpenCV/ONNX off | Passed | 249/249 passed | 41/41 passed |
| `build/native_phase3`, Release, OpenCV/ONNX on | Passed: RUDRA, CLI and test targets | 260/264 passed; 4 failures | 57/57 passed |

The minimal app run preceded the additional full-build export assertions. The full app run includes them. Existing numeric-conversion compiler warnings remain. These offscreen app tests do not establish display accuracy, GPU backend parity or 4K latency.

## Fix and stronger coverage

The new folder-export test referenced helpers available only with `RUDRA_HAVE_STILL_DECODE`, breaking minimal-build compilation. Added the same conditional guard used by the other image-decoding tests.

Strengthened the folder-to-movie test to probe the actual HDR10 output: four frames, 25 fps, BT.2020 primaries/matrix and PQ transfer. It passed, along with EXR intermediate creation and refusal to overwrite an existing export. Existing sequence tests passed for encoding/tagging, measured-light reporting and cancellation. Still-metadata, image decoding/orientation and master-provenance tests passed in the full configuration.

## Four unresolved native failures

1. `VideoQc.AlphaCheckIsThePythons`: expected golden maximum alpha error 0, actual 15 in 16-bit codes. Both verdicts pass the 128-code tolerance, but the exact JSON assertion fails. Do not simply loosen the test; establish the decoder/build cause first.
2. `FfmpegCheck.SelfTestPassesAndIsCachedPerBuild`: self-test fails with missing/unsupported color transfer, so the expected successful cache also fails.
3. `VideoGraded.TheShotHoldSteadiesTheAnchor`: generated input is rejected for missing/unsupported color transfer.
4. `VideoGraded.TheStaticHoldSteadiesAStillShot`: same input-contract rejection.

Independent ffprobe inspection of both temporal-test generated inputs shows `color_space=bt709` but no transfer or primaries, despite the test commands requesting those tags. This explains their input rejection; it does not yet establish why the encoder omitted them. Active FFmpeg: `C:\Program Files\ffmpeg-2025\bin\ffmpeg.exe`, version `2025-01-05-git-19c95ecbff-full_build-www.gyan.dev`. Do not bypass the production input contract to make tests pass.

## Evidence and next action

Saved logs and GoogleTest JSON under `reports/`:

- `windows_native_20261016.{log,json}`
- `windows_app_20261016.{log,json}`
- `windows_full_native_20261016.{log,json}`
- `windows_full_app_20261016.{log,json}`

Next: reproduce the generated-input tagging issue and alpha difference with the intended release FFmpeg build, then fix the proven cause and rerun affected tests. Keep these failures visible until resolved. Calibrated shadow review, GPU performance/parity, packaging/platform checks and commercial licensing are still pending. The latest binary is `build/native_phase3/app/Release/RUDRA.exe`; it is built, not release-cleared or deployed.
