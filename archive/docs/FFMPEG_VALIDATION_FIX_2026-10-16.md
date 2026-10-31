# FFmpeg validation fixes — 16 October 2026

All four failures from the earlier Windows run are resolved on the installed FFmpeg build. Rebuilt RUDRA, the CLI and test binaries in `build/native_phase3` (Release, OpenCV and ONNX enabled).

- Focused regression: **9/9 passed**.
- Complete native suite: **264/264 passed**, 25.743 s.
- Complete offscreen application suite: **57/57 passed**, 17.022 s.

## Confirmed causes and changes

The installed January 2025 Gyan FFmpeg build produced synthetic FFV1 and H.264 inputs with a BT.709 matrix but no transfer/primaries, despite explicit output options. A two-frame FFV1 reproduction showed that setting the frame metadata with `setparams` preserves all four requested tags. Added those frame tags to the built-in FFmpeg self-test and the two temporal-test generators. Production source validation and reconstruction are unchanged. Both temporal tests now reach and pass their original flicker thresholds.

The alpha golden was generated with Ubuntu FFmpeg 6.1.1 and recorded zero error. On the installed decoder, both the independent Python `check_alpha` reference and native implementation report exactly **15 codes**, below the existing **128-code** limit. The test retains exact comparison on the golden build; on other builds it checks the original numeric bound and all remaining report fields. The production tolerance and alpha code are unchanged.

## Evidence

`reports/ffmpeg_diagnosis_20261016/` contains the input reproductions, independent Python alpha result, FFmpeg/ffprobe SHA-256 hashes, build log, focused test JSON/log and full native/app JSON/log. Earlier failed results remain preserved separately.

Validated FFmpeg: `2025-01-05-git-19c95ecbff-full_build-www.gyan.dev`. The Windows package script pins **7.1.1**; this turn did not download or validate that release dependency. Passing this installed-build suite is not package certification.

Next: run release-candidate GPU parity and latency gates, then validate with the packaged dependency versions. Calibrated HDR shadow review, commercial licensing, signing and clean-machine/platform checks remain open. No training, dataset changes, commit, push or deployment occurred.
