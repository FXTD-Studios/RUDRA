# Third-party components and licences (release record)

Drafted 25 Oct 2026 for the ACES v2 release. **Status: draft, to verify against the exact files shipped.** Each row must be checked against the LICENSE file inside the shipped binary package before release, and NOTICE updated. This is a working record, not legal advice.

## Shipped with the desktop app or CLI (Windows package)

| Component | Version (pin) | How it ships | Licence | What must go with it |
|---|---|---|---|---|
| OpenColorIO | **2.5.2** exact (checkpoint contract) | DLL beside the exe | BSD-3-Clause | Copyright notice and licence text |
| OCIO built-in config `cg-config-v4.0.0_aces-v2.0_ocio-v2.5` | compiled into OCIO 2.5.2 | inside OpenColorIO DLL | BSD-3-Clause (OpenColorIO-Config-ACES) | Covered by the OCIO notice; name the config in NOTICE |
| OCIO dependencies: yaml-cpp, expat, Imath, pystring, minizip-ng, zlib | as built by vcpkg/OCIO | DLLs or static inside OCIO | MIT, MIT, BSD-3, BSD-3, zlib, zlib | One notice each. List what is actually in `bin\` |
| OpenEXR (if linked) | per vcpkg | DLL | BSD-3-Clause | Notice and licence text |
| LibTorch (from the pip torch used to build) | torch 2.x, record exact | DLLs | BSD-3-Clause style (PyTorch) plus bundled third-party notices | torch `LICENSE` and `NOTICE` |
| CUDA runtime, cuDNN, cuBLAS (inside CUDA LibTorch) | record exact | DLLs | NVIDIA redistributable terms (CUDA EULA attachment A; cuDNN SLA) | Ship only redistributable files; keep the NVIDIA notices |
| ONNX Runtime (DirectML build) | 1.22.0 | DLL | MIT | Licence and ThirdPartyNotices.txt |
| DirectML | per ORT nuspec (≈1.15.x) | DLL | Microsoft DirectML redistributable licence | Licence file from the NuGet package |
| OpenCV | 4.10.0 | `opencv_world` DLL | Apache-2.0 | Licence and NOTICE |
| Qt 6 (app only) | 6.8.3 | DLLs, dynamically linked | LGPL-3.0 (or commercial) | LGPL text; users must be able to relink or replace the Qt DLLs; offer the Qt source or say where it is |
| nlohmann/json | per vcpkg | header-only | MIT | Notice |
| FFmpeg (ffmpeg.exe, ffprobe.exe) | 7.1.1 gyan.dev "full", SHA-256 pinned | separate programs | **GPL-3.0** | Already handled: `ffmpeg\LICENSE.txt`, `ffmpeg\SOURCE.txt`, NOTICE section. Run as a separate process, never linked |

Not shipped: GoogleTest (tests only, BSD-3), Python and its packages (research tools only).

## Codecs and formats

- **FFV1 in MKV** (16-bit PQ master video): FFV1 is an open codec (IETF RFC 9043) and Matroska is an open container. Encoding goes through the bundled GPL ffmpeg.exe as a separate process. Record the exact ffmpeg build in each master's provenance.
- **x265, ProRes (prores_ks)** for review movies come from the same ffmpeg build. HEVC encoding may carry **patent-pool obligations** for commercial distribution, separate from copyright licences. Get a decision from counsel before a commercial release that encodes HEVC.
- **OpenEXR** files: no licence obligations on outputs.

## Model weights (ACES v2)

- Fine-tune rows used by `tools/train_partial_display.py` (25 Oct 2026): 63 rows from `aces_consistent_exposures_v2_20261024/train` and 288 from `aces_temporal_context_pairs_20261024/train`. **All 351 are CC0 with `commercial_ok = true`** (read from the corpus reports).
- The retained sampler-v2 declared-data audit covers 527 rows: 399 CC0 receipt matches and 128 original procedural declarations, with all 527 content-hash checks passing and no unresolved declared rows. Evidence: `G:/rudra_training/aces_context_sampler_v2_6000_20261024/declared_data_receipt_audit_527_20261024.json`. This records the existing audit; it is not a new download/source verification or a blanket weights-distribution clearance.
- Initialization reproducibility remains incomplete. The guarded600 provenance archive traces seven checkpoints and resolves ten recorded code hashes. Its root run declares scratch initialization, but four of its five recorded source hashes differ from current files; the archive reports no retained matching versions after its earlier search. An exact historical replay is therefore not currently available. A new fully archived scratch run would establish a new lineage, not retroactively prove the old run. Evidence: `G:/rudra_training/provenance_guarded_projected600_20261025/README.md` and the root run's `recipe.json`.
- The existing NOTICE's non-commercial statement about shipped v8 weights must not automatically be applied to this distinct ACES v2 lineage. No restricted initializer has been established by the reproducibility gap. The ACES v2 weights still need their own release determination; no licence selection or distribution authorization is implied here.
- Outputs are marked as reconstructions (`rudra.reconstructed` mask, provenance "not camera RAW").

## Release checklist (signing is last, by decision)

1. Build the release package with pinned versions and record every DLL in `bin\` with its SHA-256.
2. For each DLL, copy the matching licence file into `licenses\<component>\`.
3. Add NOTICE sections for OpenColorIO (and the ACES config), LibTorch, NVIDIA CUDA, ONNX Runtime and DirectML, OpenCV and Qt (app).
4. Confirm that the Qt LGPL relink requirement is met (dynamic linking, Qt DLLs replaceable).
5. Get a decision on HEVC patent licensing for commercial use.
6. Get a decision on the ACES v2 weights licence after the ancestry confirmation.
7. **Sign last:** code signing of the exe, DLLs and installer once everything above is final.
