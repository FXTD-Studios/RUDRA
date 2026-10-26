# RUDRA v1 ACES export (development candidate)

Exports SDR stills or video as **32-bit float scene-linear ACEScg OpenEXR** and,
optionally, a **lossless 16-bit HDR display file**. Ordinary pixels retain the
source appearance within the tested tolerance. Reconstructed highlights can
change colour substantially; this candidate is not yet a qualified grading master.

Status: development candidate that passed its registered checks on 24 Oct 2026
(`docs/production/ACES_V1_ANALYTIC_SHADOW_2026-10-24.md`). Not release-authorised.

## Run

From the repo root in PowerShell:

    .venv\Scripts\python.exe tools\rudra_aces_export.py <input> `
        --checkpoint G:\rudra_training\aces_v1_analytic_shadow_20261024\candidate.pt `
        --out <new folder> [--video] [--matrix bt709|bt601|auto] [--device cuda|cpu] [--max-frames N]

`<input>` is one still (png, jpg, tif, bmp, webp), a folder of stills (processed in name
order), or a video file. The output folder must not exist yet.

Research only: `--colour-backtrack` adds an explicit one-code appearance constraint
for partial-channel highlights and records its hash/state in the provenance. It
does not alter the default. The corrected equal-image head at
`G:/rudra_training/aces_consistent_head_equal600_v2_20261024/step600.pt` was tested
with this option. Its colour fidelity improves, but fully-clipped detail/grade
checks and a real Hunyuan motion transition fail. Do not treat this option or
checkpoint as a qualified replacement. See
`docs/production/ACES_REAL_EXPORT_DETAIL_FOLLOWUP_2026-10-24.md`.

`--colour-backtrack-all` extends that opt-in constraint to fully clipped highlights
and implies colour backtracking. This matters for research v8 checkpoints: their
unconstrained fully clipped output can also change SDR appearance. The report
records the chosen support. Backtracking is discrete and does not establish spatial
or temporal quality; every candidate still needs independent motion/detail checks.
The v8 context checkpoints remain research-only; see
`docs/production/ACES_CONTEXT_MOTION_2026-10-24.md`.

## What you get

| File | What it is |
|---|---|
| `exr/<name>.exr` | ACEScg linear, R/G/B **32-bit float**, AP1 chromaticities in the header, signed and unclamped. Use as the grading master (ACES/ACEScg working space). |
| channel `rudra.reconstructed` | 0–1 per pixel: where highlight radiance was **estimated** (source at the last 8-bit code). 0 means the value comes from the source, including crushed shadows. It is a support mask, not a confidence. |
| `display_hdr1000_pq_rgb16.mkv` (`--video`) | ACES 2.0 HDR 1000-nit Rec.2020 PQ, FFV1 `gbrp16le`, full range, BT.2020/PQ tags. Lossless 16-bit RGB, **display-referred**, a viewing/delivery derivative, not a grading master. |
| `report.json` | Checkpoint hash, contract, per-frame stats, EXR hashes and read-back check. |

Every EXR is read back and compared bit for bit before the next frame.

## How it behaves

- **Appearance:** through ACES 2.0 SDR 100-nit Rec.709 the output matches the source within one
  8-bit code outside reconstruction regions (`ordinary_within1_min` in the report).
- **Banding:** a learned restorer adjusts each ordinary pixel by at most 0.49 of one code
  (sub-code dequantisation), so it cannot change the look.
- **Highlights:** where the source hits code 255, radiance is estimated. That detail is invented
  and is marked in `rudra.reconstructed`. Saturated colours also trigger this support;
  the current predictor can introduce visible colour shifts there.
- **Shadows:** fully supported crushed shadows use the exact ACES inverse. At the support
  boundary it blends with the inverse of the bounded restored source. Tests check a
  one-code lift tolerance and newly negative luminance; this does not recover detail lost to black.

## Limits

- Not camera RAW. 32-bit float gives numerical headroom for grading; it does not restore information
  the 8-bit source never had.
- Video is decoded to 8-bit RGB with a BT.709 matrix (`--matrix` to change). 10-bit sources are
  reduced to 8 bits in v1.
- Full-frame inference with no tiling. 4K memory use and throughput are not validated.
- The export uses four CPU workers for the exact OCIO transforms (`--ocio-threads 1`
  selects serial processing). Signed HDR and odd-sized-buffer tests verified bit-exact parity.
- Frames are processed independently. The 18-clip generated review processed 826 frames,
  capped at 48 per clip; worst transition was 0.01989 EV, just below its 0.02 EV limit.
  This does not establish long-clip stability.
- Grade ramps still show small steps and ripples under ±3 EV pushes. Highlight detail is
  estimated, and measured as no better than the analytic conversion on detail audits.
- Ordinary-only appearance checks exclude reconstructed highlights. A separate generated-clip
  audit found large colour errors in those regions, including a 105-code sample error.
  See `docs/production/ACES_GENERATED_COLOUR_REVIEW_2026-10-24.md`.
- The OCIO version is part of the model contract. Run with the OCIO version recorded in the
  checkpoint (2.5.2); a different version is refused rather than silently changing the transform.
- The native desktop app does not run this model yet.
