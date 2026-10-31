# RUDRA 1.0: what is left, and how to ship it

## 31 Oct 2026: the final release

The model and the AI fill v3 are done (Ahmed, 31 Oct). One script takes it from here,
on Windows, after `hf auth login` (a write token for fxtdstudios):

    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_V1_FINAL.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_V1_FINAL.ps1

It publishes `dist\hf\rudra-models`, and only it: the final model RUDRA ACES 2 (`models/RUDRA_ACES_2`,
the same files as `dist\models\RUDRA_ACES_2`, release_authorized) and the AI fill v3 (`models/rudra_genfill_v3`)
to huggingface.co/fxtdstudios/rudra-models under Apache 2.0, tagged v1.0.0; writes
`models.json` beside the user guide; sets the version to 1.0.0; rebuilds the guide; commits
and pushes `aces-v2`; and tags v1.0.0, which publishes the release (unsigned). Then deploy
the website so www.fxtdstudios.com/rudra-docs/ serves the 1.0 guide and models.json.

## The plan from 29 Oct 2026

This replaces the steps of 12 Oct below (kept for their history and commands).

**What 1.0 is.** The analytic reconstruction with the artist in the loop, as
re-scoped on 7 Oct, now in the Simple workspace (Look, Check, Deliver) with
first-run tips; Full keeps every control. Windows x64 only. The code is Apache
2.0. The learned ACES 2.0 model is an optional "preview" download, listed by
`models.json` on www.fxtdstudios.com and fetched from Hugging Face inside the
app; it is not release-qualified (0 EV grading check .121 against .0604) and
its weights licence waits on its training ancestry. The Beta 2 weights stay
non-commercial and are not bundled.

**Decisions** (D1 assumed as above until confirmed):

- D1 what 1.0 runs on; D2 the ACES 2.0 weights licence; D3 Windows only;
  D4 signed or unsigned.

**On the Windows machine and accounts**, in order:

1. Delete the leftover transfer files in `tmp`.
2. `scripts\COMMIT_PUSH_ACES_V2.ps1 -DryRun`, then without `-DryRun`; merge
   `aces-v2` into the release branch; CI green (Gate B's GPU parity included).
3. `scripts\RUN_APP_ACES.ps1` (build, app tests, start); `scripts\NATIVE_ACES_V2.ps1`
   (OCIO 2.5.2, parity, video smoke); `NATIVE_GATE_B.ps1 -Machine rtx4080s-win`
   and `NATIVE_PHASE3_EXIT.ps1`.
4. By hand on the HDR display: Simple, the tips over the GPU viewer, Hold B, the
   View menu, Open a .mov/.mp4, every Deliver preset, a model download, About.
5. Python: `== "1"` to `== 1` in `tests/test_delivery_2026_08_22.py:83` and
   `tests/test_ui_backend_2026_08_27.py:199`; `pytest tests/test_genfill_run.py
   tests/test_make_models_json.py` in the `.venv`.
6. Upload the model folder to Hugging Face `fxtdstudios/RUDRA` under `models/`,
   then `python tools/make_models_json.py --repo fxtdstudios/RUDRA --revision
   <tag or commit> --out models.json --model dist/models/<folder> --id ... --title
   "Learned ACES 2.0" --channel preview --kind aces_v2 --licence ...` and publish it
   at https://www.fxtdstudios.com/rudra/models.json.
7. Package: `scripts\PACKAGE_WINDOWS.ps1 -NoModels -WithAcesEngine` (add
   `-NoLibTorch` for a DirectML/CPU-only engine and a much smaller package). The
   script refuses a package whose engine cannot start on its own.
8. A clean PC that never had RUDRA or Python: install, open a clip, download the
   model, export HDR10 and ProRes, play them elsewhere, uninstall.
9. Sign (`RUDRA_SIGN_PFX`, `RUDRA_SIGN_PASSWORD`) or say "unsigned" in the notes.
10. `scripts\release_v1_rc.ps1 -Rc 3`, then `-Rc 3 -Tag`; a few days of use.
11. `scripts\release_v1_rc.ps1 -Final`, then `-Final -Tag`; paste
    `docs/RELEASE_NOTES_1.0.0.md` into the release; rebuild the guide for 1.0:
    `python tools/build_user_guide.py --version 1.0.0 --tag v1.0.0 --release
    --out ..\website\rudra-docs` and deploy it.

**Done when** CI is green, the clean-PC test passes, a new user can download
the model from inside the app, and `v1.0.0` is tagged with its notes.

**After 1.0:** the learned model passing its grading check; the Mac build and
Apple signing; the commercial-tools table (R3); OFX; Dolby Vision RPU; AI fill
out of experimental; audio in learned masters.

---

## The 12 Oct 2026 plan (history)

As of 12 Oct 2026. The release definition is STATUS.md's (re-scoped 7 Oct
2026): RUDRA 1.0 ships on the **analytic inverse with the artist in the loop**,
and the model is optional. Not part of 1.0 (later, by decision of 12 Oct): the
final model (STATUS line G), the OFX plug-in and a Dolby Vision RPU.

## The release rows

| # | condition | state |
|---|---|---|
| R1 | the inverse's own scores on every bench, published as the floor | **met** (STATUS) |
| R2 | every artist control under one frame at 4K; the scripted exit passing on Windows | **met**: 1.08 to 1.33 ms at 4K on D3D12, D3D11, Vulkan and OpenGL, gated by `latency_budgets.json` (7 Oct); phase 3 exit and the by-hand pass on an HDR display (9 Oct). Re-run on the release candidate (step 3) |
| R3 | the frozen set through every external tool, the table published whatever it says | **the open-source half is in** (below; the default changed to BT.1886 because of it); the commercial tools are run by hand (step 4) |
| R4 | macOS inference and HDR output, signing, the by-hand workflow on Windows, budgets per backend | **open**: the Mac pass (step 5), signing (step 6), the clean-PC install (step 7) |
| R5 | the optional model listed with its numbers against the inverse | **met**: `sdr2hdr_image_v8` (9 Oct; shadow_v1 retired) |

## Done on 12 Oct 2026

- R3 tooling: `training/compare_market.py` (prepare, builtin, import, score).
- R3 numbers for what runs without a licence: see below.
- HDR10+ in the shipped encoders: the gyan.dev 7.1.1 "full" ffmpeg the
  Windows package carries has x265 with HDR10+ (its HDR10+ parser is in the
  binary, absent from x265 builds without it); Homebrew's x265 builds its
  10-bit library with `-DENABLE_HDR10_PLUS=ON`. The export's own probe and QC
  still decide per machine (`hdr_metadata.hdr10plus.embedded`).
- `rudra batch` (Python) runs the app's queues: `grade` jobs go to
  `rudra-native video` (PATH or `RUDRA_NATIVE`).
- Temporal: v8's flicker on the real clips, and clip-mode CVVDP against the
  clips' HDR masters (docs/TEMPORAL_RESULTS.md, STATUS row 5).
- The release workflow publishes `v1.0.0` as a release and anything with a
  suffix (`-rc.1`, `-beta.10`) as a pre-release.
- `scripts/release_v1_rc.ps1`: version, push, tag, for each candidate and for 1.0.

## R3 so far (13 Oct 2026, docs/RESULTS.md)

Real SDR video against its graded HDR, paired against the ACES inverse (RUDRA's default until 13 Oct),
95 % intervals:

| method | Meridian PU21 / CVVDP | HDRTV1K test PU21 / CVVDP |
|---|---|---|
| **RUDRA, Source = BT.1886 (the default since 13 Oct)** | **+13.96 dB / +2.28 JOD** | **+4.16 dB / +2.11 JOD** |
| BT.2408 direct mapping | +14.24 / +2.29 | +4.31 / +2.04 |
| RUDRA, Source = Rec.709 | +1.70 / +1.33 | +0.89 / +0.59 |
| RUDRA + v8 | +0.04 / +0.01 (n.s.) | |
| RUDRA, ACES inverse | 21.65 dB / 6.443 JOD | 22.03 dB / 6.361 JOD |
| HDRTVDM (CVPR 2023) | -1.15 (n.s.) / -1.38 | (trained on it) |

**Decided 13 Oct 2026: option (c).** Reconstruct > Source > BT.1886 (code^2.4 of the BT.709 video's code,
white at 203 nits) is in the picker, scored as above, and the app's sessions start on it. A project or a
request that names no curve keeps the ACES inverse (`unknown`), so every saved project, CLI request and
golden reproduces. The rendered v4b frames still favour the ACES inverse, as they must (they were rendered
with ACES); the picker's ACES and filmic entries are for that kind of SDR.

The clip fill, also on by default, helps the rendered stills (+0.53 dB, 0.14 stops better where clipped) and
hurts the rendered clip frames (-3.09 dB, -0.22 JOD, 0.36 stops worse where clipped); the loss is all
on the four fireworks clips (-4.9 to -7.7 dB each; `bench/compare_v1/*/carousel_fireworks_*`). Look at those
frames before 1.0 ships with it on, and score it over BT.1886 (it was measured over the ACES inverse).

**Decided 13 Oct 2026: the clip fill ships on, unchanged (glow 0.5).** Its one knob was picked on
scenes 1-3 and, on the held-out scenes 4-7, the mean error inside clipped pixels over +0, +1 and +2 EV
is 0.412 stops against 0.578 for the flat inverse and 0.568 with no dome; glow 1 ties it (0.414). The
+0 EV loss (truth ~0.2 stop above the ceiling, any dome overshoots) cannot be told from one frame: v9,
trained for it, beat the fill there (+0.116 stops) and lost at +1 and +2 EV. Over BT.1886 the fill is the
same log2 residual on another baseline, and real Meridian SDR clips on 3e-5 % of pixels, so the R3 rows do
not move. The artist has Strength (scales the fill), Reconstruct > Fill clipped highlights (off) and a
peak per region. The fireworks loss goes in the notes; after 1.0, roadmap 4.2.

**Fixed 13 Oct 2026: the anchor follows the Source picker.** The master's anchor (on by default) targeted
the sRGB reading at 203 nits whatever the picker said, so every anchored BT.1886 master was the sRGB one
below the knee, and calibration and reference were undone there; R3 scored the unanchored inverse. The
target is now the picker's curve with its calibration or reference (`anchor_target_knots`), in the
master, the view and composite.frag; unknown and aces with neither are unchanged bit for bit
(tests/test_anchor_target.cpp, a parity view case).

## The steps, in order

1. **Push this work** (`.\scripts\release_v1_prep_2026-10-12.ps1`) and check
   that the native checks are green, Gate B's "GPU parity (GL)" and
   "GPU parity (Vulkan)" included (the playback hold's first build).
2. **Paste "Changes in beta 9"** from docs/BETA.md into the beta 9 release on
   GitHub (it still shows older notes).
3. **R2 on the RTX machine**, on the candidate's build:
   `.\scripts\NATIVE_GATE_B.ps1 -Machine rtx4080s-win` (latency budgets and
   parity on every backend) and `.\scripts\NATIVE_PHASE3_EXIT.ps1 -Frames <240 frames>`.
   While it is open: play a still shot with Hold steady on and look for
   shimmer, and note the fps the status bar reports with the model at 960 px
   (the 27 fps estimate is not measured).
4. **R3, the commercial tools**, by hand. First rebuild the bench tree on the
   RTX machine (minutes on the GPU; `bench/` is not tracked):
   `python training/compare_market.py prepare --out bench/compare_v1 --checkpoint checkpoints/sdr2hdr_image_v8.pt --map G:\corpus_v4b=G:\datasets\corpora\corpus_v4b --device cuda`,
   then `python training/compare_market.py builtin --out bench/compare_v1 --methods bt2408 --hdrtvdm <HDRTVDM clone> --hdrtvdm-weights params.pth params_3DM.pth params_DaVinci.pth --device cuda`.
   Each tool takes `bench/compare_v1/input709/png16`
   (BT.709, sRGB, full range; README.txt beside it says so) and writes HDR10
   (PQ, BT.2020) or linear EXR with the same file names. For each tool, write
   down the version and every setting, then
   `python training/compare_market.py import --out bench/compare_v1 --name <tool> --src <its output> --encoding pq2020 --note "<tool, version, settings>"`,
   and finally `python training/compare_market.py score --out bench/compare_v1`.
   Candidates: DaVinci Resolve (HDR conversion / inverse tone mapping),
   Topaz Video (SDR to HDR), NVIDIA RTX Video HDR (if it can write a file),
   a hardware converter's software LUT (AJA, Blackmagic) if available.
   Publish the table in docs/RESULTS.md whatever it says.
5. **The Mac** (docs/MACOS_SIGNING.md): build with `scripts/package_mac.sh`
   on Apple silicon with ffmpeg in the DMG (`MAC_FFMPEG_DIR`, a static build
   with zscale and x265), run Help > Check the display and the model on the
   XDR panel (Metal EDR peak) and the model on Core ML, and record both in STATUS.
6. **Signing**: a Windows code-signing certificate (`RUDRA_SIGN_PFX` locally,
   `WINDOWS_CERT_PFX` in the release workflow) and the Apple Developer ID
   (docs/MACOS_SIGNING.md). Without them 1.0 still installs, with a
   SmartScreen prompt and a Gatekeeper right-click.
7. **A clean PC**: install `RUDRA-<version>-windows-x64-setup.exe` on a
   Windows machine that has never had RUDRA or Python: open a clip, scrub,
   grade, export HDR10 and ProRes, play it in another player, uninstall.
8. **Release candidate**: `.\scripts\release_v1_rc.ps1`, then with the
   checks green `.\scripts\release_v1_rc.ps1 -Tag` (v1.0.0-rc.1, a
   pre-release). Use it for a few days; fix what it shows and repeat with `-Rc 2`.
9. **1.0**: `.\scripts\release_v1_rc.ps1 -Final`, then `-Final -Tag`.

Every commit credits FXTD Studios Research only; the scripts refuse to push
otherwise.
