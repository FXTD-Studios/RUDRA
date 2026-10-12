# RUDRA 1.0: what is left, and how to ship it

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
