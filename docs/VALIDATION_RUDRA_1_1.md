# RUDRA 1.1 validation (priority 6)

The harness: `tools/validate_rudra_1_1.py`. It renders every shot through `rudra-native`
(analytic, float32 EXR, so half rounding never hides a difference), delivers HDR10 and HLG,
reads everything back and reports PASS / FAIL / INFO / MANUAL per check. Thresholds are in
the script's docstring; results in `<out>/report.md` and `report.json`.

## Run it

```powershell
# 1. Build rudra-native with the still decoder (as for the app), then:
$env:RUDRA_NATIVE = "native\build\<preset>\cli\rudra-native.exe"
.venv\Scripts\python.exe tools\validate_rudra_1_1.py --selftest --out validation\selftest
.venv\Scripts\python.exe tools\validate_rudra_1_1.py --manifest docs\validation\shots.json --out validation\2026-11
# with the AI fill (16 GB GPU):
.venv\Scripts\python.exe tools\validate_rudra_1_1.py --manifest docs\validation\shots.json --out validation\2026-11-fill `
    --genfill --base <Wan2.1-VACE-1.3B-diffusers> --lora <rudra_genfill.safetensors> --prompt "warm sunset sun with a soft halo"
```

Needs `ffmpeg` / `ffprobe` with libx265 on the PATH, and `OpenEXR`, `numpy`, `Pillow` in the
Python (the repository's `.venv` has them; `PyOpenColorIO` for `--genfill`; `open_clip_torch`
optional, for a CLIP score on prompt adherence). `rudra-native master analytic <image> ...`
(new in this change) runs the analytic master with no model package.

## The shots to collect

Copy `docs/validation/shots.example.json` to `shots.json` and point it at real plates; 8-bit
or 10-bit, movies preferred (the first 24 frames are used, `--frames N` for more). At least
two per category, and at least half of them **moving** (pan, handheld or subject motion):

| Category | What it must contain | What it stresses |
|---|---|---|
| sky | a blue sky with only the blue channel clipped; an overcast white sky | the region's highlight ramp; the clip fill's partial-channel stage (see finding F1) |
| sunset | the sun disc, a backlit haze | small fully clipped sources, the gradient around them |
| window | an interior with a blown window, one panned | hard region edges against dark detail; temporal stability |
| reflection | car paint / chrome speculars, water glints | the band region (0.70 to 1.0), tiny highlights |
| fire | a campfire, candles handheld | moving, flickering highlights: C5 and the AI fill fallback |
| other | a night street (lamps + crushed shadows) | highlights and shadows together |

## What each check proves

C1 and C2 are the RUDRA 1.1 contracts (pixels outside the region bit-identical; the mattes
mark every changed pixel). C3 is preview = export for the master itself. C4 is INFO: with the
default master chain the anchor's frame-wide hold, chroma carry and grain settling reach
pixels outside the region by a little. C5 is temporal. C6 is the HDR10 / HLG file: the codec
and transfer (fidelity) and, separately, how far the deliverable is from what the viewer
previewed. C7 is the AI fill (contracts PASS / FAIL; prompt adherence MANUAL from the contact
sheet, plus a CLIP score when available).

## Selftest, 5 Nov 2026 (cloud container, synthetic shots: the harness, not footage)

`docs/validation/selftest_report_2026-11-05.md`. C1, C2, C3, C5 PASS on every shot; C4 at most
0.005 stop; C6 PASS on sunset, window, reflection and the moving fire (neutral-pixel error
0.0002 to 0.014 stop median). C7 not run (no fill weights in the container).

**Finding F1: decided and fixed (5 Nov 2026).** The synthetic sky has the blue channel at 255
everywhere and red / green below it. Measured: the clip fill's partial-channel stage is not the
cause (`max_partial_ev` 0, 2 or 4 give the same fill); a blue code of 255 decodes to the analytic
ceiling, ~1,150 nits of blue at ~140 to 250 nits of luminance, on every partly clipped pixel.
Delivered at 1,000 nits, every pixel with a channel over the peak was scaled down whole, and
HLG's scene-light step did the same more strongly: the sky's SDR range came out **0.16 stop
darker in HDR10 and 0.70 stop darker in HLG** than the master.

The delivery now keeps such a pixel's luminance and gives up saturation instead
(`PeakFit::KeepLuminance`, the default of `rudra-native deliver` and the app's movie export;
`--peak-fit scale` is the Python's). Selftest after the fix: **-0.009 stop (HDR10) and -0.016
stop (HLG)**, everything else unchanged. The sky's C6 fidelity check still FAILs (0.025 / 0.037
stop median on neutral pixels next to the saturated sky, delivered at 6,400 nits where no fit
happens): 4:2:0 Y'CbCr is not constant luminance, a property of the HDR10 / HLG formats rather
than of RUDRA. Run the real sky shots before release.
