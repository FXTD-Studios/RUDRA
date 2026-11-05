# RUDRA 1.0.2

**SDR in, HDR out — in three steps.** RUDRA turns SDR stills and footage into
scene-referred HDR: Look, Check, Deliver. It works out of the box with its
analytic reconstruction; the learned ACES 2.0 model, a free download inside the
app, rebuilds more of the light the SDR clipped. Every master says what is in it.

Windows 10 / 11 (x64) and macOS (Apple silicon). Free for commercial use: the code
and the 1.0 models are Apache 2.0.

## Download

| File | For |
|---|---|
| `RUDRA-1.0.2-windows-x64-setup.exe` | the installer (per user, Start menu, uninstaller) |
| `RUDRA-1.0.2-windows-x64.zip` | the portable app: extract the whole folder, run `RUDRA.exe` |
| `RUDRA-1.0.2-macos-arm64.dmg` | macOS on Apple silicon: open it and drag RUDRA to Applications |
| `*.sha256` | the SHA-256 of each file |

The builds are not code-signed: on Windows choose More info › Run anyway in the
SmartScreen prompt; on macOS, Open Anyway in System Settings › Privacy & Security.

User guide: https://www.fxtdstudios.com/rudra-docs/
Models: https://huggingface.co/fxtdstudios/rudra-models (tag v1.0.0)

## New in 1.0.2

- **The learned ACES 2.0 model on every platform.** The packages now carry its
  engine (OpenColorIO 2.5.2 and ONNX Runtime) in `rudra-native`, so the model you
  download in the app runs straight away: on Windows on the GPU through DirectML,
  else the CPU; on a Mac on the CPU, or the GPU / Neural Engine through Core ML
  (chosen in the Learned ACES 2.0 master window). The 1.0.1 packages did not include
  the engine, so the downloaded model could not be used there.
- **AI fill on a Mac.** The fill runs on Apple silicon's GPU through Metal (PyTorch
  MPS, float32). Set it up with `scripts/genfill_setup.sh`; it needs 16 GB of
  memory at least (32 GB is comfortable) and stays experimental, as on Windows.

## Fixed in 1.0.1

- **Wipe.** The seam stays where you leave it when you change the display peak or
  exposure, and the hint says which side is which: SDR left, RUDRA right.
- **Clipped highlights.** A large blown area (a sun in a hazy sky) runs toward white
  at its centre instead of carrying the sky's colour all the way in; measured on
  the clip benches, colour closer to the HDR reference, brightness unchanged.
- **macOS** builds again (Apple's C++ library has no `std::stop_token`).
- The macOS bundle names the Apache 2.0 licence.

## New in 1.0 (since Beta 2)

- **Simple workspace.** New users start in Simple: **1 Look** (monitor and trims),
  **2 Check** (peak, average, light above SDR white, out-of-P3 share, the scopes
  beside the picture) and **3 Deliver** (Grading master, HDR10 movie, Broadcast HDR
  or Custom…). Hold **B** for the original SDR; the **View** menu has the wipe, what
  changed, the invented pixels and false colour. **Full** keeps every control.
- **First-run tips.** A welcome card, then four tips the first time a shot is open;
  Help › Show tips brings them back.
- **Learned ACES 2.0 model.** Downloaded inside the app (File › Model packages…
  or Look › Download the model…), every file checked against its SHA-256,
  resumable. Kept loaded on the GPU and used by the viewer and every render.
- **Delivery.** ACES 2065-1 EXR (ST 2065-4, half) and ACEScg EXR; DPX 10-bit with
  full headers; ProRes 422 HQ / 4444 / 4444 XQ; HEVC 10-bit HDR10 with MaxCLL /
  MaxFALL and mastering-display metadata; HLG; PQ 1000 / 2000 / 4000; SDR Rec.709;
  ARRI LogC4 and ACEScct copies; Dolby Vision XML for PQ renders.
- **For the colourist.** Per-shot looks with Copy / Paste; a batch queue; EDL
  conform with handles and retimes; timecode and reel in every render; reference
  stills with wipe; RGB parade and gamut; the monitor through the ACES 2.0
  SDR 100-nit or HDR 1000-nit output.
- **Safer renders.** Files are written as `.part` and renamed when complete;
  nothing is overwritten; quitting asks while a render runs.
- **Open…** takes movies (MOV, MP4, MXF, MKV and more) as well as stills.
- **About** shows the version, the model and the GPU, with links to
  www.fxtdstudios.com, the guide, the licences and issue reporting.
- **AI fill v3 (experimental, off by default)** for large blown regions:
  Wan2.1-VACE-1.3B with the FXTD LoRA draws the shape, a small estimator sets the
  brightness, and Highlight brightness (±3 stops) lets you move it. It passed its
  registered gates on 58 held-out clips. Downloadable in Model packages; the base
  model comes from Wan-AI. Review every frame before it reaches a client.

## Licence

The code and the 1.0 models (learned ACES 2.0, AI fill v3) are licensed under
the **Apache License 2.0**: commercial use is allowed. Each model shows its
licence in Model packages before a download. The Beta 2 weights
(`sdr2hdr_image_v8`, `sdr2hdr_shadow_v1`) remain non-commercial because of a
training source's terms. See `LICENSE`, `NOTICE` and `checkpoints/LICENSE`.

## Known limitations

- HDR10 and HLG movies from the Simple presets need the learned model; without
  it, Custom… writes the analytic PQ movie.
- What any reconstruction adds above the clip is an estimate, not recovered
  detail. Use View › Invented pixels and compare with the SDR.
- The macOS build has not had a full pass on an HDR display. The learned model
  on Core ML is faster than on the CPU but only within the model's ONNX tolerance;
  the CPU is the default.
- The builds are not code-signed (see Download).

## Thanks

Built by FXTD Studios · Radiance Research — https://www.fxtdstudios.com
