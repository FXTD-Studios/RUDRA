<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ui/assets/brand/rudra_logo_dark.png">
    <img src="ui/assets/brand/rudra_logo_light.png" width="220" alt="RUDRA">
  </picture>
</p>

<p align="center">
  <b>SDR in, HDR out.</b><br>
  RUDRA turns SDR stills and footage into scene-referred HDR masters,<br>
  and shows you exactly what it added.
</p>

<p align="center">
  <a href="https://github.com/FXTD-Studios/RUDRA/releases/download/v1.0.1/RUDRA-1.0.1-windows-x64-setup.exe"><img alt="Download for Windows" src="https://img.shields.io/badge/Download-Windows%2010%20%2F%2011-0078D6?style=for-the-badge&logo=windows&logoColor=white"></a>
  <a href="https://github.com/FXTD-Studios/RUDRA/releases/download/v1.0.1/RUDRA-1.0.1-macos-arm64.dmg"><img alt="Download for macOS" src="https://img.shields.io/badge/Download-macOS%20(Apple%20silicon)-111111?style=for-the-badge&logo=apple&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://github.com/FXTD-Studios/RUDRA/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/FXTD-Studios/RUDRA?label=release"></a>
  <a href="https://github.com/FXTD-Studios/RUDRA/releases/latest"><img alt="Downloads" src="https://img.shields.io/github/downloads/FXTD-Studios/RUDRA/total?label=downloads"></a>
  <a href="LICENSE"><img alt="Licence: Apache 2.0" src="https://img.shields.io/badge/licence-Apache%202.0-blue"></a>
  <a href="https://huggingface.co/fxtdstudios/rudra-models"><img alt="Models on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20models-rudra--models-yellow"></a>
  <a href="https://www.fxtdstudios.com/rudra-docs/"><img alt="User guide" src="https://img.shields.io/badge/user%20guide-rudra--docs-111"></a>
</p>

---

## What it does

An 8-bit SDR frame throws away everything brighter than its white. RUDRA puts it
back. It reads the SDR through the curve that made it, rebuilds the highlights the
SDR clipped, and writes a master you can grade, deliver or hand to a colourist:
ACES EXR, HDR10, HLG, ProRes or DPX.

You stay in control. Every step is live on screen, the original SDR is one key
away, and RUDRA marks every pixel it had to invent.

<p align="center">
  <img src="docs/images/rudra_look.png" width="900" alt="RUDRA 1.0.1, the Look tab: a sunset opened in the Simple workspace, with Exposure, Strength and Display peak">
  <br><sub><b>1 Look</b>: open a shot and shape its HDR. Hold B for the original SDR, W for a wipe.</sub>
</p>

<p align="center">
  <img src="docs/images/rudra_check.png" width="900" alt="RUDRA 1.0.1, the Check tab: peak 5,856 nits and average 56 nits, with the waveform, histogram, vectorscope and RGB parade beside the picture">
  <br><sub><b>2 Check</b>: peak, average and light above SDR white, with the scopes beside the picture.</sub>
</p>

## Download

| | |
|---|---|
| **Windows 10 / 11** (x64) | [Installer](https://github.com/FXTD-Studios/RUDRA/releases/download/v1.0.1/RUDRA-1.0.1-windows-x64-setup.exe) · [Portable ZIP](https://github.com/FXTD-Studios/RUDRA/releases/download/v1.0.1/RUDRA-1.0.1-windows-x64.zip) |
| **macOS** (Apple silicon) | [DMG](https://github.com/FXTD-Studios/RUDRA/releases/download/v1.0.1/RUDRA-1.0.1-macos-arm64.dmg) |

All files, checksums and release notes are on the
[releases page](https://github.com/FXTD-Studios/RUDRA/releases/latest).

The builds are not code-signed yet. On Windows, choose **More info › Run anyway**
in the SmartScreen prompt. On macOS, open it once from **System Settings › Privacy
& Security › Open Anyway**.

## Three steps

1. **Look.** Open a still, a folder of frames or a movie (MOV, MP4, MXF, MKV and
   more). Set the exposure, the strength and your display's peak. Hold **B** to
   see the original SDR, or press **W** for a wipe: SDR on the left, RUDRA on the
   right.
2. **Check.** Read the peak, the average light, how much is above SDR white and
   how much is outside P3, with the scopes beside the picture. **Invented pixels**
   shows what RUDRA made up.
3. **Deliver.** Pick a preset: **Grading master** (ACES EXR), **HDR10 movie**,
   **Broadcast HDR** (HLG), or **Custom…** for everything else. Files are written
   whole or not at all, and nothing is ever overwritten.

New users start in the **Simple** workspace. **Full** has every control: masks,
per-region exposure, batch queues, EDL conform and more.

## The learned model (optional)

RUDRA works out of the box. For more of the clipped light, download the learned
**ACES 2.0** model from inside the app: **Look › Download the model…** or
**File › Model packages…**. It is free and checked file by file before use. The
models are also on [Hugging Face](https://huggingface.co/fxtdstudios/rudra-models).

**AI fill** (experimental, off by default) can draw detail inside large blown
areas. It needs a 16 GB GPU and the Wan2.1-VACE-1.3B base model. What it draws is
a plausible guess, not recovered detail, so check every frame.

## System requirements

- **Windows** 10 or 11, x64. The learned model runs on NVIDIA (CUDA) or any
  DirectX 12 GPU (DirectML); without one, on the CPU.
- **macOS** on Apple silicon. In 1.0 the learned model and AI fill are Windows
  features.
- An HDR display is recommended for judging the result. RUDRA also simulates HDR
  on an SDR screen.

## Help

- [User guide](https://www.fxtdstudios.com/rudra-docs/): every control, step by step
- [Release notes](https://github.com/FXTD-Studios/RUDRA/releases/latest)
- [Report a problem](https://github.com/FXTD-Studios/RUDRA/issues)
- Developers and researchers: [project overview](docs/PROJECT_OVERVIEW.md),
  [building from source](native/README.md), [training](docs/TRAINING.md)

## Licence

The code and the 1.0 models (learned ACES 2.0, AI fill v3) are licensed under the
[Apache License 2.0](LICENSE): free for commercial use. Keep the
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) files when you redistribute.

The older Beta 2 weights (`sdr2hdr_image_v8`, `sdr2hdr_shadow_v1`) are
non-commercial; see [`checkpoints/LICENSE`](checkpoints/LICENSE). Each model shows
its licence in the app before you download it.

---

<p align="center"><a href="https://www.fxtdstudios.com">www.fxtdstudios.com</a> · FXTD Studios</p>
