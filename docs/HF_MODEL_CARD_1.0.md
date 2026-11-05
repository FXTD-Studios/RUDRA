---
license: apache-2.0
library_name: rudra
inference: false
pipeline_tag: image-to-image
base_model:
  - Wan-AI/Wan2.1-VACE-1.3B-diffusers
tags:
  - hdr
  - sdr-to-hdr
  - inverse-tone-mapping
  - aces
  - vfx
  - color-grading
  - lora
---

# RUDRA models — FXTD Studios

The models behind [RUDRA](https://github.com/FXTD-Studios/RUDRA), FXTD Studios'
SDR → scene-linear HDR reconstruction for VFX and grading. **Apache 2.0,
commercial use permitted.**

| Folder | What it is | Size |
|---|---|---|
| [`models/RUDRA_ACES_2/`](models/RUDRA_ACES_2) | **RUDRA ACES 2**: the learned ACES 2.0 reconstruction. 8-bit SDR in, an ACEScg master out, whose ACES 2.0 SDR view keeps the source's 8-bit codes. Ships as TorchScript and ONNX (DirectML, Core ML or CPU). | 2.2 MB |
| [`models/rudra_genfill_v3/`](models/rudra_genfill_v3) | **RUDRA AI fill v3**: a LoRA for Wan2.1-VACE-1.3B that draws the *shape* inside fully clipped highlights, and a small brightness estimator (`brightness.json`) that sets *how bright* they are. | 271 MB |

## Use

**In the RUDRA app** (Windows and macOS):

- **RUDRA ACES 2:** File › Model packages, or Look › Download the model. Every
  file is checked against its SHA-256 before it is used.
- **AI fill:** File › Model packages › AI fill v3 › Download. The app then uses
  this LoRA and `brightness.json`. It also needs:
  - the base model [Wan2.1-VACE-1.3B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)
    (set its folder in Deliver › AI fill › Setup…, or put it in `<RUDRA folder>/models/genfill/`);
  - Python (`scripts\GENFILL.ps1 -Setup` on Windows, `scripts/genfill_setup.sh` on a Mac);
  - a 16 GB GPU (on a Mac, 16 GB of memory or more: it runs on Metal).

  Then use Deliver › AI fill.

**From the command line** (RUDRA repository):

```bash
rudra-native aces-v2 models/RUDRA_ACES_2 <frames or movie> --out <dir>
python -m rudra.genfill.run --master <EXR folder> --out <new folder> \
    --base Wan2.1-VACE-1.3B-diffusers --lora models/rudra_genfill_v3/rudra_genfill.safetensors
```

`run.py` loads `brightness.json` from the LoRA's folder by default. Add
`--brightness-offset <stops>` to shift the brightness (±3).

## What the AI fill guarantees

- **Codes kept:** no 8-bit code of the master's ACES 2.0 SDR 100-nit view
  changes. Pixels it does not touch are copied bit for bit.
- **Highlights only:** it fills only fully clipped areas (code 255), and adds
  at most +3 stops over the master.
- **Marked:** every generated pixel is marked in a `rudra.generated` EXR
  channel.
- **What it draws is plausible, not recovered.** It is experimental and off by
  default in the app.

## Evaluation (AI fill v3, pre-registered, 30 Oct 2026)

The test used 58 clips. The verdict counts only 25 Poly Haven scenes that were
held out and never seen before (`registration_v3.json`, `report_v3.json`).

| Gate | Result |
|---|---|
| Clipped-highlight log2 error vs the exact inverse | **−0.128 stops** (95 % CI −0.226 … −0.042) |
| Detail (PU21) at −3 / −6 EV | **−7.1 / −6.5** (lower is better) |
| Brightness estimate vs leaving it | **−0.199 stops** (CI −0.339 … −0.076); overshoot > 1 stop in 3.3 % |
| 8-bit codes / shadows / flicker / window seams | kept / unchanged / 0.007 EV / 1.0 |

On footage with small clipped areas it is neutral.

## Training data

All training data is commercial-clean:

- [Poly Haven](https://polyhaven.com/license) HDRIs (CC0);
- FXTD Studios' own footage and renders;
- procedurally generated scenes;
- CC0 plates.

The base model, Wan2.1-VACE-1.3B, is Apache 2.0 from Wan-AI.

Not covered here: the older Beta 2 weights (`sdr2hdr_image_v8`,
`sdr2hdr_shadow_v1`) are a separate, non-commercial release.

## Licence

[Apache License 2.0](LICENSE). Copyright 2026 FXTD Studios. Wan2.1-VACE-1.3B
is © Wan-AI under Apache 2.0.

[www.fxtdstudios.com](https://www.fxtdstudios.com) · FXTD Studios
