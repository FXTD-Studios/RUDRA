# RUDRA 1.0.3

A licence fix first, then honesty about the learned model.

## Packages carry no model weights

The Windows and macOS packages up to 1.0.1 included the Beta 2 model
`sdr2hdr_image_v8` in their `models` folder. Those weights are
**non-commercial** (`checkpoints/LICENSE`), while the app is Apache 2.0, so
they never belonged in the package. From 1.0.3 the packages carry no model at
all: the learned ACES 2.0 model and AI fill are downloaded in
**File › Model packages**, and each shows its licence before you download it.

**If you installed 1.0.0 or 1.0.1**, update to 1.0.3, or delete the
`models/sdr2hdr_image_v8` folder from the install, before commercial work.

The build now refuses a public package that carries any weight file, on both
platforms and again before anything is published.

## The learned ACES 2.0 model is labelled preview

RUDRA ACES 2 is unchanged, but it is now called a **preview** in the app's
documentation, the README and the model card. In the last published test it
graded better than the analytic inverse at −3 and +3 EV, but it fails RUDRA's
own 0 EV grading check (0.121 against a limit of 0.0604): at the delivered
exposure its highlights sit further from the reference than the analytic
result. Compare the two before a client sees it. The record, and what would
change the label, is in `docs/MODELS.md`.

## Tested in CI

A new workflow downloads the model you download, checks its files against the
published hashes, and runs it through the native engine on Linux (LibTorch and
ONNX Runtime) and macOS (ONNX Runtime CPU and Core ML). It checks stills and
movies (8-bit H.264 and 10-bit ProRes): every master's ACES 2.0 SDR view gives
back the source within one code, and the backends agree within the model's
published tolerance. It runs weekly as well.

## Smaller fixes

- The installer, `RUDRA.exe`'s file properties and the OFX plug-in's
  description in Resolve and Nuke said "PolyForm Noncommercial". They now say
  Apache License 2.0, which has been the code's licence since 29 October.
- The first-run window said every model was licensed for non-commercial use.
  It now says the 1.0 models are Apache 2.0 and only the older Beta 2 weights
  are non-commercial.
- The release notes are picked from the tag, so a release can no longer go out
  with the previous version's notes.
