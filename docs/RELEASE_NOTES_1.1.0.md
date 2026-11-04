# RUDRA 1.1.0

Inspection, delivery and AI fill, built around one promise: the SDR you
delivered still shows within one 8-bit code of the original, and anything
RUDRA invents is marked.

## The licence fix, for certain

1.1.0 carries the 1.0.3 fix: **the packages hold no model weights**. Every
model is a download in **File › Model packages** that shows its licence first.
The build refuses a public package that carries any weight file.
**If you installed 1.0.0 to 1.0.3 from a GitHub release**, delete any
`models/sdr2hdr_image_v8` folder from the install (that model is
non-commercial), or simply install 1.1.0.

## The viewer

- **A Nuke-style bar under the viewer**: f/stop and γ, −6 to +2 brackets,
  an SDR | RUDRA wipe with A/B pickers, clamp and clip zebras, and a **Range**
  layer colouring each pixel by its stops above SDR white. None of it touches
  the master; an amber VIEW badge shows whenever the view is gained.
- The viewer now reads the SDR exactly as the masters do (BT.1886 by default).
- **AI fill** on the viewer bar: every frame that has a fill is shown filled,
  stepping and playing; with **Changes** on, what it invented.

## Learned ACES 2.0 route

- **Rebuild only highlights and shadows** (τ high / τ low): the model's result
  only where the source is that bright or that dark; everything between is the
  plain ACES inverse, pixel for pixel.
- **Clean shadows** (off by default): an edge-preserving denoise of the deep
  shadows, held to the one-code guarantee. It invents nothing.
- **A highlight ceiling at ACES white**: slightly tinted near-white codes no
  longer invert to hundreds of thousands of nits. Pixels the ACES view itself
  cannot return are reported apart (`aces_inverse_limit`), not as RUDRA errors.
- The learned model stays **preview**: it grades better at −3 and +3 EV, but
  not yet at the delivered exposure on real footage.

## Delivery

- An **ACES Metadata File (AMF v2.0)** beside every EXR master, validated
  against the Academy's schema.
- **Alpha** passes through: EXR masters (premultiplied), 16-bit TIFF and
  ProRes 4444 log copies (straight, 10-bit). ProRes 422 HQ, HEVC, DPX and the
  16-bit QuickTime have no alpha; the provenance says where it went.
- The master stays at the **source's native resolution**. Resize for a
  delivery in your finishing tool (Resolve's Deliver page, Nuke's Reformat),
  and label an up-conversion as such.

## AI fill

- **It fixes the master in place**: when a fill finishes, the clipped areas of
  the master frames are filled and the originals are kept in
  `<master>_before_aifill`. A stopped or failed fill never touches the master.
- **It says which model runs**: the LoRA is checked against the published
  AI fill v3 by its SHA-256; an older one is flagged, with **Use AI fill v3**.
- **Set up automatically** and **Preview AI fill** around the playhead.

## Playback and the app

- **Playback is prepared in the background** ("Preparing playback: N of M"),
  then Play is real time, kept by the clock: if the screen cannot keep up it
  drops frames rather than slowing, and says so when you stop.
- Remove a shot from the list (× or right click); the files stay on disk.
- The right-hand columns no longer slide sideways.

## Tools

`tools/measure_flicker.py` and `scripts/RUN_FLICKER_CLIPS.ps1` (flicker on any
real clip against 0.02 EV), `scripts/RUN_4K_TEST.ps1` (a minute of 4K on a
16 GB card), `tools/inspect_highlights.py`, `tools/holdout_kit.py`,
`tools/build_real_partial_corpus.py`.

## Known limits

Clean shadows is spatial only and in the learned route only. ProRes alpha is
10-bit. The learned model's 0 EV result on real footage is the open problem;
see `docs/competitive/`.
