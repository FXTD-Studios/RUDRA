# RUDRA 1.1.0

Control over where the reconstruction acts, and an AI fill you can steer, preview and set
up in one click. The promise is unchanged: the SDR you delivered still shows within one
8-bit code of the original, and anything RUDRA invents is marked.

## Opening it the first time

RUDRA is free and open source (Apache 2.0), and the packages are not signed with a paid
certificate, so your system asks once:

- **Windows**: "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS**: right-click RUDRA.app → **Open** → **Open** (or System Settings › Privacy &
  Security › **Open Anyway**).

Download it only from this GitHub page; the checksums are listed with each release.

## The licence fix, for certain

1.1.0 carries the 1.0.3 fix: **the packages hold no model weights**. Every
model is a download in **File › Model packages** that shows its licence first.
The build refuses a public package that carries any weight file.
**If you installed 1.0.0 to 1.0.3 from a GitHub release**, delete any
`models/sdr2hdr_image_v8` folder from the install (that model is
non-commercial), or simply install 1.1.0.

## Limit the reconstruction to a luminance region

On the **Reconstruct** card, **Limit to a luminance region**:

- **Highlights + shadows**: highlights from an SDR level and shadows below one, each with
  its own softness and strength; or one **Band** between two levels.
- Every pixel outside the region keeps the plain inverse **bit for bit**.
- A **histogram** of the frame shows how strongly the reconstruction acts at each
  brightness ("Acts on x % of this frame · at half strength or more on y %"), and
  **Show on the viewer** tints the picture by the same amount. On the analytic route that
  includes the built-in highlight and shadow priors, the mode and the strength, so the tint
  shows what actually changes.
- The region reaches the learned ACES 2.0 route (strength up to 1), the masters, the
  projects and the AI fill.

## AI fill: steer it, preview it, set it up in one click

The AI fill (experimental, off by default) draws detail inside fully clipped
highlights and crushed shadows. It never changes an 8-bit code of the master's SDR view,
and every pixel it invents is in `rudra.generated`.

- **Set up automatically** now works from the Windows and macOS packages. On a computer
  with no Python for it, RUDRA installs its own (Python 3.12, PyTorch and the fill's
  packages) in its data folder, nothing else on the system changes, then downloads the
  base model. **Download AI fill v3** takes you to the last part, the LoRA.
- **Prompts**: a highlight and a shadow prompt, **Prompt strength** and **Detail
  strength**; nine presets (sky, sunset, sun, lamp, window daylight, glints, flame, shadow
  detail). **Suggest** reads the frame's clipped areas and proposes a prompt with its reason.
- **A prompt bar over the viewer** while AI fill is on: type a prompt, pick a suggestion or
  a preset, press Return to preview.
- **Preview AI fill** fills the frames around the playhead and shows them in the viewer
  (AI fill, Master, Invented), **on the analytic route as well as the learned one**. The
  first time a typed prompt is used the status says it is being embedded (once).
- **More detail in the filled area** (off by default): the fill model draws only a box around
  the shot's clipped areas instead of the whole frame, so a blown sky that is part of the
  frame gets up to twice the detail per side.
- **Drop flickering fill frames** (on by default): a frame whose fill flickers for one frame
  keeps the master's own frame.
- **It fixes the master in place** (the originals in `<master>_before_aifill`), for EXR
  sequences and now for **learned-route movies** (the movie is encoded from the filled frames).
- **It says which model runs**: the LoRA is checked against the published AI fill v3 by its
  SHA-256, with **Use AI fill v3** when an older one is found.

## Delivery

- **Mattes (EXR)** beside each master: what changed, the pixel class, a confidence, the
  highlight, shadow and region weights and the painted masks; the AI fill splits
  `rudra.generated` into highlight and shadow.
- **EV brackets** (−2, 0, +2) beside each master, with a check that the preview and the
  export match.
- **Four analytic containers**: ACES 2065-1, ACEScg, scene-linear Rec.709 and Rec.2020.
- An **ACES Metadata File (AMF v2.0)** beside every EXR master; **alpha** passes through
  EXR, 16-bit TIFF and ProRes 4444; the master stays at the source's native resolution.

## The viewer

- A Nuke-style bar: f/stop and γ, −6 to +2 brackets, an SDR | RUDRA wipe, clamp and clip
  zebras, and a **Range** layer colouring each pixel by its stops above SDR white. An amber
  VIEW badge shows whenever the view is gained.
- The display signal is named on screen (transfer, primaries, white, peak).
- **AI fill** on the viewer bar shows every frame that has a fill, stepping and playing.

## Learned ACES 2.0 route (preview)

- **Rebuild only highlights and shadows** (τ high / τ low), **Clean shadows** (off by
  default) and a highlight ceiling at ACES white.
- The model is **unchanged from 1.0.3** and stays labelled **preview**: it does not yet
  pass the grading check at the delivered exposure on real footage. The analytic
  reconstruction, which needs no model, is the default.

## Models

No new weights in 1.1.0. **RUDRA ACES 2** (Apache 2.0, preview) and **AI fill v3**
(Apache 2.0, experimental) are downloads in File › Model packages, checked file by file
before use. See `docs/MODELS.md`.

## Known limits

- The AI fill needs a 16 GB GPU (NVIDIA with a current driver, or Apple silicon), about
  5 GB for its Python environment and 25 GB for the base model.
- More detail in the filled area has not been through AI fill v3's gates; compare it with
  the default on your shot.
- Clean shadows is spatial only and on the learned route only. ProRes alpha is 10-bit.
- The analytic route has no AI fill for movies yet (EXR sequences only).
