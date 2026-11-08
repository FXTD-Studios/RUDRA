# RUDRA 1.2.0

The learned ACES 2.0 model that passed our hold-out test, real-time playback through it, and
an AI fill you preview on one frame before it fills the shot. The promise is unchanged: the
SDR you delivered still shows within one 8-bit code of the original, and anything RUDRA
invents is marked.

## Opening it the first time

RUDRA is free and open source (Apache 2.0), and the packages are not signed with a paid
certificate, so your system asks once:

- **Windows**: "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS**: right-click RUDRA.app → **Open** → **Open** (or System Settings › Privacy &
  Security › **Open Anyway**).

Download it only from this GitHub page; the checksums are listed with each release.

## RUDRA ACES 2.0 with the lock

- **A new learned model, RUDRA ACES 2 (2.0)**, a download in **File › Model packages**
  (Apache 2.0, no weights in the app). It is the first learned model to pass our hold-out test
  on footage it was never trained on: at the delivered exposure it keeps the SDR within 0.1
  of an 8-bit code of the plain ACES 2.0 inverse, and it rebuilds clipped highlights and
  shadows better than the analytic route at −3 and +3 EV.
- **The lock** runs inside the engine on every frame. It ties the model's result back to
  the SDR codes you delivered, so the model can only add above and below them. The C++ engine
  gives the same result as the reference implementation, bit for bit.
- **1.2.0 or later is needed for this model.** It is a new package version (contract 3);
  earlier apps refuse it instead of running it without the lock.
- **What was tested:** one production and one camera (ARRI ALEXA LogC3), single frames, and
  a visual review of 24 frames against the analytic inverse at −3, 0 and +3 EV (SDR review
  sheets and the EXR masters; not yet on an HDR monitor). Published as **release** in Model
  packages on 8 November 2026. The analytic reconstruction, which needs no model, is still
  the default.

## The viewer

- **Real-time playback on the learned route.** The engine kept decoding the movie from its
  first frame for every frame it showed; it now reads on from an open decoder, so "Preparing
  playback" finishes quickly and the cached frames play in real time.
- **The window fits the screen.** The viewer bar wraps onto a second line on narrow
  windows, and the right-hand panels stay inside the window at any size.
- **The region tint tells highlights from shadows:** warm for the highlight side, blue for
  the shadow side, on the viewer and on the histogram's curve.
- The region histogram now reads the learned route's picture when that route is shown.

## AI fill (experimental)

- **Preview AI fill fills the frame at the playhead** and shows it in the viewer. If it looks
  right, rendering applies the same prompt and settings to the whole shot.
- **Several areas at once:** with **More detail in the filled area**, each clipped area of
  the shot (up to eight) gets its own box at the fill model's full resolution, so separate
  windows or lamps each get their own detail rather than sharing one large box.

## Models

**RUDRA ACES 2** (2.0, Apache 2.0) and **AI fill v3** (Apache 2.0, experimental) are
downloads in File › Model packages, checked file by file against their SHA-256 before use.
See `docs/MODELS.md`.

## Known limits

- RUDRA ACES 2 was checked on one production and one camera; review it on your own shots.
- The lock adds about 0.2 to 0.4 s per frame to a master.
- **+3 EV previews of very bright, slightly tinted highlights** can show isolated black pixels
  on the SDR view: the ACES 2.0 output transform in OpenColorIO 2.5.2 maps some off-neutral
  values around 130 linear to black. The EXR master is correct at those pixels.
- **MaxCLL above the monitor's peak:** bright highlights can reach thousands of nits. Use
  **Peak limit** on the Learned ACES 2.0 card (for example 1,000 nits) to keep them in range.
- The AI fill needs a 16 GB GPU (NVIDIA with a current driver, or Apple silicon), about
  5 GB for its Python environment and 25 GB for the base model.
