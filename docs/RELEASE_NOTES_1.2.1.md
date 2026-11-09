# RUDRA 1.2.1

Fixes from user feedback on the first start, and a viewer that opens as a
comparison.

## Opening it the first time

RUDRA is free and open source (Apache 2.0), and the packages are not signed with a paid
certificate, so your system asks once:

- **Windows**: "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS**: right-click RUDRA.app → **Open** → **Open** (or System Settings › Privacy &
  Security › **Open Anyway**).

Download it only from this GitHub page; the checksums are listed with each release.

## The first start

- **The GPU is found without a model.** *Welcome to RUDRA* now names your graphics card
  (and on Windows its memory) in the Acceleration step as soon as it opens, and says which backends
  will run a model on it. Before, the step stayed blank until a model was loaded and
  tested, so it looked as if no GPU had been detected.
- **No licence to accept.** The Windows installer shows the licence for information only;
  Apache 2.0 asks for no acceptance. The licence text now reads
  "Copyright 2026 FXTD Studios" instead of the template's placeholder, and the welcome
  window names FXTD Studios and the licence.

## The viewer opens as a comparison

- RUDRA starts with a wipe: the **SDR original on the left, RUDRA on the right**, at a
  **1,000-nit** display peak instead of the SDR white. A project opens at the peak it was
  saved with.
- The learned ACES 2.0 route opens on the **HDR 1000** monitor. Installs that had saved
  SDR 100 or Scene are moved to HDR 1000 once; your next choice is kept.

## Also since 1.2.0

- The learned ACES 2.0 route steers you to the analytic route when the source's curve is
  not ACES, where the model would push near-white areas too bright.
- The AI fill preview is shown through the chosen monitor.
- **Highlight detail** on the analytic route: amplifies the structure that survives just
  under white (a sky, a window, an open door).
