# RUDRA 1.2.2

Two fixes to the first start, found by checking 1.2.1 on a real screen.

## Opening it the first time

RUDRA is free and open source (Apache 2.0), and the packages are not signed with a paid
certificate, so your system asks once:

- **Windows**: "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS**: right-click RUDRA.app → **Open** → **Open** (or System Settings › Privacy &
  Security › **Open Anyway**).

Download it only from this GitHub page; the checksums are listed with each release.

## Fixes

- **An installed RUDRA ACES 2 is recognised.** With only the learned ACES 2.0 model
  downloaded, *Welcome to RUDRA* said "No model package found" and **File › Model
  packages** listed the model as "unreadable", beside its own "Installed and checked" row.
  The Welcome now names the model with a green tick, and the list no longer shows it as
  broken: it runs through the learned ACES 2.0 route, not the SDR to HDR loader.
- **The Welcome window fits its content.** With the GPU line added in 1.2.1 the HDR test
  card could draw over the Acceleration heading; the window is now never shorter than
  what it shows.
