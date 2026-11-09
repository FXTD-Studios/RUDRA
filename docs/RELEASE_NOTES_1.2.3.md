# RUDRA 1.2.3

Review fixes to the first start and the model list, and tidier documentation.

## Opening it the first time

RUDRA is free and open source (Apache 2.0), and the packages are not signed with a paid
certificate, so your system asks once:

- **Windows**: "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS**: right-click RUDRA.app → **Open** → **Open** (or System Settings › Privacy &
  Security › **Open Anyway**).

Download it only from this GitHub page; the checksums are listed with each release.

## Fixes

- **"Installed and checked" means checked.** *Welcome to RUDRA* now hashes the learned
  ACES 2.0 model's files before it says so. A damaged download reads "Found, but its
  files do not check" and points you to download it again.
- **A model added through the Learned ACES 2.0 window is found** by the Welcome too, not
  only one in the models folder.
- **A learned ACES 2.0 package this version cannot use stays in File › Model packages**
  with the reason, instead of disappearing from every list.
- **The GPU on a Mac is named by its chip** (for example "Apple M2 Pro GPU"). On Windows,
  if the system cannot be asked, the Welcome says the GPU was not checked rather than
  "No GPU found".
- **The HDR 1000 monitor is saved** when an older install is moved to it, so it is not
  undone on the next start.

## Documentation

The model card, the model table and the project overview were tidied: third-party
product names and internal footage names are gone, and the model card states the code's
licence correctly (Apache 2.0).
