# RUDRA 1.2.4

AI fill's new region re-render (experimental), a calmer screen, security fixes, a repair
for model updates, and a new title bar on Windows.

## Opening it the first time

RUDRA is free and open source (Apache 2.0), and the packages are not signed with a paid
certificate, so your system asks once:

- **Windows**: "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS**: right-click RUDRA.app → **Open** → **Open** (or System Settings › Privacy &
  Security › **Open Anyway**).

Download it only from this GitHub page; the checksums are listed with each release.

## If RUDRA ACES 2 says "its files do not check"

Updating the learned ACES 2.0 model from 1.0 to 2.0 could keep the old weights under the new
manifest: the two versions' files are the same size, and RUDRA took a file of the right size
as already installed. The Welcome then said "Found, but its files do not check" while
**File › Model packages** said "Installed and checked".

In 1.2.4 a file counts as installed only when its checksum matches. Open **File › Model
packages**: a package like this now reads "Part downloaded". Press **Resume** and only the
stale files are fetched again.

## Security

- **A crafted EXR file can no longer write outside RUDRA's memory.** The EXR reader checks
  every size and every row against the file before it uses them.
- **AI fill's Set up automatically installs fixed versions.** The installer it downloads is
  one fixed release, checked by its checksum, and the Python packages are the versions the
  fill was tested with, not whatever is newest that day.
- **ffmpeg and other helper programs are started from where RUDRA expects them**, never from
  the folder a project happens to sit in.
- **Updated libraries**: OpenCV 4.14.0, ffmpeg 8.1.2 and ONNX Runtime 1.24.4, each replacing a
  version with a published flaw in reading files or models.
- **Links from the model list and the update check open only when they are https.**
- **Checkpoints are loaded as data only.** A `.pt` file can carry code; RUDRA's Python tools
  no longer run it. For a file you made yourself and trust, set `RUDRA_ALLOW_PICKLE=1`.
- **RUDRA Studio asks for a token** when it is started for the network rather than for this
  computer alone. Its console prints the address to open.
- **RUDRA downloads only the models it knows.** The checksums of every released model are
  built into the app, so a changed model list on the website cannot make RUDRA fetch anything
  else. A model released after this version asks you to update RUDRA first.
- How to report a security problem: [SECURITY.md](../SECURITY.md).

## Windows: one dark title bar

The white title bar and the menu row are gone. The toolbar is now the top of the window, in
RUDRA's own colours: the menus are behind the button at its left, and minimise, maximise and
close are at its right. Dragging, double-click to maximise and snapping work as before, and
the viewer is taller.

- Hovering the maximise button does not open the snap layouts; **Win+Z** does.
- To go back to the system's title bar, set the environment variable
  `RUDRA_NATIVE_TITLE_BAR=1` before starting RUDRA.

## AI fill v4: region re-render (experimental)

AI fill v3 draws only inside pixels that are fully clipped, often a fraction of a per cent
of the frame. The new mode draws a whole bright region again: a window, an open door, a sky
parked just under white.

- Turn it on in **Deliver › AI fill › Region re-render (v4)**, or with **Fill: v3 / v4
  re-render** over the viewer. It uses its own model.
- Type what is in the bright area in **Prompt**. With no prompt it draws texture with no
  subject.
- **Preview AI fill** fills 17 frames round the playhead and shows the one on screen. The
  model works on moving pictures; of a single frame it draws grain.
- Inside the region the SDR's codes are not kept, unlike v3. Outside it every pixel is the
  master's.
- It is tested on the analytic route (**Source › Analytic**). What it draws is a plausible
  guess: check every frame.

## A calmer screen

- **The Delivery column is shorter.** The prompt presets are one button, the tuning controls
  sit behind **Advanced**, the status and the preview result are one line each (the rest on
  hover), and controls that do not apply to the chosen fill are not shown.
- **The technical readouts are off until you ask.** The range line, the pipeline bar under
  the picture and the frame's timings show with **Help › Technical details**.

## Fixes

- Clicking another movie in the Shots list opens it. It was added as one more frame of the
  shot on screen, and the list seemed dead.
- The log at start no longer says "no model loaded" when only the optional reconstruction
  model is absent.
