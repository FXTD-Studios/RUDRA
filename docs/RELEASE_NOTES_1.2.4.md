# RUDRA 1.2.4

Security fixes, a repair for model updates, a new title bar on Windows and a calmer screen.

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
  no longer run it, whatever the settings. An old checkpoint of your own with objects in it is
  re-saved once as tensors (a state_dict).
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
