# RUDRA user guide

RUDRA turns SDR footage (stills, frame sequences, movies) into scene-linear
HDR: an ACES 2065-1 EXR master for grading, or HDR10, HLG and ProRes movies
for delivery. It always starts from an analytic inverse of the SDR's tone
curve (the *baseline*) and can add a learned model's reconstruction on top,
and it shows you where it made something up.

This guide is for the desktop app (0.9.0 beta 6 and later). Help > Getting
started shows the short version inside the app; F1 opens this page.

## 1. Install

- **Windows 10 or 11 (x64):** run `RUDRA-<version>-windows-x64-setup.exe`
  (installs for your user, no administrator prompt), or unzip the portable
  ZIP anywhere and run `RUDRA.exe`. ffmpeg comes with the package.
- **macOS 13 or later (Apple silicon):** open the DMG and drag RUDRA to
  Applications. Movies need an ffmpeg with `libx265`, `prores_ks` and
  `zscale` on the PATH (`brew install ffmpeg`) unless the DMG says it carries
  one.

The first start checks the display (its real peak brightness) and the model
on your GPU or CPU. Help > Check the display and the model runs it again.

## 2. The five steps

### Source: open the shot

- **File > Open frames…** (O) adds stills; **File > Open folder of frames…**
  opens a numbered sequence; a movie (`.mov`, `.mp4`, `.mxf`, `.mkv`, …)
  opens as a shot of its own. Dropping files or a folder on the window does
  the same.
- **Reconstruct > Source** says how the SDR was made. This matters more than
  anything else: the baseline is the inverse of that curve.
  - *Unknown (ACES inverse)*: the safe default for graded or broadcast SDR.
  - *ACES*, *filmic (Hable)*, *AgX*: CG renders and games with those tone mappers.
  - *Camera log*: a log clip viewed through a LUT you know.
  - *Rec.709, plain clip*: video that was simply clipped at white.

**Generated video** (text-to-video and image-to-video tools, renders from
game engines, upscaler output): these are 8-bit SDR with clipped highlights
and no real camera curve. Start with *Source: unknown*; if the picture has a
filmic shoulder (soft roll-off into white), try *filmic (Hable)* or *AgX* and
keep the one whose baseline looks most natural at a high view peak. Leave the
model's strength low (or mode *Highlights*): generated frames change from
frame to frame, and a strong reconstruction can flicker. Check a few frames
across the clip with the scrub bar before exporting.

### Calibrate (optional)

On the Reconstruct tab, **Calibrate** lets you click up to three patches
(black, an 18% grey card, a highlight) and type the nits each should be. The
curve is fitted through them. **Reference** fits the curve to a graded HDR
version of the same frame (EXR, or 16-bit PQ PNG/TIFF) when you have one.

### Reconstruct

- **Mode** (1 2 3 4): what the model may change: *All*, *Highlights*,
  *Shadows*, or *Off* (the baseline alone).
- **Strength** ([ and ]): how much of the model's residual is added.
- **Preserve outside masks**: with painted masks, keep everything outside
  them at the baseline.
- Hold **B** to flip to the baseline; **W** wipes baseline (left) against
  RUDRA (right); drag the image to move the wipe.

### Check

- **View peak**: how bright the viewer shows the HDR. It only changes the
  view, never the master. The Compare line under the viewer says what RUDRA
  changed and whether the peak hides it.
- **False colour** shows luminance zones in nits; **Difference** what the
  network changed.
- **Invented** marks in magenta the values RUDRA made up where the SDR
  clipped or crushed. Look at these before you deliver.
- The scopes (waveform, histogram, vectorscope) and the probe (hold Alt over
  the picture) read the HDR values.
- **Grade > Region EV** pushes a range of nits up or down; painted masks
  limit a band to part of the frame.

### Deliver

- **Master EXR** (M): ACES 2065-1 (or linear Rec.2020) EXR, a frame, all
  frames or a range, with a JSON sidecar describing exactly how it was made.
- For a movie, the export sheet adds **HDR10**, **HLG**, **ProRes 422 HQ** and
  **ProRes 4444** (graded PQ, or scene-referred ACEScct / ARRI LogC4) with
  the audio. Exports run in a queue you can stop and resume (the export
  sheet's Queue button shows it).
- **Anchor to source exposure** keeps mid-tones where the SDR had them; the
  **Anchor knee** says where that lets go.

## 3. Projects and autosave

- **File > Save project** (Ctrl+S / Cmd+S) writes a `.rudra` file: the shot,
  the frame, the model and every setting of the grade, the view peak and the
  Deliver tab. Painted masks go beside it as `<name>.rudra.masks.png`.
- **File > Open project…** (Ctrl+O), a drop on the window, or on Windows a
  double-click on a `.rudra` file opens one. A project moved together with its footage
  still opens: paths under the project's folder are stored relative to it.
  Missing footage or a missing model is named in the log; the grade still
  opens.
- The session is **autosaved** a few seconds after every change. If RUDRA
  closed unexpectedly, **File > Reopen last session** brings it back. With
  two RUDRA windows open, the first one keeps the autosave.
- Opening another project, or closing RUDRA, with unsaved changes to a
  project asks whether to save them first.

## 4. Updates

RUDRA asks GitHub once a day whether a newer release exists (a beta looks
for betas too) and says so in the status bar and the log. Nothing is
downloaded or installed for you. **Help > Check for updates…** asks now.
To turn the daily check off, set `updates/auto=false` in RUDRA's settings or
start it with the environment variable `RUDRA_NO_UPDATE_CHECK=1`.

## 5. Command line

`rudra-native` (beside `RUDRA.exe`, or `RUDRA.app/Contents/MacOS/`) runs the
same pipeline without the window: `video` (a movie in, HDR10/HLG/ProRes
out), `deliver`, `batch` (a queue file), `diff`, `info`, `bench` and
`ffmpeg-check` (is this ffmpeg complete?). Run `rudra-native` alone for
the commands and their options.

The packaged ffmpeg is used first. To use your own instead, set
`RUDRA_FFMPEG_DIR` to its folder, or `RUDRA_FFMPEG_DIR=path` for the one on
the PATH.

## 6. When something goes wrong

- **Help > Open log folder** opens the log (`rudra.log`, kept to 5 MB, three
  old copies). Attach it to a report.
- A failed write says why: the disk is full (with the space free and
  needed), access is denied (Windows: Security > Controlled folder access),
  or the folder is missing.
- "no Qt platform plugin could be initialized" should no longer happen;
  if it does, the log names the setting that caused it.
- Report issues at https://github.com/FXTD-Studios/RUDRA/issues with the
  version (Help > About) and the log.

## Keyboard

| Key | Action |
| --- | --- |
| O | Open frames |
| Ctrl+O / Ctrl+S / Ctrl+Shift+S | Open / save / save project as |
| M | Master EXR |
| , / . | Previous / next frame |
| Home / End | First / last frame |
| Space | Play / pause |
| [ / ] | Weaker / stronger residual |
| 1 2 3 4 | Recovery: all, highlights, shadows, off |
| P | Preserve outside masks |
| B (hold) | Flip to the baseline |
| W | Wipe |
| Z / Y | Undo / redo |
| G / Shift+G | Action / title safe guides |
| F1 | This guide |
| ? | The shortcut sheet |

On macOS, Ctrl is Cmd.
