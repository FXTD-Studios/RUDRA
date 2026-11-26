# RUDRA 1.3.2

RUDRA 1.3.2 brings RUDRA into DaVinci Resolve and Nuke as an OFX plug-in, installed from RUDRA's Help menu. It
also fixes AI fill in the installed app: since 1.3.0 its preview failed before it started.

## Opening it the first time

- **Windows**: if Windows asks, choose **More info** › **Run anyway**. Signing through SignPath
  Foundation follows once SignPath has approved RUDRA ([code signing policy](CODE_SIGNING_POLICY.md)).
- **macOS**: right-click RUDRA.app › **Open** › **Open** (or System Settings › Privacy & Security ›
  **Open Anyway**).
- **Linux**: unpack `RUDRA-1.3.2-linux-x86_64.tar.gz` and run `RUDRA` inside it.

Download it only from this GitHub page; the checksums are listed with each release.

## RUDRA in Resolve and Nuke (OFX)

The **RUDRA SDR to HDR** effect turns an SDR clip into a scene-linear HDR master inside the host. It runs on
the RUDRA installed on the same computer, so it downloads no model of its own.

- **Route**: Analytic, or RUDRA's learned ACES 2.0 route, frame by frame.
- **Luminance limit**: caps the reconstruction's brightest light.
- **AI region fill** (experimental, needs AI fill v5 set up in RUDRA): Prompt, Seed, Steps (10 to 50, default
  30) and From/To frames. The status says which frames it will draw and about how long that takes. Generated
  fills are stored per shot and reused, so a frame is drawn once.
- **Mask** (Nuke): limit RUDRA to a matte, with **Invert**. Outside the mask the picture is the plain inverse.
- The RUDRA icon in both hosts, and a RUDRA menu on Nuke's toolbar.

### Install

Close Resolve and Nuke, then in RUDRA choose **Help › Install the OFX plug-in for Resolve and Nuke…**. It copies
`RUDRA.ofx.bundle` into the system's OFX folder (asking once for an administrator), writes the plug-in's settings
for this RUDRA, and adds the RUDRA menu to Nuke. It says when an older or newer plug-in is already installed,
and when Resolve or Nuke is still open. Details: [OFX_PLUGIN.md](OFX_PLUGIN.md).

## Fixed

- **AI fill in the installed app.** The AI fill preview and the region re-render failed with
  "ModuleNotFoundError: No module named 'tools'" in every installed RUDRA since 1.3.0, so pressing AI fill seemed
  to do nothing. They read the source through a helper that only a source checkout has. The helper is now part
  of RUDRA itself, a test loads AI fill the way the installed app does, and the Windows build stops if any of
  RUDRA's Python imports from outside what it ships.

## Also

- `RUDRA.ofx` carries a version resource and is signed with RUDRA.exe and rudra-native.exe.
- Release builds: when the app tests crash on macOS, the log now shows where, and a test run can be repeated to
  catch a crash that comes and goes (1.3.1's macOS build crashed twice before it passed).

## Tested

- OFX: the plug-in's picture tests (26 of 26), a test host that loads the bundle and renders through it, with
  the mask (6 of 6), and the AI fill job (8 of 8). App tests: 223 pass; 11 skip because they need the learned
  model loaded.
- AI fill in an installed layout: `rudra.genfill` loads with only RUDRA's shipped folders on the path.
- In Resolve and Nuke: to be completed before release.

## Known limits

- AI fill (in the app and in the plug-in) is experimental. In the plug-in it draws about 150 frame-steps a
  minute on an RTX 4090 (100 frames at 30 steps took about 20 minutes); each new prompt first takes a while to
  read.
- The Mask input is Nuke's; Resolve's filter context has none (use Resolve's own window or matte there).
- The limits listed for 1.3.1 still apply.
