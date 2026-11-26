# RUDRA 1.3.2

RUDRA 1.3.2 brings RUDRA into DaVinci Resolve and Nuke as an OFX plug-in, installed from RUDRA's Help menu. It
also fixes AI fill in the installed app: since 1.3.0 its preview failed before it started. AI fill's prompts are
better worded and read faster, and the plug-in suggests one too.

1.3.2 is the last 1.x release with new features: 1.3.x takes fixes only from here, and new work goes into RUDRA 2.0.

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
- **AI region fill** (experimental, needs AI fill v5 set up in RUDRA): Prompt, **Suggest prompt**, **Prompt
  strength** (0 to 2), Seed, Steps (10 to 50, default 30) and From/To frames. The status says which frames it will
  draw and about how long that takes. Generated fills are stored per shot and reused, so a frame is drawn once.
- **Mask** (Nuke): limit RUDRA to a matte, with **Invert**. Outside the mask the picture is the plain inverse.
- The RUDRA icon in both hosts, and a RUDRA menu on Nuke's toolbar.

### Install

Close Resolve and Nuke, then in RUDRA choose **Help › Install the OFX plug-in for Resolve and Nuke…**. It copies
`RUDRA.ofx.bundle` into the system's OFX folder (asking once for an administrator), writes the plug-in's settings
for this RUDRA, and adds the RUDRA menu to Nuke. It says when an older or newer plug-in is already installed,
and when Resolve or Nuke is still open. Details: [OFX_PLUGIN.md](OFX_PLUGIN.md).

## Prompts for AI fill

- **Read in seconds.** A prompt not used before waited while the 11 GB text model read it on the CPU: 8.6 minutes
  for a short prompt on the test machine. The text model now reads only the prompt's own words (about 40 of the
  512 it read before) and gives exactly the same result, in about a second plus loading the model. In the plug-in,
  Generate starts reading the prompt while the frames are collected.
- **No extra time for a prompt.** A prompt used to be blended into the fill with a second pass of the model at
  every step. At Prompt strength 1 it now drives the fill in one pass: on the gate shots the picture moved by
  under 0.03 stop (p95, in the clipped areas), passed the same checks, and took 20 to 26 % less time, the same as
  no prompt. Other strengths still blend, about a third longer; the time the plug-in announces counts it.
- **Better wording.** The suggested prompts (the presets and "From this shot") describe more of what the clipped area
  looks like, a sky's gradient or a lamp's core and falloff, and end with the words the AI fill model was trained
  under ("scene-referred high dynamic range footage, physically plausible highlights"). The model learned to fill
  under those words only, so a prompt now steers what is drawn without leaving them. On 09V (a sky through a cargo
  door) the new wording drew softer clouds and flickered less than 1.3.1's (0.024 stop against 0.028). A welding
  shot is now suggested as a welding arc, not as specular glints.
- **Suggest prompt in Resolve and Nuke.** The plug-in's Suggest prompt writes a prompt for the frame on screen, as
  the app's Suggest a prompt does, with Florence-2 when RUDRA's AI fill setup installed it.
- **Prompt strength in the plug-in**, as in the app's Deliver: above 1 the fill follows the prompt more closely
  (experimental). At 1.5 the sky on 09V came out smoother and still passed every check.

## Fixed

- **AI fill in the installed app.** The AI fill preview and the region re-render failed with
  "ModuleNotFoundError: No module named 'tools'" in every installed RUDRA since 1.3.0, so pressing AI fill seemed
  to do nothing. They read the source through a helper that only a source checkout has. The helper is now part
  of RUDRA itself, a test loads AI fill the way the installed app does, and the Windows build stops if any of
  RUDRA's Python imports from outside what it ships.

- **AI fill in Resolve collected no frames.** Resolve numbers a clip's frames from its source media but gives
  the clip's range from 0, so every frame fell outside the range and the status stayed at "0 of n". In Resolve,
  AI fill now counts its frames from the playhead: put it on the first frame to draw, then press Generate.

## Also

- `RUDRA.ofx` carries a version resource and is signed with RUDRA.exe and rudra-native.exe.
- Release builds: when the app tests crash on macOS, the log now shows where, and a test run can be repeated to
  catch a crash that comes and goes (1.3.1's macOS build crashed twice before it passed).

## Tested

- OFX: the plug-in's tests (33 of 33: the picture, the learned engine, AI fill, Suggest prompt, Prompt strength) and
  a test host that loads the bundle and renders through it, with the mask and Suggest prompt (7 of 7). App tests
  for AI fill, Deliver, Batch, the QC report and Help: 68 pass; 6 skip because they need the learned model loaded.
- Python, AI fill and the plug-in's helpers with PyTorch: 149 pass.
- AI fill in an installed layout: `rudra.genfill` loads with only RUDRA's shipped folders on the path.
- Prompts on the gate shots (33 frames each, the analytic route): no prompt, 1.3.1's and 1.3.2's wording, one pass
  against the blend, strength 1.5 and guidance 2. 09V passes every check in every variant. 04 fails the flicker
  check in every variant, prompt or not (0.22 stop against 0.10), as in 1.3.1 (below).
- In Resolve and Nuke: to be completed before release.

## Known limits

- AI fill (in the app and in the plug-in) is experimental. In the plug-in it draws about 150 frame-steps a
  minute on an RTX 4090 (100 frames at 30 steps took about 20 minutes); each new prompt first takes a while to
  read.
- The Mask input is Nuke's; Resolve's filter context has none (use Resolve's own window or matte there).
- The limits listed for 1.3.1 still apply.
