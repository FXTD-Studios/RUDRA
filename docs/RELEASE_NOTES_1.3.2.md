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

## Opening frames

- **File › Import image sequence…**: pick any one frame of a numbered sequence (`shot_0001.exr` to `shot_0240.exr`,
  say) and RUDRA opens every frame of it, in order, as one shot. EXR, PNG, JPEG, TIFF, BMP and WebP.
- **Media's right-click menu**, anywhere in the panel on the left: **Import…** for videos and images, with an
  **Image sequence** box that brings a numbered sequence in as one clip; **Clear from Media** (everything, or only
  the videos, images or image sequences; the files stay on disk); and **Clear VRAM**, which frees the GPU memory RUDRA
  holds (the AI fill preview's model, about 16 GB, and the learned model's cache, which loads again at once).
- **EXR stills and sequences** open as PNG, JPEG and TIFF frames do, and in Open frames, Open folder of frames and a
  drop on the window. RUDRA reads an EXR as a linear Rec.709 picture of an SDR display: values above 1.0 are clipped,
  the rest encoded with the Rec.709 display curve (BT.1886, gamma 2.4), like the SDR frames it is made for. An EXR
  that is already HDR loses what is above 1.0.

## Deliver

- **Every EXR compression, on both routes.** Deliver's EXR compression lists every OpenEXR one: lossless **ZIP** (the
  default), **ZIPS**, **PIZ**, **RLE** and **None**, and lossy **DWAA**, **DWAB**, **B44**, **B44A** and **PXR24** for
  reviews and dailies. It applies to the analytic and the learned ACES 2.0 masters alike, to a learned movie's kept EXR
  masters and to the masters AI fill draws on, whose filled frames keep the same compression. The learned route wrote
  ACES 2065-1 uncompressed whatever was chosen: the AP0 file now takes the compression too. Compressed, it is no longer
  the SMPTE ST 2065-4 container (which is uncompressed), and its header says so; choose None for the strict container.
  Masks and mattes stay lossless under a lossy choice. ZIP, ZIPS and None are written exactly as before; the others by
  the OpenEXR library, which RUDRA now builds with. The command line takes the same names: `rudra-native aces-v2 ...
  --exr-compression piz`.
- **A progress bar for this shot's render**, under the Render card and apart from the Batch queue: what is being
  written (an EXR or DPX still or sequence, a MOV's frames and then its encode, the learned ACES 2.0 master, an
  AI-filled movie, a headroom export), frames done of all, the share and, once a few frames are in, the time left.
  **Stop** beside it stops after the frame being written and keeps what was written. At the end it says
  "Done: 240 frames in 3 min 12 s" with **Open folder**, or how far it got. The AI fill keeps its own bar in its card.

## Fixed

- **The timeline jumps to the frame you click, also while the shot plays.** A click on the frame strip stopped only
  the plain playback: on the learned ACES 2.0 route and the headroom route the playback carried on over the frame
  asked for, so you had to wait or pause first. A click (or a drag) now stops any playback and shows that frame;
  while dragging, only the playhead and the timecode follow, so the drag keeps up, and hovering says which frame a
  click shows.
- **AI fill: "Set up automatically" stopped at "could not make the Python environment"** on a computer where Windows
  does not let a program through a folder junction a user made ("The path cannot be traversed because it contains
  an untrusted mount point. (os error 448)"). uv, which installed AI fill's private Python, keeps that Python behind
  such a junction, so the setup never reached the model. The setup now downloads the Python itself
  (python-build-standalone 3.12.13, checked by its SHA-256), makes AI fill's environment with Python's own venv and
  uses uv only to install packages: no junction anywhere. An environment that does not start is made again.
- **AI fill setup: "the AI fill's packages did not install" on every new computer.** One of the pinned package
  versions (tokenizers, raised by an automatic update) could not go with the pinned transformers, so the package step
  failed for everyone setting up fresh. It is back to the tested version, and every change to the pinned list is now
  checked for Windows, Linux and macOS before it is merged.
- **An AI fill setup that stopped once kept failing.** A setup stopped partway (the window closed, the network
  dropped) left a Python without PyTorch, or a model download with its small files and not its weights. Every later
  Set up automatically then failed ("PyTorch is not installed", "the model does not match the manifest"), and the app
  took the half-downloaded model for an installed one. Set up automatically now carries on where it stopped, and a
  model whose weights are missing says so: "the download … did not finish".
- **AI fill setup now says why it stopped.** A setup that stopped before it could say why (Windows refusing the
  setup script, a Python error) left the step before it on the screen. Every end now gives its reason: an NVIDIA
  driver too old for PyTorch is named with its version, and a PyTorch that does not load names the Visual C++
  runtime it needs. Everything the setup printed is kept in `genfill\setup.log` in RUDRA's data folder, and
  **Copy log** puts it on the clipboard. Closing the window while the setup runs asks first, then stops all of it.
- **AI fill setup stopped before downloading Wan2.1-VACE, and showed nothing while it ran.** The setup script handed
  its last option to the model step one letter at a time, so the step refused it and the base model never downloaded.
  It is handed over whole now. The window also said nothing until the first step (a slow start looked like a hang),
  and nothing during the 20 GB download when its progress bars did not come through: it now says at once that it
  started and where its log is, names each check as it runs, and gives the GB downloaded and the speed every
  15 seconds. A RUDRA repository saved by an older RUDRA (its Python left in Downloads, say) no longer runs instead
  of this RUDRA's own.
- **AI fill's base model and Python in Install selected.** In Check this computer, "AI fill: base model and Python"
  now has a tick like the other downloads: ticked, **Install selected** sets up the Python and downloads
  Wan2.1-VACE-1.3B (about 23 GB in all) along with the ticked models, in its own window that shows each step and the
  log. Ticking it ticks RUDRA's AI fill model too. It is never ticked by itself, and only offered where AI fill can run.
- **Install from a folder or zip** works on a computer without AI fill's Python: it sets the Python up first (about
  5 GB from the internet), then installs the model from the folder or zip with no download.
- **AI fill in the installed app.** The AI fill preview and the region re-render failed with
  "ModuleNotFoundError: No module named 'tools'" in every installed RUDRA since 1.3.0, so pressing AI fill seemed
  to do nothing. They read the source through a helper that only a source checkout has. The helper is now part
  of RUDRA itself, a test loads AI fill the way the installed app does, and the Windows build stops if any of
  RUDRA's Python imports from outside what it ships.

- **AI fill in Resolve and Nuke never drew.** Once every frame was in, the fill stopped before it began ("the AI
  fill ended without finishing"): on Windows the plug-in hands the fill its log file already open, and the fill
  opened it a second time, which Windows refused. The fill now writes to the log it is given, and no longer takes on
  a host's own Python settings (PYTHONHOME, PYTHONPATH). A test starts it the way the plug-in does.
- **In a color-managed Resolve project RUDRA seemed to do nothing.** Such a project hands a node its HDR working
  picture, not the SDR clip: the clipped sky of the test shot arrived flat at code 195 of 255, so RUDRA found nothing
  to recover and AI fill nothing to draw. RUDRA now takes part in OpenFX colour management (Resolve's Enable OpenFX
  Color Management): it asks for the clip as Rec.709 display codes and tells Resolve what its Output is, so Resolve
  converts in and out of its working space. Where a host hands it a picture whose highlights stop flat below white
  anyway, RUDRA says so and what to do. Help › Install says so too.
- **A finished fill was not shown in Resolve.** RUDRA checked that the picture coming in is the one the fill was
  made from with an exact fingerprint of its codes, and Resolve decodes a paused frame a little differently from the
  same frame played. It now compares a small thumbnail with a tolerance (1.5 codes on average), so a changed grade
  above RUDRA is still caught, and says by how much a picture changed.
- **AI fill drew nothing without saying why.** In a picture where nothing clips (no code of 240 or more), AI fill has
  nothing to draw. The picture now says so before Generate, with the frame's brightest code, and a fill that found
  nothing to draw says so too. A To set before the first frame made a fill of a single frame; Generate now refuses it
  and says why.
- **AI fill in Resolve collected no frames.** Resolve numbers a clip's frames by their source media and does not
  tell a plug-in where a trimmed clip starts, so every frame fell outside the range AI fill waited for and the
  status stayed at "0 of n". In Resolve, AI fill now takes the frames from From (else the playhead) to To (else the
  clip's end): put the playhead on the first frame to draw, then press Generate.
- **AI fill in the plug-in left you guessing.** Its state was only in the Status line, which changes only when a
  button is pressed and which Resolve may not show. Until the fill is shown, the picture itself now says what to do
  and how far it got, in a line of words over a bar along its bottom: green while frames come in ("12 of 97 frames
  in. Play the clip through once on the Color page"), blue while the fill is drawn ("drawing, 37% done. Keep Resolve
  open; press Refresh status to update this"), red when something needs you (Generate refused and why, a fill that
  failed and what to do, a frame that changed since Generate). A ready fill, and a frame the fill was not made for,
  show clean, so nothing is burned into a render. Refresh status says which frames are still to play, and the
  plug-in's log records each stored frame, what the picture said, and what the host said about its frame numbers
  at Generate.
- **AI fill beside a host could run out of GPU memory.** The fill was set up for a whole 16 GB card, and Resolve or
  Nuke hold part of it. It now chooses its memory mode from the memory free when it starts: as tested with room for
  it, else a mode that draws the same fill in 10.8 GB instead of 16.2, else one that needs 6.7 GB. If it runs out
  all the same, it draws again in low-memory mode by itself, and if that fails too the picture says to close other
  programs that use the GPU.
- **Nuke: "no picture from the input" after a knob moved.** Dragging a knob above RUDRA (a Grade's gain) while a
  frame rendered made Nuke abandon that render without a picture, and RUDRA put an error on the node that could stay
  after the picture came back. An abandoned render now ends quietly, and the node's message is always the latest
  render's.
- **A full disk was found out late.** Generate, and the clip's first frame, now check there is room for the clip's
  stored frames and fills (about 30 MB a frame at 1080p, and 1 GB spare) and say how much is needed when there is not.
- **A learned engine that hung froze the host.** If RUDRA's learned engine did not load in 5 minutes or answer a frame
  in 2, every render waited on it, and so did closing Resolve or Nuke. It is now ended with an error on the node, and
  the next frame starts it again; closing the host gives it 2 seconds.
- **The learned route and AI fill no longer fight over the GPU.** While a fill draws, a frame on the learned route
  (a proxy render, another instance) loaded the learned model onto the same GPU. It now shows the analytic master
  and says so until the fill is done, and such a frame is not stored for a fill.
- **Two renders of one frame at once** (the viewer and a render, two instances) could leave one render's codes
  beside the other's master, or fail to store the frame on Windows. Each writes files of its own and puts them in
  place together; the fill's status file is written the same way.

## Also

- The user guide lists the system requirements: the computers and GPUs RUDRA runs on, and those AI fill needs
  (an NVIDIA GPU with 16 GB and driver 570 or newer, or Apple silicon with 16 GB), with recommended cards.
- `RUDRA.ofx` carries a version resource, and the release's signing now covers it and AI fill's setup script
  (`python\genfill_bootstrap.ps1`) along with RUDRA.exe and rudra-native.exe, from the first release signed through
  SignPath Foundation (above). Until then, a computer whose policy runs only signed PowerShell scripts refuses the
  setup script, and Set up automatically says so instead of stopping without a reason.
- Release builds: when the app tests crash on macOS, the log now shows where, and a test run can be repeated to
  catch a crash that comes and goes (1.3.1's macOS build crashed twice before it passed).

## Tested

- OFX: the plug-in's tests (47 of 47: the picture, the learned engine, AI fill, Suggest prompt, Prompt strength) and
  a test host that loads the bundle and renders through it, with the mask, Suggest prompt, an abandoned render and a colour-managing host (9 of 9). App tests
  for AI fill, Deliver, Batch, the QC report and Help: 68 pass; 6 skip because they need the learned model loaded.
- Python, AI fill and the plug-in's helpers with PyTorch: 149 pass.
- AI fill in an installed layout: `rudra.genfill` loads with only RUDRA's shipped folders on the path.
- Prompts on the gate shots (33 frames each, the analytic route): no prompt, 1.3.1's and 1.3.2's wording, one pass
  against the blend, strength 1.5 and guidance 2. 09V passes every check in every variant. 04 fails the flicker
  check in every variant, prompt or not (0.22 stop against 0.10), as in 1.3.1 (below).
- In Resolve and Nuke, as an end user would: RUDRA installed from its installer, AI fill set up in it, the plug-in
  from Help, then in Resolve the effect, Generate, frames coming in, drawing, ready, the whole clip, out of date and
  Deliver, and in Nuke the node, Generate and play, ready, proxy and Cancel: 18 of 18 on an RTX 4080 SUPER (16 GB).
- AI fill setup (Python): a download that stopped carries on, a model of another size or revision is refused and not
  downloaded over, a setup that breaks still says why, the driver, Visual C++ and GPU memory reasons, nvidia-smi read,
  only the missing packages installed. The Windows setup run from a new folder, as Set up automatically runs it:
  uv, Python 3.12.13, its environment, PyTorch 2.11 (CUDA 12.8) and the packages in 7 minutes, with no junction or
  other link anywhere under it; it then found the test machine's model three small files short and set out to fetch them.
- EXR: a linear Rec.709 EXR frame decodes to its BT.1886 codes; a numbered sequence is found from one of its frames.
- Deliver's progress bar: a render shows its count and ends with Done and Open folder; Stop from the bar keeps the
  frames written and starts nothing when nothing runs.

## Known limits

- AI fill (in the app and in the plug-in) is experimental. In the plug-in it draws about 150 frame-steps a
  minute on an RTX 4090 (100 frames at 30 steps took about 20 minutes); each new prompt first takes a while to
  read.
- The Mask input is Nuke's; Resolve's filter context has none (use Resolve's own window or matte there).
- The limits listed for 1.3.1 still apply.
