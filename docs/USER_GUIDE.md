# RUDRA user guide

RUDRA turns SDR footage (stills, frame sequences, movies) into scene-linear
HDR: an ACES 2065-1 EXR master for grading, or HDR10, HLG and ProRes movies
for delivery. It always starts from an analytic inverse of the SDR's tone
curve (the *baseline*), fills what the SDR clipped, and can add a learned
model's reconstruction on top (optional, off by default), and it shows you
where it made something up.

This guide is for the desktop app, RUDRA 1.3.2. The illustrated guide is at
https://www.fxtdstudios.com/rudra-docs/. Help > Getting started shows the short
version inside the app; F1 opens this page.

## 1. Install

### System requirements (1.3.2)

**RUDRA** (the reconstruction, the learned ACES 2.0 model, every export):

| | |
|---|---|
| Windows | 10 (22H2) or 11, 64-bit. Windows on Arm runs the x64 build through emulation, more slowly. |
| macOS | 13 or later on Apple silicon (M1 or later). Intel Macs are not supported. |
| Linux | Built from source; not released. |
| GPU (Windows) | Any DirectX 12 GPU (NVIDIA, AMD or Intel): the learned model runs on it through DirectML. Without one, on the CPU, more slowly. |
| GPU (Mac) | The built-in GPU and Neural Engine (Core ML). |
| Memory | 8 GB or more; long 4K movies need more. |
| Display | An HDR monitor is recommended for judging the result; RUDRA simulates HDR on an SDR screen. |

**AI fill** (optional, experimental):

| | |
|---|---|
| Windows | An NVIDIA GPU with 16 GB of memory or more, and an NVIDIA driver of version 570 or newer (AI fill's PyTorch is the CUDA 12.8 build). AMD and Intel GPUs do not run it. |
| Recommended cards | RTX 30, 40 or 50 series and the professional Ampere, Ada and Blackwell cards, with 16 GB or more: for example RTX 3090, 4060 Ti 16 GB, 4070 Ti Super, 4080, 4090, 5070 Ti, 5080, 5090, RTX A4000 to A6000, RTX 4000 / 5000 / 6000 Ada. Tested on an RTX 4090. |
| Older cards | Turing (RTX 20, Quadro RTX) and Pascal (Quadro P5000 / P6000) are untested and not recommended: AI fill runs in bfloat16, which they do not do natively, and current PyTorch may not support Pascal. |
| Mac | Apple silicon with 16 GB of memory or more. |
| Disk | About 23 GB for its setup (its Python and PyTorch, and Wan2.1-VACE-1.3B), and room for each fill (about 30 MB a frame at 1080p). |
| Speed | On an RTX 4090, 100 frames at 30 steps take about 20 minutes. |

Check this computer (below) reads your computer and says what runs on it. On a
computer that cannot run AI fill, everything else in RUDRA still runs.

**The OFX plug-in:** DaVinci Resolve (tested with Resolve 21) and Nuke, on
Windows ([OFX_PLUGIN.md](OFX_PLUGIN.md)). AI fill in a host needs the GPU above,
shared with the host.

### Installing

- **Windows 10 or 11 (x64):** run `RUDRA-<version>-windows-x64-setup.exe`
  (installs for your user, no administrator prompt), or unzip the portable
  ZIP anywhere and run `RUDRA.exe`. ffmpeg comes with the package.
- **macOS 13 or later (Apple silicon):** open the DMG and drag RUDRA to
  Applications. Movies need an ffmpeg with `libx265`, `prores_ks` and
  `zscale` on the PATH (`brew install ffmpeg`) unless the DMG says it carries
  one.

**The first start.** *Welcome to RUDRA* checks the display (HDR or SDR, and
its real peak brightness, shown on an HDR test card), names the GPU it found
(with or without a model), and, if a model package is installed, loads and
tests it on the backend you pick (**Load and test**).
The model is optional: **Continue** or **Start on CPU** both get you to work,
and nothing runs a model until Reconstruct > Use the model is on. Then
*Check this computer* opens (below), then *Getting started* shows the five
steps, once (and once again after an update that changes them). Help > Check
the display and the model runs the welcome again; Help > Getting started
shows the steps.

**Check this computer.** It opens after the welcome, and again after each
update; the shield button in the toolbar (with a dot when something wants a
look) and **Help > Check this computer…** open it any time. It says, in plain
words:

- whether this computer can run RUDRA: the system, the GPU and its memory,
  memory, free disk where models are kept, ffmpeg for movies, and whether the
  model list could be read. A blocker is red; a warning (no GPU, little
  memory) still lets you work;
- what RUDRA can install, with size and licence: the learned ACES 2.0 model
  (recommended, ticked) and AI fill's model, and AI fill's base model and
  Python (1.3.2: tick it and **Install selected** sets it up with the rest,
  about 23 GB, in its own window; or **Set up…**). AI fill needs a 16 GB NVIDIA GPU (on a Mac, 16 GB of
  memory); on a smaller card it is not offered, and the window says why.
  **Install selected** downloads the ticked models one after the other while
  you work;
- what RUDRA put right by itself: an old Beta 2 model it no longer opens
  (**Move aside** puts it in a `.retired-beta2` folder), and Reconstruct >
  Use the model turned off when no reconstruction model was installed.

**Copy report** puts the whole check on the clipboard for a support request.

The viewer opens as a comparison: a wipe with the SDR original on the left
and RUDRA on the right, at a 1,000-nit display peak. A project opens at the
peak it was saved with.

## 2. The five steps

### Source: open the shot

The Source card reports movie dimensions, pixel format, primaries, transfer,
matrix and range from the file headers. Missing tags say "not reported";
these are reported tags, not a measured tone curve. Limited-range movies
show the current super-white decoding limitation (not a claim that such
pixels were detected), and PQ/HLG inputs show an HDR-source warning.
Still/sequence inspection shows the current decoded frame's recognised profile
and primaries, any Rec.709 fallback, and applied EXIF orientation. It updates
when you change frames; unavailable information stays explicit. Profile names
are displayed as plain text. This inspection does not override decoding or your
grade and does not identify the original tone mapper.

- **File > Open frames…** (O) adds stills; **File > Open folder of frames…**
  opens a numbered sequence; a movie (`.mov`, `.mp4`, `.mxf`, `.mkv`, …)
  opens as a shot of its own. Dropping files or a folder on the window does
  the same.
- **Media, on the left** (1.3.2): right-click anywhere in it for **Import…**
  (videos and images; tick **Image sequence** and the numbered frames of a
  sequence come in as one clip), **Import a folder of frames…**, **Clear
  from Media** (everything, or only the videos, images or image sequences;
  the files stay on disk) and **Clear VRAM** (frees the AI fill preview's
  model and the learned model's cache; not while a render or a fill runs).
- **File > Import image sequence…** (1.3.2): pick any one frame of a
  numbered sequence (`shot_0001.exr` to `shot_0240.exr`, say) and RUDRA
  opens every frame of it, in order, as one shot. Frames of the sequence share
  the name before and after the number and the extension; other files in the
  folder stay out. EXR, PNG, JPEG, TIFF, BMP and WebP.
- **EXR frames** (1.3.2) are read as a linear Rec.709 picture of an SDR
  display: values above 1.0 are clipped and the rest are encoded with the
  Rec.709 display curve (BT.1886, gamma 2.4), like the SDR frames RUDRA is
  made for. An EXR that is already HDR (scene-linear light above 1.0) loses
  what is above 1.0: RUDRA rebuilds clipped SDR, it does not grade HDR.
- **Stills keep their colour and orientation.** A still's own colour tag
  (an ICC profile, a PNG cICP / sRGB / cHRM chunk) sets the primaries the
  master converts from: Display P3 (phones), Adobe RGB (1998), Rec.2020 or
  sRGB / Rec.709. Untagged files are read as sRGB / Rec.709, as before. A
  profile RUDRA does not recognise is read as sRGB / Rec.709, and you should
  check the colours. The EXIF orientation of a phone or camera photo is
  applied, so portrait shots come out upright. The viewer still shows the
  picture as Rec.709; the master converts correctly.
- **Limited-range video above white** (codes 236–254, "super-whites") is
  clipped at decode in 1.0, and Fill clipped highlights rebuilds it. If that
  detail matters, keep the original and validate a preprocessing workflow
  that maps headroom into the supported range. Merely switching the range
  tag to full does not guarantee preservation (docs/SUPER_WHITES_DESIGN.md).
- **Reconstruct > Source** says how the SDR was made. This matters more than
  anything else: the baseline is the inverse of that curve.
  - *BT.1886* (the default): SDR as a display shows it, each code to the
    power 2.4, white at 203 nits. Right for graded and broadcast video, and
    the best measured on real SDR by far (docs/RESULTS.md). On a still read
    as sRGB the mid-tones come out about a third of a stop darker.
  - *BT.1886 shadows* (opt-in): BT.1886 from SDR code 40 up, with the blacks
    real HDR masters keep below it. Measured on five Netflix Open Content
    titles, real graded HDR sits 2–7 stops above code^2.4 in deep shadow.
    Try it when the default looks crushed in the darks. On held-out real SDR
    it scored +0.04 to +1.3 dB PU21 over BT.1886, but on very dark scenes it
    can lift blacks about half a stop above some grades, so judge it on an
    HDR display (docs/ai/SHADOW_CORRECTION_RESULT.md).
  - *BT.1886 shoulder* (opt-in, 1.3.1): BT.1886 up to SDR code 165 and,
    above it, the highlight shoulder an ACES-style render puts on video
    rolled back out: +0.1 stop at code 200, +0.5 at 222, +1.25 at 242 and
    +2 stops at white. Lamps, windows and the bright side of a face get
    light above SDR white instead of stopping at it, and Fill clipped
    highlights builds on top. It is one curve for every shot: the learned
    ACES 2.0 route measured +1.7 stops at code 242 on a welding shot and
    +0.5 on a sunrise. Large blown areas get brighter but no more detail:
    judge them at -3 EV, and use AI fill or a lower Strength there.
  - *Unknown (ACES inverse)*: RUDRA's default before 13 Oct 2026; projects
    saved with it keep it.
  - *ACES*, *filmic (Hable)*, *AgX*: CG renders and games with those tone mappers.
  - *Camera log*: a log clip viewed through a LUT you know.
  - *sRGB, plain clip* (called *Rec.709, plain clip* before 1.3.1): the SDR's
    codes read as sRGB, the light a still or a CG render clipped at white
    holds. A master made with it lines up with the SDR read as sRGB, as Nuke's
    **sRGB Encoded Rec.709 (sRGB)** reads it.

**Generated video** (text-to-video and image-to-video tools, renders from
game engines, upscaler output): these are 8-bit SDR with clipped highlights
and no real camera curve. Start with the default, *Source: BT.1886*; if the picture has a
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

- **Analytic or AI-assisted.** By default RUDRA reconstructs analytically: the
  inverse of the source curve with your calibration, reference and grade; no
  model runs and none needs to be installed. **Reconstruct > Use the model
  (AI-assisted)** adds the model's reconstruction on top; the status bar says
  which is in use and every master's sidecar records it. Movie exports
  follow it too, with your whole grade.

- **Learned ACES 2.0, and when not to use it.** The learned ACES 2.0 model
  inverts the ACES 2.0 SDR output exactly, so its master renders back to the
  SDR you delivered. Use it for SDR that was made through ACES 2.0 (set the
  Source curve to ACES). On a grade made another way, bright areas parked just
  under white, an open door, a window, a sky, come back far brighter and more
  saturated than they were, and the Learned ACES 2.0 trims do not reach them;
  the card and the Deliver page say so and point at the analytic route
  (Deliver > Source > Analytic). From 1.2.1 the route opens on the **HDR
  1000** monitor (the ACES 2.0 HDR output, a deliverable's picture) rather
  than *Scene* (the unbounded master through exposure and clip), and an AI
  fill preview's Fill and Master pictures go through the same monitor as the
  live frame.

- **Exposure.** The slider at the top of the Reconstruct card moves the whole
  master brighter or darker, -3 to +3 stops, live on the viewer. It is part of
  your grade: it survives the anchor (mid-tones keep the offset), the Region
  EV bands see the exposed picture, and masters, movie exports and projects
  carry it. Ctrl+Shift+Up / Down: a third of a stop; double-click: back to 0.

- **Clipped highlights.** Where the SDR clipped (a blown sky, a lamp, the
  sun), the inverse alone leaves a flat ceiling. **Reconstruct > Fill clipped
  highlights** (on by default) puts something sensible there with no model:
  a channel that clipped while the others did not is rebuilt from them, a
  blown sky gets a smooth fill from its edge, and a lamp or the sun a peak
  and a falloff from the glow around it. Nothing is made up as texture, and
  the *Invented* layer still shows every value it set. **Set the peak of a
  clipped region…** then a click on the region asks for its peak in nits
  (0 removes it); **Clear clipped peaks** removes them all. Strength scales
  the fill, *Off* removes it. With the model on, the model fills instead.
  Where a highlight was only just above the clip the fill can overshoot (by
  about a quarter of a stop), and small bright sparks such as fireworks look
  wrong: lower Strength, set the peak, or turn the fill off for those shots.
- **Highlight detail** (1.2.1). An SDR's shoulder parks a bright sky, a
  window or an open door just under white with its structure flattened to a
  few codes, and the inverse shows it as a flat wall. The slider amplifies
  the structure that survives there (1.00 leaves it as the source has it;
  1.5-2 is the range to judge), the same on every channel so the hue holds,
  and leaves one code of noise alone. It is part of the clipped-highlight
  fill: off with it, and only on the analytic route. Where the source is
  truly flat (every pixel at the last two or three codes) there is nothing
  to amplify, and it changes nothing; that is the AI fill's territory.
- **Mode** (1 2 3 4): what the model may change: *All*, *Highlights*,
  *Shadows*, or *Off* (the baseline alone).
- **Strength** ([ and ]): how much of the model's residual is added.
- **Limit to a luminance region** (1.1): confine the reconstruction to a
  range of the SDR's brightness. *Highlights + shadows* gives each its own
  threshold (Highlights from, Shadows below, as SDR luma 0 to 1), softness
  and strength; *Band* reconstructs only between Band low and Band high.
  Every pixel outside the region keeps the analytic baseline exactly, bit
  for bit, in the viewer, the master and movie exports; the master's sidecar
  and EXR record the region (`reconstruction_region`). The region only ever
  narrows what Mode allows. Not available with Keep super-whites. Under the
  sliders a **histogram** of the frame on screen shows its brightness with the
  region's weight drawn over it and how much of the frame the region reaches;
  **Show on the viewer** tints the region on the picture as you move the
  sliders (and on every frame you scrub to).
- **Preserve outside masks**: with painted masks, keep everything outside
  them at the baseline.
- Hold **B** to flip to the baseline; **W** wipes baseline (left) against
  RUDRA (right); drag the image to move the wipe.

### Play

- **Space** plays. With the model on, it runs ahead of the playhead on a
  smaller copy of each frame (**Clip > While playing**: 960 px, 640 px or full
  size) and the frame you stop on is reconstructed again at full size.
- **Keep time** (on by default) skips frames the model has not reached so the
  clip plays at its own rate; turn it off to see every frame. The status bar
  shows the rate reached, and the log says it when you stop.
- **Clip > Hold steady within a shot** (on by default) keeps the values RUDRA
  works out for a whole frame (the model's shadow weight, and the anchor's
  hold) steady across a shot, so the picture does not pump with grain or
  busy content; they start again at each cut. Movie exports do the same, so
  what plays is what exports. A stopped frame shows its own values, as Master
  EXR does.
- Movie exports also hold still areas steady: where the source pixel did not
  change, how much RUDRA brightened it is held, so grain lifted by the
  expansion stops shimmering (anything that moves is left to each frame).
  Playback does the same while Hold steady within a shot is on.

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

#### Colour-managed viewer (OpenColorIO, as in Nuke)

The viewer toolbar shows the picture the way Nuke's viewer does, so you do not
need to open the master in Nuke to check it:

- **View** (the first box): *RUDRA view* (absolute nits, no tone mapping, as
  before) or an OpenColorIO display and view: SDR sRGB, SDR Rec.709 (BT.1886),
  HDR Rec.2100 PQ 1000 nits, P3-D65, Raw (the working-space values as they
  are), then every display and view of the config. On an SDR monitor the
  view's code values are shown as Nuke shows them; on an HDR monitor their
  light, so PQ 1000 shows at its real level.
- **ACEScg / ACES2065-1 (AP0)**: the working space, as a render to that space
  holds the picture (1.0 = diffuse white, 203 nits). The probe shows the
  values in it, and the view's 10-bit codes.
- **Original** (*SDR shown as*): how the SDR comes in when you wipe RUDRA
  against it through an OCIO view. The default, *sRGB texture*, reads its codes
  as linear light, as Nuke's Read does with **sRGB Encoded Rec.709 (sRGB)** in an
  ACES 2.0 config (since 1.3.0). The inverse of the ACES 2.0 SDR view makes the
  original look as it was graded instead; through the HDR view that is the plain
  inverse-view up-conversion. See *Checking a master in Nuke* below for which
  reading lines up with which master.
- **Gamut** (Alt+G): magenta bands where the picture is outside the gamut on
  the right (AP1 is what an ACEScg render holds without negative values).
- **Invalid** (Alt+I): solid green on NaN or infinite values, colours outside
  AP0, and values above 65,504 (what a half-float EXR can hold).
- **Clipping**: through an OCIO view, where the view itself clips.
- **View › Colour management (OpenColorIO)…** picks the config: the built-in ACES 2.0 studio
  config, `$OCIO`, or a `config.ocio` of your own.

None of these changes the master.

#### Checking a master in Nuke

RUDRA's EXR masters are ACEScg (AP1 primaries in the file's header; ACES
2065-1 when you chose AP0), with 1.0 = diffuse white (203 nits). In Nuke with
an ACES 2.0 config (Nuke's own `fn-nuke_studio-config-v3.0.0_aces-v2.0_ocio-v2.4`,
for example):

1. Read the master with Input Transform **ACEScg** (that config's float-file
   role, scene_linear, already is), or **ACES2065-1** for an AP0 master.
2. Read the SDR, put it on A and the master on B, and wipe. Viewer:
   **sRGB - Display / ACES 2.0 - SDR 100 nits (Rec.709)**. RUDRA's viewer does
   the same wipe under View › Colour.
3. How the SDR is read decides what should line up:
   - **sRGB Encoded Rec.709 (sRGB)** (RUDRA's viewer default, a common Nuke
     setup) reads the SDR's codes as linear light. Next to it, an analytic master
     of a video made with Source BT.1886 (the default for video) is darker in the
     shadows, by design: about 0.17 stop at code 128, 0.66 at code 51 and 1.3 at
     code 26. BT.1886 is how a display shows video (code^2.4), and sRGB is a
     different curve. To make a master that lines up with this reading, choose
     Source *sRGB, plain clip*. A learned ACES 2.0 master looks as the SDR was graded, so it
     looks brighter than an SDR read this way.
   - **Gamma 2.4 Encoded Rec.709** lines up with an analytic master made with
     Source BT.1886, everywhere outside the highlights.
   - **The inverse of the ACES 2.0 SDR view** lines up with a learned ACES 2.0
     master, everywhere outside the highlights. In Nuke: Read the SDR with
     **Raw Data** on, then an **OCIODisplay** node with display **Rec.1886
     Rec.709 - Display**, view **ACES 2.0 - SDR 100 nits (Rec.709)** and
     **invert** on.

   Where the SDR was clipped, the master holds more: that is the
   reconstruction.
4. For HDR, view **Rec.2100-PQ - Display / ACES 2.0 - HDR 1000 nits (P3 D65)**
   on an HDR monitor.

Reconstruct › Source › sRGB (1.3.1) makes an analytic master line up with an SDR
read as sRGB.

### AI fill: the viewer, Reconstruct and Deliver

The AI fill is in three places, each with one job:

- **Viewer**: one **AI fill** button on the viewer's bar turns the AI fill on
  (blue, **AI fill on**) or off (grey, **AI fill**) in one click: on, the
  viewer shows the fill wherever there is one; off, the master, to compare.
  It is on the bar whenever a shot is open (in Full), and it is the same
  switch as Reconstruct › AI generation.
- **Reconstruct › AI generation** (experimental): its switch turns the AI
  fill on for the session. It starts on when the computer can run the fill
  (an NVIDIA GPU with 16 GB of memory, or Apple silicon with 16 GB, and the
  AI fill installed) and off on any other computer. Below it is everything else: what it
  regenerates (**AI fill v3** or **v4.3 region**, **Keep small and thin bright
  parts**, **Where it fills**, and the thresholds in **Advanced**), the prompt
  and **Suggest a prompt…**, **Preview AI fill**, **Show what it invented**,
  **Setup…**, **Review fill…** and the settings described below.
- **Deliver › AI Fill Generation (Experimental)**: the only AI fill control in
  Deliver. Ticked, the export includes the fill, made with Reconstruct's
  settings. It is greyed out until AI generation is on in Reconstruct.

### Deliver

- **AI fill** (experimental; Deliver's box starts unticked; set up in Reconstruct › AI
  generation, put in the export with Deliver › AI Fill Generation) draws detail inside fully
  clipped areas. Optional **Highlight / Shadow prompts** steer what it draws
  in each region (only there). **Prompt strength** (1.1, 0 to 2) sets how
  strongly the prompts steer: 1 is how AI fill v3 was tested, 0 ignores them,
  above 1 is experimental. **Detail strength** (0 to 2) sets how much the
  drawn shape varies the brightness; 0 is a flat fill. **Suggest a prompt**
  (1.1) looks at the frame on screen: the size, place and shape of its fully
  clipped areas and the colour just around them (and, on a moving shot, how
  much a warm area changes) and offers a prompt with the reason, beside the
  presets (clear blue sky, overcast sky, sunset sky, sun, lamp, window
  daylight, specular glints, flame, shadow detail). Picking one fills the
  prompt box; you can still edit it. No model runs for the suggestion. A
  preset marked *ready* is already embedded, so the fill starts without
  loading the 11 GB text model; `scripts\GENFILL.ps1 -PrecachePrompts` embeds
  them all once. (1.3.2) The suggestions describe more of what the area looks
  like and end with the words the AI fill model was trained under
  ("scene-referred high dynamic range footage, physically plausible
  highlights"), so a prompt steers what is drawn without leaving the
  conditions the model learned in. A new prompt is read in seconds, not
  minutes: the text model reads only the prompt's own words. At Prompt
  strength 1 a prompt no longer slows the fill (one pass of the model a
  step); another strength takes about a third longer. With **Limit to a
  luminance region** on (Reconstruct), the fill acts only inside that region
  and every pixel outside it is the master's, bit for bit. No 8-bit code of
  the SDR view ever changes, and every drawn pixel is marked in
  `rudra.generated`.
- **Region re-render (v4)** (1.2.4, experimental) is AI fill's second mode,
  with its own model. Where v3 draws only inside fully clipped pixels, v4
  draws a whole bright region again: a window, an open door, a sky parked just
  under white. Choose it with the **AI fill v3 / v4.3 region** switch in
  Reconstruct › AI generation. Type what is in the bright
  area in **Prompt** ("overcast sky over the sea, soft grey clouds"); with no
  prompt it draws texture with no subject. **Advanced** holds the seed and the
  steps. **Preview AI fill** fills 17 frames round the playhead and shows the
  frame on screen; the viewer's **AI fill** button switches between the fill
  and the master, and **Show what it invented** shows the invented pixels.
  (1.2.7) What the fill draws is above SDR white, so a finished preview opens
  with the wipe off and the view exposure at −2 EV (only when it was at 0); a
  line on the AI generation card says so, **Alt+E** goes back to 0 EV and **W** brings
  the wipe back. **Where it fills** shows what v4.3 really redraws: the
  near-white region in solid orange, and tinted what stays the master (a
  saturated highlight such as a sunset or a flame is never redrawn). **Keep
  small and thin bright parts** (1.2.7, off by default) leaves a floor
  grating, a railing, a practical lamp or a glint to the master.
  Inside the region the SDR's codes are not kept, unlike v3; outside it every
  pixel is the master's. Tested on the analytic route (Source > Analytic). In
  a smooth glare (a hazy sun) it sets the brightness and draws no texture; on
  fast-moving water it can flicker slightly.
- **Master EXR** (M): ACES 2065-1 (or linear Rec.2020) EXR, a frame, all
  frames or a range, with a JSON sidecar describing exactly how it was made.
- **The render's progress bar** (1.3.2), under the Render card: what is
  being written, frames done of all, the share and the time left; **Stop**
  stops after the frame being written and keeps what was written. At the end:
  "Done: … frames in …" and **Open folder**. It is this shot's render only;
  the Batch queue keeps its own.
- **Containers** (1.1): the analytic master writes ACES 2065-1, ACEScg,
  linear Rec.2020 or linear Rec.709 (Deliver › Gamut).
- **EXR compression** (1.3.2): every OpenEXR compression, on the analytic and
  the learned route. Lossless, every value kept: ZIP (the default, about half
  the size), ZIPS (one line a block), PIZ (smallest on grainy pictures), RLE
  (fast, for flat areas) and None. Lossy, much smaller, for reviews and
  dailies rather than the grade: DWAA, DWAB, B44 and B44A (half float only)
  and PXR24 (32-bit float kept to 24 bits). Masks and mattes stay lossless.
  An ACES 2065-1 file can be compressed too, but then it is not the SMPTE
  ST 2065-4 container, which is uncompressed: choose None when a facility asks
  for the strict container.
- **AI fill on the analytic route** (1.1): with the analytic master in ACES
  2065-1, AI fill runs after the render as it does on the learned route, so
  the luminance region, its mattes and the prompts work on one render.
- **Write mattes (EXR)** (1.1): beside each analytic EXR master (for a
  movie: one per frame in `<name>_mattes`, beside its frames folder),
  `<name>.mattes.exr` with float channels for compositing:
  `rudra.highlight` and `rudra.shadow` (how much each reconstruction acted),
  `rudra.changed` (1 exactly where the reconstruction differs from the
  baseline), `rudra.region` (the luminance region, when on) and
  `rudra.paint.N` (each painted mask). They describe the reconstruction
  before the grade (Region EV and exposure). The AI fill's EXRs split
  `rudra.generated` into `rudra.generated.highlight` and
  `rudra.generated.shadow`, which add up to it exactly.
- **Write EV brackets** (1.1): beside each analytic EXR master,
  `<name>.ev-2.png`, `.ev+0.png` and `.ev+2.png`: the master as written, read
  back and shown on an SDR screen with diffuse white (203 nits) at 0 EV.
  −2 EV shows the highlights up to about 800 nits, +2 EV the shadows. The
  sidecar's `brackets.preview_vs_export` says how far the written master is
  from what the viewer showed, in 8-bit codes (0 for a float32 master, at
  most 1 for a half master).
- With **Write mattes**, also `<name>.regions.png`: every pixel's class over
  a grey picture: grey preserved (bit for bit the baseline, i.e. the source
  decoded) or within 0.1 stop of it; cyan reconstructed (the SDR has detail
  there); magenta estimated (the SDR is clipped or crushed, so the value is
  an extrapolation); amber generated by the AI fill; white lines the edge of
  the luminance region. The mattes EXR adds `rudra.class` and
  `rudra.confidence` (1 where the SDR pins the value, falling as the value
  moves away from it, at most 0.5 where the SDR only bounds it, 0 for AI
  fill: a constraint score, not a probability), and the sidecar the share of
  each class (`pixel_classes`). In the viewer, the **Invented** layer shows
  the same reconstructed / estimated split live.
- **More detail in the filled area** (AI fill, off by default): the fill model works at about
  832 × 480. On, it sees only a box around the shot's clipped areas (one box for the whole
  shot, so the fill still moves as one picture) instead of the whole frame, so a blown sky in a
  door a third of the frame wide is drawn at roughly twice the detail. Small specks far from the
  main area keep the master there; `genfill.json` gives the box and the share left out
  (`focus`). Off is AI fill v3 as tested.
- **Drop flickering fill frames** (AI fill, on by default): a frame whose
  fill jumps more than a quarter stop into it and out of it is written as
  the master's own frame; a fill that starts or ends is kept. A fill that is
  not a finite number always falls back. `genfill.json` lists every
  fallback (`fell_back`).
- **Technical details** (1.2.4): the range line under the viewer's tools, the
  pipe bar under the picture and the frame's timings show when **Help >
  Technical details** is ticked. RUDRA starts without them.
- The pipe bar under the viewer ends with the **display signal**: the
  transfer (sRGB, scRGB linear, PQ or EDR linear), primaries, the reference
  white (203 nits) and the display's peak.
  The EXR and the sidecar both record:
  - the app version and build commit (`rudra:appVersion`,
    `rudra:buildCommit`);
  - the source file's SHA-256 (`rudra:sourceSha256`);
  - with the model on, the checkpoint's SHA-256 (`rudra:checkpointSha256`);
  - whether the network ran in tiles (`rudra:tiled`).
- For a movie, the export sheet adds **HDR10**, **HLG**, **ProRes 422 HQ** and
  **ProRes 4444** (graded PQ, or scene-referred ACEScct / ARRI LogC4) with
  the audio. Exports run in a queue you can stop and resume (the export
  sheet's Queue button shows it).
- A colour too saturated for the movie's peak (a clipped blue sky can carry
  more than 1,000 nits of blue at a few hundred nits of brightness) keeps its
  brightness and loses a little saturation, so the movie is as bright as the
  master. `rudra-native deliver --peak-fit scale` scales such a pixel down
  whole instead (the older behaviour: hue and saturation kept, darker).
- **Anchor to source exposure** keeps mid-tones where the SDR had them, read
  through the Source you picked (BT.1886 reads the SDR as a BT.1886 display
  shows it) and your calibration or reference; the **Anchor knee** says where
  that lets go.
- HDR10 and ProRes (PQ) exports write their **Dolby Vision and HDR10+
  metadata** per shot beside the file: `<file>_hdr10plus.json`,
  `<file>_dovi_generate.json` (for `dovi_tool generate` or your finishing
  tool), `<file>_hdr_analysis.json` and `<file>_hdr10plus_scenes.json`. HDR10
  carries HDR10+ inside the file when the ffmpeg's x265 can write it; the
  report (`hdr_metadata`) says whether it did and, if not, why.
- A movie's report (`<file>.json`) measures flicker: `flicker_ev` for each
  frame and `temporal.flicker_ev_mean` / `flicker_ev_max` for the clip, the
  EV the HDR's average moved from one frame to the next beyond what the
  source did (not scored at a cut), and `local_flicker_ev`, the same measured
  area by area (a 36 × 64 grid) and averaged. A few thousandths are below what you can
  see; from about 0.02 EV (1.4 %) a steady shot can visibly pump.

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
To turn the daily check off, untick **Help > Check for updates automatically**
(the installer asks too), or start RUDRA with the environment variable
`RUDRA_NO_UPDATE_CHECK=1`. Everything RUDRA connects to, and when:
[PRIVACY.md](PRIVACY.md).

## 5. Command line

`rudra-native` (beside `RUDRA.exe`, or `RUDRA.app/Contents/MacOS/`) runs the
same pipeline without the window: `video` (a movie in, HDR10/HLG/ProRes
out), `deliver`, `batch` (a queue file), `diff`, `info`, `bench` and
`ffmpeg-check` (is this ffmpeg complete?), and `bench-clip-fill` (the
clipped-highlight fill as a bench tree). Run `rudra-native` alone for the
commands and their options. In a `--params` or `--grade` settings file,
`"clip_fill": false` turns the fill off and `"clip_peaks": [{"x": 0.5,
"y": 0.3, "nits": 2500}]` sets peaks (x and y across the frame, 0 to 1).

The packaged ffmpeg is used first. To use your own instead, set
`RUDRA_FFMPEG_DIR` to its folder, or `RUDRA_FFMPEG_DIR=path` for the one on
the PATH.

### In Resolve and Nuke: the OFX plug-in

**RUDRA SDR to HDR** (in development) is the same master chain as an OpenFX
effect for DaVinci Resolve and Nuke: the analytic and learned ACES 2.0
routes, the luminance limit with its matte, and AI fill (Generate, then play
the clip once). It downloads no model: it uses the RUDRA installed on the
same computer: **Help › Install the OFX plug-in for Resolve and Nuke…** puts it in both hosts. Setup, the controls and AI fill's steps are in
[OFX_PLUGIN.md](OFX_PLUGIN.md).

## 6. When something goes wrong

- **Help > Check this computer…** first: it names what is missing and
  installs it.
- **AI fill setup** (Reconstruct › AI generation › **Setup…**, or **Set up…**
  in Check this computer): **Set up automatically…** installs AI fill's own
  Python (in RUDRA's data folder, nothing else on the computer changes) and
  downloads its base model; pressed again after a stop, it carries on where
  it stopped. When it stops, the window says why, and **Copy log** puts
  everything it printed on the clipboard (the file is `genfill\setup.log` in
  RUDRA's data folder: on Windows `%LOCALAPPDATA%\FXTD Studios\RUDRA`). Send
  that with a report. **Install from a folder or zip…** installs the base
  model (or Florence-2) from a copy you were given, checked file by file,
  with no 18 GB download; on a computer without AI fill's Python it sets that
  up first (about 5 GB from the internet).
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
| F | Fit image to window (while not typing in a text field) |
| Z / Y | Undo / redo |
| G / Shift+G | Action / title safe guides |
| F1 | This guide |
| ? | The shortcut sheet |

**Help › Keyboard** (or **?**) shows the shortcuts on a card over the window,
in groups; Esc, its close button or a click anywhere closes it.

On macOS, Ctrl is Cmd.
