# RUDRA SDR to HDR: the OFX plug-in

**RUDRA SDR to HDR** is RUDRA as an OpenFX effect for DaVinci Resolve, Nuke and other OFX hosts. It takes the
SDR picture of the node above it and puts out scene-linear or PQ HDR. It runs the RUDRA app's own master chain,
so a frame from the plug-in is the frame RUDRA masters.

## What it needs

The plug-in carries no model and downloads nothing. It works with the RUDRA app installed on the same computer:

| | Needs |
|---|---|
| **Analytic** route | Nothing: it runs inside the plug-in, on the CPU. |
| **Learned ACES 2.0** route | RUDRA's learned ACES 2.0 model, installed in RUDRA (Help › Check this computer…, or File › Model packages). The plug-in starts RUDRA's own engine (`rudra-native`) beside the host once, and keeps the model loaded. |
| **AI fill** (experimental) | RUDRA's AI fill set up (Reconstruct › AI generation › Setup…, with AI fill v5 from File › Model packages) and an NVIDIA GPU with 16 GB. The fill runs beside the host, as a background job. |

The engine and AI fill run outside the host on purpose: Nuke ships its own LibTorch, and the two clash inside one
process. A small settings file tells the plug-in where RUDRA's engine, model and AI fill are:
`%LOCALAPPDATA%\FXTD Studios\RUDRA\ofx_engine.json` on Windows, written by Help › Install the OFX plug-in. Without it only the Analytic route works, and the others say what is missing.

## Install

Every RUDRA package carries the plug-in (from 1.3.2). Close Resolve and Nuke, then in RUDRA choose
**Help › Install the OFX plug-in for Resolve and Nuke…**. It:

1. copies `RUDRA.ofx.bundle` into the system's OFX folder, asking once for an administrator
   (`C:\Program Files\Common Files\OFX\Plugins`, `/Library/OFX/Plugins`, `/usr/OFX/Plugins`);
2. writes `ofx_engine.json` for this RUDRA: its engine, its learned ACES 2.0 model and, when set up, AI fill v5;
3. gives Nuke's toolbar the RUDRA icon (`~/.nuke/RUDRA` and one line in `~/.nuke/init.py`).

Run it again after installing the learned model or AI fill, or after updating RUDRA: it says whether the installed
plug-in is older and replaces it. A development build installs it with `tmp\BUILD_OFX.ps1 -Install` instead.

The effect is **RUDRA › RUDRA SDR to HDR**: in Resolve in the OpenFX library, in Nuke from the Tab menu or the
RUDRA toolbar menu.

Feed it the SDR's own code values:

- **Nuke:** read the SDR with **Raw Data** on. In an ACES script, set Output to **Scene-linear ACEScg**, Nuke's
  working space.
- **Resolve:** a YRGB timeline (not ACES-managed), and an Output to suit the timeline.

## The controls

- **Route:** Analytic, or Learned ACES 2.0. The learned route is made for SDR rendered through ACES 2.0; on other
  SDR it changes the colour (the RUDRA release notes give the numbers).
- **Source, Recovery, Fill clipped highlights, Anchor:** the analytic route's, as in the app (greyed on the
  learned route).
- **Exposure, Strength, Output:** both routes. Output is scene-linear Rec.2020 or ACEScg (1.0 = 203 nits), or PQ
  Rec.2020.
- **Luminance limit:** the app's Reconstruct › Limit to a luminance region. It limits where RUDRA reconstructs,
  and where AI fill may draw, by the SDR's brightness: Off, Highlights above, Shadows below, both, or a band, each
  edge with its Softness. Outside the limit the picture is RUDRA's plain inverse (of the Source curve on the
  analytic route, of the ACES 2.0 view on the learned route).
- **Alpha:** the source's alpha, the luminance limit as a matte, or what AI drew (`rudra.generated`), for keying
  or checking in the host. With a Mask, each is the mask's too.
- **Mask** (Nuke's mask input): where RUDRA acts, the reconstruction and AI fill alike; outside it the picture is
  the plain inverse, as outside the luminance limit. **Invert mask** (in the Luminance limit group) swaps inside and
  outside. In Resolve, limit the node with its own Power Window or qualifier instead: Resolve keys an OpenFX node
  itself.

## AI fill (experimental)

OFX hands a plug-in one frame at a time, but AI fill draws a shot in windows of 33 frames. So it works in three
steps, the same in Nuke and Resolve:

1. **Generate.** Open the AI fill group and press Generate, with a Prompt naming what the clipped areas show if
   you like, a **Seed** (another seed draws another fill) and **Steps** (30, as AI fill v5 was tested: about 20
   minutes for 100 frames; fewer is faster, but below 20 it can draw stripes). **From frame** and **To frame**
   choose the frames it draws (0: the clip's first and last): try it on a few, then the shot. The status says how
   many frames and roughly how long. The node starts collecting.
2. **Play the clip once at full size.** Nuke: proxy off and the viewer's downscale at 1:1, or render the node.
   Resolve: playback with no timeline proxy. Every frame the host renders is stored, its SDR codes and its master;
   **Refresh status** shows "collecting frames, k of n".
3. **The fill is drawn.** When the last frame is in, the fill starts beside the host on the GPU, as the RUDRA app
   runs AI fill v5: about 20 minutes for 100 frames. The status reads "drawing, frame k of n" and the master is
   shown meanwhile. When **Refresh status** says ready, the host renders the clip again and every frame shows the
   stored fill, after Exposure and in the chosen Output. **Alpha › What AI drew** shows where it drew.

A prompt costs time once: a prompt the fill has not read before loads the 11 GB text model on the CPU to read it,
a minute or two before drawing starts (the status says so). Every prompt read is kept for every RUDRA node, beside
the stored fills (`ofx_fill\prompt_cache`), and the prompts the RUDRA app already knows are copied in: the same
prompt again costs nothing. An empty prompt draws from the picture alone and needs no text model.

A stored fill belongs to the settings it was made with. A changed route, Source, Strength, Recovery, luminance
limit, prompt, seed or steps makes it **out of date**, and the master is shown until you Generate again or set it back.
Exposure, Output, Alpha and the Mask can change freely: the mask limits where the fill is shown. A frame whose picture coming into the node has changed since
Generate (a grade above it, say) shows the master too. **Cancel** stops collecting, or the fill being drawn;
**Clear stored fill** deletes the node's stored frames and fills from the disk.

What AI draws is generated, not recovered: review it before delivery. The node's frames and fills are kept in
`%LOCALAPPDATA%\FXTD Studios\RUDRA\ofx_fill\<node>\<settings>\`: the SDR codes, the masters, `fill\` (an ACEScg EXR
per frame, with `rudra.generated`), `status.json` and `job.log`. Allow roughly 30 MB a frame at 1080p (about 3 GB
for 100 frames), four times that at 4K.

## When something goes wrong

- Why a render failed, and the node's warnings, are in `%TEMP%\rudra_ofx.log` (and in Nuke's terminal): Nuke
  itself shows only "Effect raised an error during render".
- The learned engine's own output is in `%TEMP%\rudra_ofx_engine.log`; an AI fill's is `job.log` in its folder.
- Collecting does not move: the node says why. "Outside the clip (A to B)" means the host renders frame numbers
  outside the clip's range; "full-size renders only" means a proxy is on.
- The fill finished but the master is still shown: press **Refresh status**. If the host still shows the old
  frames, turn AI fill off and on.
- Learned ACES 2.0 in Nuke ending at once with **0xC0000139** ("a DLL of another version") came from the DLL
  folder Nuke hands every program it starts. The engine now starts from a folder holding its own DLLs: the
  installed RUDRA's, or `build\ofx_engine` from `tmp\BUILD_OFX.ps1 -Install`.

## Limits

- The whole frame, one frame at a time; the analytic route on the CPU.
- AI fill collects frames from full-size renders only. A proxy render shows the master; the stored fill is shown at
  full size only.
- A running fill ends with the host. After a restart press Generate again: the frames already stored are kept, so
  one full-size frame starts the fill again.
- AI fill takes at most 2,000 frames: cut a longer shot.
