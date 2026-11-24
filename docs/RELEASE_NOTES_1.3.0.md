# RUDRA 1.3.0

RUDRA 1.3 brings AI fill v5, which redraws practical lamps, doors, windows, sun glare and thin objects
against a bright sky better and more steadily than v4.3. The batch queue is now the Batch tab, with the
video exports in the same list. Deliver can now write the master at twice the size, UHD or HD, and the EXR as ZIP, ZIPS or uncompressed. The QC
report beside every render says where the AI drew. Clean compression takes JPEG and MP4 block artifacts out
before the reconstruction. Analytic masters with Source BT.1886 (the default) are no longer too dark in the shadows. The Source panel says whether a movie has detail above white. An AI fill checks the disk, the time it will take and the GPU before it starts.

## Opening it the first time

- **Windows**: if Windows asks, choose **More info** › **Run anyway**. Signing through SignPath
  Foundation follows once SignPath has approved RUDRA ([code signing policy](CODE_SIGNING_POLICY.md)).
- **macOS**: right-click RUDRA.app › **Open** › **Open** (or System Settings › Privacy & Security ›
  **Open Anyway**).
- **Linux**: unpack `RUDRA-1.3.0-linux-x86_64.tar.gz` and run `RUDRA` inside it.

Download it only from this GitHub page; the checksums are listed with each release.

## AI fill v5 (experimental)

AI fill v5 is the region fill's new model. Get it from **File › Model packages… › RUDRA AI fill v5**;
the card's switch then reads **v5 region**. Where only v4.3 is installed, v4.3 runs exactly as in
1.2.9.

v5 was trained on v4.2's material plus 803 CC0 panoramas and 200 procedural Blender scenes chosen for
what v4.3 drew worst: practical lamps, windows, doors, sun glare, and thin objects (antennas, cables)
against a clipped sky, with the clipped area's edge varied (soft, smeared, haloed).

RUDRA runs v5 with one more step. Where the source shows a **thin clipped shape that holds still** (a strip
light, a tube, a small lamp), a sudden change the model draws there is held back. Without it v5 redrew a
thin LED strip a little differently on every frame, so it shimmered. A wide clipped area such as a sky is
left as v5 draws it. Holding every still clipped pixel, the first thing tried, kept faint copies of a
moving antenna in the sky beside it for several frames; this version leaves the sky alone.

Measured on the four check shots and 29 held-out scenes (registration `research_genfill_v5d_thin_20261123.json`, report `docs/models/genfill_v5/report_v5.json`):

| | v4.3 | v5 alone | v5 with the hold |
|---|---|---|---|
| Porch: a thin LED strip, frame-to-frame change where nothing moves (limit 0.10 EV) | 0.058 | 0.229 | 0.08 |
| Water: the same (limit 0.115 EV) | 0.112 | 0.089 | 0.079 |
| 09V: the practical lamp, brightest part over the shot (stops) | 0.539 | 0.184 | 0.186 |
| 09V: the lamp at the worst window join (x the change elsewhere) | 7.69 | 1.97 | 1.38 |

On 22 held-out scenes it is closer to the truth than the plain reconstruction in 21 (PU21 -30.3); every check passed. Beside 09V's antenna it draws what v5 alone draws; holding every still clipped pixel instead had kept faint copies of the antenna for several frames.

**A steadier blend (24 Nov).** Where the fill meets the reconstruction, the share of each used to follow
the source's codes, so a code or two of compression noise at the edge of a clipped area moved it from frame
to frame, and the picture there flickered by up to 0.2 stops. 1.3.0 holds the blend nearly still while the
source there stays within 3 codes of where it settled, and holds a thin light's fill more firmly against a
sudden jump. On the learned ACES 2.0 route the blend also never lets the learned reconstruction come through
brighter than the fill, and a pixel that clips mid-shot is filled from its first clipped frame.
Frame-to-frame change where nothing moves (limit 0.10 EV; water 0.115), registration
`research_genfill_v5e_20261124.json`:

| Shot | Analytic: before | Analytic: 1.3.0 | Learned: before | Learned: 1.3.0 |
|---|---|---|---|---|
| 01V | 0.164 | 0.079 | 0.112 | 0.069 |
| 02V | 0.088 | 0.072 | 0.124 | 0.118 |
| 03V | 0.173 | 0.094 | 0.318 | 0.048 |
| Porch | 0.080 | 0.050 | 0.073 | 0.041 |
| Water | 0.079 | 0.079 | 0.136 | 0.107 |

On the analytic route all seven check shots now pass (five did before), and on the held-out scenes the fill
is as close to the truth as before (PU21 -30.5 on 22, closer in 21; -31.3 on 7, closer in 7). The analytic
shots' numbers come from re-running the blend on their saved fills, which reproduces a full run exactly.

The check registered before this run asked for all ten learned-route shots to pass, so by its own rule this
version failed: 02V still flickers there (above). We ship it anyway, as a recorded decision rather than a
pass, because it measured better than or equal to the earlier blend on every shot, on both routes; the
record is in the registration.

What it draws is still a plausible guess, not recovered detail: check every frame (the viewer's
**What AI drew** button shows exactly where it drew).

On the learned ACES 2.0 route, 9 of the 10 check shots now pass (5 did before the steadier blend,
registration `research_genfill_v5_learned_20261123.json`): 01V, 03V and water no longer flicker past the
limit, and Porch's LED strip is no longer too bright (3.69 times white, from 4.07; limit 4). 02V still
flickers (0.118 EV against 0.10). On held-out scenes, measured before the steadier blend, the fill there was
further from the truth than the learned reconstruction alone (PU21 +43.1 on 21 scenes, closer in 1; +71.2
on 6, closer in 0); that check was not repeated. On these check shots the analytic route was the steadier
one for AI fill; on the first fresh shot checked since, it was the other way round (see Known limits).

## The Batch tab

Batch is now a tab of its own, beside Reconstruct and Deliver. One list holds every shot waiting to render
through the learned ACES 2.0 model and every video export, each with its picture and its status. Pick a shot
to see what happened to it, change its output, switch its AI fill, or open **Details** for its frames, name,
folder, timecode, trims, MaxCLL and the EDL event it came from. **Needs attention** shows only the shots that
failed, stopped or stand out. Add shots with **+** (files, folders, the shot on screen, an EDL) or drop them
on the list. Deliver › Batch queue… now opens the tab. The pictures load in the background, so a long list
opens at once.

## Output size in Deliver

Deliver › Format › **Output size**: Source, 2×, UHD (3840 wide) or HD (1920 wide), keeping the
picture's shape (both sides even, as ProRes and HEVC need). It is a clean resize, never an AI upscale:

- upscaling is Lanczos 3 in linear light, held between the two nearest source pixels, so a bright lamp
  gets no halo or dark ring and nothing goes below its darkest neighbour;
- downscaling uses a filter with no negative weights, so it cannot ring;
- about 0.2 s a frame from 1080p to UHD.

It applies to both routes (the learned master's EXR, DPX and movie; the analytic master with its alpha,
mattes and brackets). The reconstruction and its measurements run at the source size. It is not
offered with the AI fill (the fill works at the source size) or for the analytic movie of a movie.
No side is written past 8192 pixels: a size that would go past it (2× of a source wider than 4096) is
greyed out.

## EXR compression in Deliver

Deliver › Format › **EXR compression**: **ZIP** (the default, as before), **ZIPS** or **None**. ZIP and
ZIPS are lossless, so every value is kept exactly, and make smaller files. ZIPS packs one
scanline at a time instead of 16, which reads a little faster in programs that read a line at a time
and makes slightly larger files. ACES 2065-1 (AP0) files stay uncompressed, as their container standard
(SMPTE ST 2065-4) requires; the menu shows None there. The choice applies to the EXR masters of both
routes, and a shot in the Batch tab keeps its own. On the command line: `--exr-compression none|zip|zips`.
PIZ, DWAA and B44 are not offered yet.

Checked before release on 1080p frames of one shot, read back with OpenEXR's own library: on both
routes, ZIP and ZIPS give back exactly the values None does, bit for bit. How much smaller the files get
depends on what is in them:

| Master | ZIP | ZIPS |
|---|---|---|
| Analytic, 32-bit float (the app's grading master), 10 frames | 8% smaller | 4% smaller |
| Analytic, half float, 10 frames | 39% smaller | 36% smaller |
| Learned ACES 2.0, 32-bit float with its reconstruction mask, 1 frame | 28% smaller | 25% smaller |

32-bit float values made from video keep fine noise in their last bits, which does not compress much.
With ZIP chosen, the ACES 2065-1 files stayed uncompressed, and the half-float ones carry the
`acesImageContainerFlag` that SMPTE ST 2065-4 asks for.

## The QC report and the AI fill

The QC report written beside every render now has an **AI fill** section: the share of pixels the AI
generated, the most generated frames, and the frames flagged for flicker or fallen back. A render
with AI fill no longer gets a plain green Pass: it says **Pass · contains AI fill**, because the fill
still needs a person's review. After a fill a second report, `<name>_aifill`, covers the filled frames.

## Clean compression (JPEG and MP4)

A JPEG still or an H.264/HEVC movie carries its codec's 8x8 block grid and ringing, and the HDR expansion makes
them stronger. **Reconstruct › Clean compression (JPEG/MP4)** measures the block grid on five frames of the shot,
cleans it with ffmpeg's spp filter at the smallest strength that takes it down to a clean picture's, and opens the
cleaned copy (lossless FFV1, or PNG for stills, in RUDRA's cache) as the shot. The viewer, both routes, the AI fill
and the QC report then all read the same pictures. The original is not touched; the same item goes back to it, and
a sidecar beside the copy records what was done. Where there is no grid to clean (most well-encoded movies), it
says so and changes nothing.

The same check is a button in **Reconstruct › Source**: **Check compression**. The line beside it says what it
found: "OK · no compression artifacts" with the measured block grid (a clean picture is 1.00), or that it found
artifacts and is cleaning them, then the grid before and after. On a cleaned copy the button reads **Use the
original**.

Measured on Netflix Meridian frames (24 Nov 2026), against the clean frames: JPEG quality 50 +0.43 dB PSNR on
average (block grid 1.66 to about 1.06), quality 75 +0.18 dB, H.264 CRF 35 +0.05 dB, CRF 28 +0.03 dB, and no
frame more than 0.08 dB worse. Experimental: banding in skies is not cleaned yet.

## Detail above white

Video often keeps detail in the codes above white (236–254 in 8-bit), which a normal decode clips.
The Source panel now measures it on 8 frames, in the movie's own codes, and says whether the movie has
any. Where it does, it points to **Keep super-whites** (still experimental and opt-in).

## Before an AI fill

An AI fill writes about 45 MB a frame and takes minutes per shot. Before it starts it now:

- refuses when the disk is short of what it will write;
- estimates the time (learned from your own fills);
- warns when the GPU is busy with something else.

## Seeing the fill

- On an SDR screen the default RUDRA view does no tone mapping, so a fill above white looked plain
  white. Showing the fill switches to a tone-mapped view (ACES 2.0 SDR, −2 EV) and switches back after,
  unless you changed it yourself.
- **What AI drew** sits on the viewer bar beside **AI fill**.
- The fill shows on the RUDRA side only: the SDR side (the wipe's left, holding B, Original SDR) always shows the
  source, on both routes. Turning AI fill on no longer takes the SDR | RUDRA wipe away; drag it to see the fill.
- The viewer's **AI fill** button is the AI fill itself, on or off, and the same switch as Reconstruct › AI
  generation. It is on the viewer bar whenever a shot is open (in Full): grey **AI fill** when off, blue
  **AI fill on** when on. On, the viewer shows the fill wherever there is one; with none yet, Reconstruct › AI
  generation › Preview AI fill makes one. Off, the viewer shows the master.
- The AI fill starts on when the computer can run it: an NVIDIA GPU with 16 GB of memory (on a Mac, Apple silicon
  with 16 GB) and the AI fill installed. On any other computer it starts off, and the button's tooltip says why.
  A click changes it until RUDRA closes; its prompts and settings are kept. Deliver's **AI fill** box starts
  unticked every time, so a render includes the fill only when you tick it.

## Comparing with the SDR in an ACES view (as in Nuke)

Under an OCIO view, the viewer's SDR side is now read as an sRGB texture by default (View › Colour › SDR shown
as), as Nuke reads an SDR plate with **sRGB Encoded Rec.709 (sRGB)** in an ACES 2.0 config, so RUDRA's SDR | RUDRA
wipe is the one done in Nuke. The EXR masters need nothing in Nuke: ACEScg, 1.0 = diffuse white. Which reading of
the SDR lines up with which master, and why an analytic master looks darker in the shadows next to an SDR read as
sRGB, is in the user guide (*Checking a master in Nuke*).

## Keyboard shortcuts

**Help › Keyboard** (or **?**) is now a card over the dimmed window, like the welcome tips, instead of a window
with a title bar. The same keys are grouped (Viewer, Playback, Shot, Grade, Edit) and drawn as keycaps. Esc, the
close button or a click anywhere closes it.

## Fixed

- **Analytic masters made with Source BT.1886 (the default) were too dark in the shadows and mid-tones.** The app
  reads a movie's frames, and a still, as the source's own code values. BT.1886 read those codes through the sRGB
  curve and then the BT.709 camera curve, instead of as video (code^2.4): 0.36 stop too dark at code 128, 1.3 at
  code 51 and 2.7 at code 26 (on 09V, 1.95 stops at the median and 3.6 in the darkest 5 %). It now reads them as
  the video codes they are, as the command-line video export always did, so an analytic master matches the SDR
  outside the highlights: on 09V's frames it is now within 0.02 stop of the SDR there (2.6 stops dark before),
  every pixel within 0.05 stop, and the highlights are unchanged (the lamp still peaks at 635-906 nits). Analytic
  masters made with BT.1886 by earlier versions are darker than 1.3.0's; render again to compare. The learned
  ACES 2.0 route and the other Source curves are unchanged.
- On Windows, the processes RUDRA starts (the learned engine, an AI fill, ffmpeg) now end with RUDRA, also
  when it crashes. Before, a crash could leave the engine or an AI fill running and holding the GPU.
- RUDRA runs ffprobe and nvidia-smi only from the PATH (nvidia-smi also from Windows' own System32 folder),
  never from the folder it was started in.
- Two tests that failed only on a computer with Florence-2 beside its Wan base now use a stand-in.

## Known limits

- AI fill (v3, v4.3 and v5) is experimental. It starts on only where the computer can run it, and a render
  includes it only when Deliver's AI fill box is ticked.
- v5's hold for thin lights and 1.3.0's steadier blend were both chosen on the same shots they were then
  checked on (03V passes by 0.006 EV); a check on more fresh shots is still to come.
- On the first fresh shot checked after these choices (a welder under pendant lamps, 24 Nov), the fill
  flickered on the analytic route (0.227 EV against 0.10, at the lamps and the welding arc) and passed on the
  learned route (0.094). Neither route is steady on every shot: play each fill before delivery. The QC report
  does not yet measure flicker inside the fill.
- The steadier blend ships although the check registered for it failed on one shot (02V, learned route): a
  recorded decision, not a pass.
- On the learned ACES 2.0 route, v5 passes 9 of 10 check shots; 02V flickers (0.118 EV against 0.10), and on
  held-out scenes the fill was further from the truth than the learned reconstruction alone. v3 and v4.3 were
  not checked on the learned route.
- An SDR read as sRGB (the viewer's default under an OCIO view, and a common Nuke setup) does not line up with
  an analytic master made with Source BT.1886 in the shadows (up to 1.3 stops at code 26): two readings of the
  same SDR, not a reconstruction error. Read the SDR as Gamma 2.4 Encoded Rec.709 to compare; an sRGB Source
  choice is planned for 1.3.1.
- Analytic masters keep the SDR's hue but read its colour as sRGB, also under Source BT.1886: against the SDR
  on a gamma 2.4 display they are less saturated (on a fresh shot, 16 % in the mid-tones and 38 % in the shadows;
  brightness matches). Colour that follows the Source curve is planned for 1.3.1.
- The Radiance (ComfyUI) nodes do not have v5 yet.
- Keep super-whites stays experimental.
