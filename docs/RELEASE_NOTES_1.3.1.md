# RUDRA 1.3.1

RUDRA 1.3.1 fixes the colour of analytic masters. With Source BT.1886 (the default), they now keep the SDR's
saturation in the shadows and mid-tones. The QC report now measures flicker inside the AI fill. A fill that
flickers for one frame is dropped region by region, not frame by frame. Source has two new choices: **sRGB**,
for an SDR read as sRGB (as in Nuke), and **BT.1886 shoulder**, which gives lamps and windows light above SDR
white.

## Opening it the first time

- **Windows**: if Windows asks, choose **More info** › **Run anyway**. Signing through SignPath
  Foundation follows once SignPath has approved RUDRA ([code signing policy](CODE_SIGNING_POLICY.md)).
- **macOS**: right-click RUDRA.app › **Open** › **Open** (or System Settings › Privacy & Security ›
  **Open Anyway**).
- **Linux**: unpack `RUDRA-1.3.1-linux-x86_64.tar.gz` and run `RUDRA` inside it.

Download it only from this GitHub page; the checksums are listed with each release.

## Analytic masters keep the SDR's colour

Before 1.3.1, an analytic master took its brightness from the Source curve but its colour from the sRGB curve.
That held whatever the Source was set to. Under BT.1886 the two curves disagree in the dark, so masters came out
less saturated than the SDR on a gamma 2.4 display. Their brightness and hue still matched.

In 1.3.1 the colour follows the Source curve too. On a welding shot under pendant lamps (04), the analytic
master's saturation against the SDR, read as BT.1886:

| SDR code range | 1.3.0 | 1.3.1 |
|---|---|---|
| Shadows (10–50) | −41 % | +0.1 % |
| Mid-tones (50–180) | −13 % | 0.0 % |
| Bright (180–250) | −7 % | 0.0 % |

Hue stays within 0.04 degrees. Analytic masters made with BT.1886 (and BT.1886 shadows) by earlier versions
are less saturated than 1.3.1's: render again to compare. With Source Unknown, ACES and sRGB, masters are
unchanged. The learned ACES 2.0 route is unchanged.

## Source: sRGB

The Source choice that used to be called **Rec.709 · plain clip** is now **sRGB · plain clip**. It reads the
SDR's codes as sRGB and always did: exactly the reading of Nuke's **sRGB Encoded Rec.709 (sRGB)** and of
RUDRA's viewer default under an OCIO view. A master made with it lines up with an SDR read that way, shadows
included. Projects that used it keep it.

## Source: BT.1886 shoulder (opt-in)

An SDR made through an ACES-style render has a shoulder: the render rolls the brightest light into the top
codes. **BT.1886 shoulder** rolls it back out. It is BT.1886 up to SDR code 165. Above that it adds light:
+0.1 stop at code 200, +0.5 at 222, +1.25 at 242 and +2 stops at white. Lamps, windows and the bright side of a
face get light above SDR white instead of stopping at it, and Fill clipped highlights builds on top.

It is one curve for every shot. The learned ACES 2.0 route adds +1.7 stops at code 242 on the welding shot and
+0.5 on a sunrise; this curve sits between the two. Large blown areas get brighter but no more detail, so check
them at −3 EV, and use AI fill or a lower Strength there. BT.1886 stays the default.

## Flicker in the AI fill, in the QC report

Every AI fill now measures its own flicker. The measure covers pixels the SDR clips and holds still: their
frame-to-frame change, as the release check measures it (95th percentile, limit 0.10 stop). The QC report beside
the render shows it in the AI fill section. A fill over the limit is listed for review with its number, so the
QC report no longer passes until someone has looked at it. Shots with no still clipped pixels say "not measured".

## Drop flickering fill, one region at a time

Deliver's **Drop flickering fill** (on by default) used to drop a whole frame's fill when it jumped more than a
quarter stop into that frame and out of it. It now does this per fill region (a lamp, an arc, a window). A
region that flickers goes back to the master's own pixels in that frame, and the frame's other regions keep
their fill. If every region in a frame flickers, the whole frame goes back, as before. The QC report counts the
regions that fell back.

## Fixed

- The colour of analytic masters under Source BT.1886 and BT.1886 shadows (above).
- Three files that were never meant to ship had been in the source since 1.2.6: a 61 MB copy of the source
  (`rudra_src.tgz`), its file list and an empty CI log. They are removed.

## Tested

- Native tests: 554 of 554 pass, including the new colour, sRGB and shoulder tests and the master chain's
  golden files. App tests for Deliver, Batch, AI fill and the QC report: 184 pass; 8 skip because they need the
  learned model loaded.
- Python tests: 1630 pass. Three fail only where the installed OpenColorIO has no ACES 2.1 built-in
  configuration (the ACES 2.1 pipeline pairs); they are unrelated to this release.
- On real shots (analytic route, AI fill v5): 04 and 09V pass the new colour check.
- 09V passes every gate: flicker 0.030 stop against 0.10. A test with 8 fill steps instead of 30 also passed
  every gate (and took 40 % less time). Its pictures, though, showed horizontal stripes over the sky, so AI fill
  stays at 30 steps.

## Known limits

- AI fill (v3, v4.3 and v5) is experimental. It starts on only where the computer can run it, and a render
  includes it only when Deliver's AI fill box is ticked.
- On the welding shot (04), the analytic route's fill still flickers: 0.226 stop against 0.10, at the lamps and
  the arc. The flicker is steady pumping, not one-frame jumps, so the per-region fallback does not catch it.
  The learned route passes there (0.094). The QC report now flags such a fill: play it before delivery, or turn
  the fill off for that shot.
- v5's hold for thin lights and 1.3.0's steadier blend were chosen on the same shots they were then checked
  on. A check on more fresh shots is still to come.
- On the learned ACES 2.0 route, v5 passes 9 of 10 check shots; 02V flickers (0.118 stop against 0.10). On
  held-out scenes the fill was further from the truth than the learned reconstruction alone.
- The learned ACES 2.0 route undoes ACES 2.0's look also on a source not made through ACES 2.0. On 04 it is
  38 % less saturated than the SDR in the shadows and almost twice as saturated in the bright range. That is the
  model, not the colour fix above: use the analytic route where the SDR was not made through ACES 2.0.
- The release check does not yet see horizontal stripes in a fill; the 8-step test was caught by eye.
- BT.1886 shoulder is one curve for every shot. The learned route adapts the shoulder to the content.
- Keep super-whites stays experimental.
