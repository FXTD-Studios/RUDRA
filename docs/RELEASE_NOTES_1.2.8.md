# RUDRA 1.2.8

RUDRA 1.2.8 reorganises the AI fill. It now sits in three places, each with one job: a button on the
viewer to compare the fill with the master, a section in Reconstruct to set it up and preview it,
and one checkbox in Deliver to include it in the export. The viewer also shows the fill where it was
hidden before, and AI fill v4.3 no longer leaves ghost trails behind thin moving objects.

## Opening it the first time

- **Windows**: if Windows asks, choose **More info** › **Run anyway**. Signing through SignPath
  Foundation follows once SignPath has approved RUDRA ([code signing policy](CODE_SIGNING_POLICY.md)).
- **macOS**: right-click RUDRA.app › **Open** › **Open** (or System Settings › Privacy & Security ›
  **Open Anyway**).
- **Linux**: unpack `RUDRA-1.2.8-linux-x86_64.tar.gz` and run `RUDRA` inside it.

Download it only from this GitHub page; the checksums are listed with each release.

## Join us on Discord

**Help › About RUDRA Studio** has a new **Join our Discord** link: questions, previews and the RUDRA
community, at [discord.gg/C7aTNFhJeP](https://discord.gg/C7aTNFhJeP).

## AI fill: the viewer, Reconstruct and Deliver

- **Viewer: one button.** **AI fill** is on the viewer's bar, no longer inside **Overlays ▾**. One
  click shows the generated fill, and another shows the master, so you can compare. The button
  appears once there is a fill to show (a preview or a finished render) or AI generation is on.
- **Reconstruct › AI generation** (new, experimental, off by default). Its switch turns the AI fill
  on for the shot. Everything else that was in the Deliver card is here:
  - what it regenerates: **AI fill v3** or **v4.3 region**, **Keep small and thin bright parts**,
    **Where it fills**, and the thresholds in **Advanced**;
  - the prompt and **Suggest a prompt…**;
  - **Preview AI fill** and its progress;
  - **Show what it invented**;
  - **Setup…** and **Review fill…**.
- **Deliver: one checkbox.** **AI Fill Generation (Experimental)** decides whether the export
  includes the fill, made with Reconstruct's settings. It is greyed out until AI generation is on.
  The batch follows the same rule.
- **The prompt bar over the viewer is gone.** Its prompt, suggestions, v3 / v4.3 switch and
  progress are in Reconstruct.
- **Your settings carry over.** If AI fill was ticked in Deliver in 1.2.7, AI generation starts on,
  and every setting is kept.

## AI fill: the fill is no longer hidden behind the wipe

RUDRA opens every shot as an SDR | RUDRA wipe at the middle of the frame. A fill on the left half
was under the SDR side, so it could not be seen. On 09V, the whole cargo-door sky is on that side.
1.2.7 turned the wipe off only when a preview finished. Now the **AI fill** button turns the wipe off
and shows RUDRA (not Baseline) whenever you switch it on, and puts the wipe back where it was when
you switch it off.

To see a fill drawn above SDR white on an SDR screen, use a tone-mapped view (**Colour ▾** › **sRGB -
Display · ACES 2.0 - SDR 100 nits (Rec.709)**) and bring the view exposure down to about −2 EV
(**Alt+[**). The default **RUDRA view (nits)** does no tone mapping, so the fill clips to plain white
there.

## AI fill v4.3: no ghost trails

- **No more ghost trails behind thin moving objects.** In 1.2.7, a thin dark object moving against
  the redrawn sky left faint copies of its old positions behind it. On 09V the radio antenna dragged
  a fan of them through the clouds. The step that keeps the fill steady from frame to frame carried
  the master's dark edge pixels forward along with the fill, and the model's own drawing of the
  antenna at the region's edge. It now carries only the fill's own sky: nothing of the master, and
  less of any pixel that sat beside something dark. On 09V the trail is gone; what is left near the
  antenna is as deep as the cloud texture itself. Checked on the GPU: brightness, detail, flicker
  and the pixels outside the door all pass as before (flicker 0.044 EV against a limit of 0.10).

This applies to v4.3 region only; v3 is unchanged. The model, its LoRA and every other v4.3 setting
are those of 1.2.7.

## Known issues (AI fill v4.3, experimental)

- **The horizon can move where two model windows meet.** The fill is drawn in overlapping 33-frame
  windows, each from the source alone, crossfaded over the 8 frames they share. On 09V the last
  window (frames 75–96) draws its sea about 4 px higher with a different texture, and it slides in
  while the camera stands still. Lining each window up with the one before it was tried and left
  off: it followed the clouds rather than the sea and moved the horizon more than it saved.
- **testwindow:** v4.3 invents a sun with flare rays in a sky and a bright line along the building's
  edge.

Review every v4.3 fill frame by frame (**Review fill…**) before it goes to a client.

## For developers

- New card `reconAiCard` (Reconstruct) with the switch `reconAiGeneration`, setting
  `reconstruct/aiGeneration`, `set_deliver("ai_generation", …)`. When the setting is missing, it
  starts from `deliver/aiFill`.
- The controls keep their `deliverAi*` object names, and the `deliver/ai*` settings keys are
  unchanged.
- `MainWindow::deliver_effective()` and the new `deliver_for_batch()` clear `ai_fill` while AI
  generation is off.
- `deliverAiFill` is relabelled "AI Fill Generation (Experimental)".
- Removed: `viewerAiPromptBar`, `viewerAiPrompt`, `viewerAiProgress`, `viewerAiHint` (now
  `deliverAiViewHint` on the card), `viewerAiRerender`, `viewerAiPreviewAgain`, `deliverAiShowFill`
  and `deliverAiShowMaster`.
- `sync_ai_prompt_bar` is now `sync_ai_generation`.
- `vtAiFill` moved from the Overlays panel to the hud.
- `set_fill_view(true)` turns the wipe off and shows `model`. The wipe's position is kept in
  `fill_view_hid_wipe_` and put back by `set_fill_view(false)`.
- App tests:
  - New: `AppFillView.TheAiFillIsSplitIntoViewerReconstructAndDeliver`.
  - Updated: `AppFillView.TheViewerHasAnAiFillToggleThatIsSafeWithoutAFill` (the wipe), the Deliver
    AI fill test, `TheAiFillHasAProgressBar`, `TheAiFillCardShowsWhereItFillsOnTheFrameOnScreen`,
    `FullModeIsCardsInTheOrderADeliveryIsDecided`, `TheViewerBarIsThreeButtonsAndNothingIsLost`,
    `UnusableChoicesAreNotOffered`.
- `docs/USER_GUIDE.md`: a new section, "AI fill: the viewer, Reconstruct and Deliver".
- `rudra/genfill/composite.py`: `finish_rerender`'s smoothing carries the fill's own log luminance
  (`whole` with a level), as much as the previous frame's pixel was the fill's own sky (`state['trust']`
  = the code ramp `s` times the fringe confidence `conf`), and mixes the master's share in after it.
- `rudra/genfill/run.py`: `window_alignment` and `align_frames` (cv2.findTransformECC,
  MOTION_TRANSLATION, inside the region, median over the shared frames; at most 3 % of the working
  width, correlation at least 0.5), `Rerender.align`, `--rerender-align`, the warm server's
  `rerender_align`, and `window_align` in the stats. Off by default and **not** sent by the app: on
  09V's GPU check it moved the horizon over frames 0–74 from 1.7 to 3.4 px while saving 1.4 px over
  75–96. Kept for the next experiments (a longer overlap, aligning on the sea band alone).
- About: `aboutDiscord` (https://discord.gg/C7aTNFhJeP), a row of its own under the six links; app
  test `AppAbout.TheVersionTheMachineAndTheLinks` checks it.
- `tests/test_release_1_2_8.py` (the trail test fails on 1.2.7's composite); app test
  `GenFill.RegionRerenderIsOptInAndRunsAsTested` checks that the app does not send `--rerender-align`.
- `tmp/09V_test/v4_gates/run_v4_3_fix_check.py`: the GPU check on 09V (gates, horizon, trails).
