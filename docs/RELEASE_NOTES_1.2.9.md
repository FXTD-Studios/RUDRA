# RUDRA 1.2.9

RUDRA 1.2.9 steadies the end of an AI fill. Before, the last few seconds of a shot could change
character: a lamp in the fill grew a hot spot and the sea horizon stepped. The fill now draws the
end of the shot the way it draws the rest. About's Licences line is no longer cut off.

## Opening it the first time

- **Windows**: if Windows asks, choose **More info** › **Run anyway**. Signing through SignPath
  Foundation follows once SignPath has approved RUDRA ([code signing policy](CODE_SIGNING_POLICY.md)).
- **macOS**: right-click RUDRA.app › **Open** › **Open** (or System Settings › Privacy & Security ›
  **Open Anyway**).
- **Linux**: unpack `RUDRA-1.2.9-linux-x86_64.tar.gz` and run `RUDRA` inside it.

Download it only from this GitHub page; the checksums are listed with each release.

## AI fill v4.3: a steady end of the shot

The fill is drawn in overlapping 33-frame windows. A shot rarely ends on a whole window, so the last
one was short, and it was padded with copies of the shot's last frame before the model drew it. The
model drew that padded window differently from the others. On 09V (97 frames) the last window was
frames 75–96, and from frame 82:

- the practical lamp's brightest part rose by 1.4 stops and grew a hot spot;
- the sea horizon stepped by about 2 px, with the sea's texture changing with it.

The last window is now always a whole window: it starts earlier (frame 64 on 09V) and shares more
frames with the one before it, so nothing is padded. Measured on 09V on the GPU, against 1.2.8:

| | 1.2.8 | 1.2.9 |
|---|---|---|
| Horizon movement over the shot | 4.4 px | 1.7 px |
| Horizon step where the last window takes over | 2.2 px | 0.5 px |
| Lamp's brightest part over the shot | rises 1.4 stops | within 0.5 stops |
| Lamp's largest frame-to-frame jump | 0.16 stops | 0.03 stops |

Brightness, detail, flicker and the pixels outside the door pass the same checks as 1.2.8. The fill
takes about as long (13 minutes on 09V, against 15). This applies to v4.3 region only; v3 is unchanged. The model, its LoRA and every
other v4.3 setting are those of 1.2.8.

## About

**Help › About RUDRA Studio**: the Licences link read "Apache 2.0 · models · third part", cut off. It
now reads "Apache 2.0 · third party", in full.

## Known issues (AI fill v4.3, experimental)

- **Joins inside the shot.** Where two of the earlier windows meet (frames 25–33 and 50–58 on 09V),
  the sky's texture still changes a little, about as much as it changes from frame to frame anyway.
- **testwindow:** v4.3 invents a sun with flare rays in a sky and a bright line along the building's
  edge.

Review every v4.3 fill frame by frame (**Review fill…**) before it goes to a client.

## For developers

- `rudra/genfill/run.py`:
  - `plan_windows(..., full_last=True)`: the last window starts at `n - window` when it would be short;
    every frame is still written once, in order. `Rerender.full_last`, `--rerender-full-last`, the warm
    server's `rerender_full_last`. The previous window holds back (`carry`) and keeps (`gen_work`) every
    frame from the next window's start, not only the last `overlap` frames.
  - **Sent by the app** (`genfill_arguments` and the preview's request): app test
    `GenFill.RegionRerenderIsOptInAndRunsAsTested` checks it.
  - `--rerender-follow <0..1>` (`Rerender.follow`, `rerender_follow`): window continuation. Each window's
    first latent frames (the frames it shares with the previous window, VAE-encoded) are held at the
    previous window's picture, noised to each step's level, for this share of the steps
    (`WanFill.denoise(known=, pin=)` in `rudra/genfill/model.py`; `follow_frames`, `follow_latents`).
    Off by default and **not** sent by the app: on 09V it steadied the lamp a little more (0.54 → 0.38
    stops) and moved the horizon a little more (1.7 → 2.0 px). Kept for the v5 checks.
- `tools/window_joins.py`: the window-join measure, the step at each handover against the shot's own
  change over spans of the same length (about 1 is smooth, 3 and more is visible), for the region's
  level and texture, a box (a lamp) and any per-frame series. The table above comes from it, run on
  09V's 1.2.8 render and two 1.2.9 renders on the GPU.
- About: `aboutLicences`'s subtitle shortened, spare width shared by the link grid's three columns; app test
  `AppAbout.TheVersionTheMachineAndTheLinks` checks that no link's text is wider than its button.
- AI fill v5 training material (no model change in this release):
  - `tools/collect_genfill_v5.py`: CC0 HDR panoramas from Poly Haven and ambientCG, sorted into themes
    (thin things against the sky, doors and openings, practical lights, windows, sun, skies), with
    licence receipts and checksums.
  - `tools/blender/genfill_v5_scenes.py`: authored Blender scenes lit by those skies (antennas, cables,
    railings, branches, cargo-bay doors, windows with glazing bars, lamps, the sun in frame).
  - `tools/build_genfill_clips.py --recipe v5`: those sources, lens optics on the truth (defocus, motion
    blur, glare) and exposure to a quarter stop. Recipes v1–v4.2 are unchanged.
  - `scripts/GENFILL_V5_DATA.ps1`, registration `configs/experiments/research_genfill_v5_data_20261122.json`.
- Tests: `tests/test_release_1_2_9.py`, `tests/test_genfill_v5.py`.
