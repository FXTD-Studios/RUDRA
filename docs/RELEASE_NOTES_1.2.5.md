# RUDRA 1.2.5

RUDRA now checks the computer on the first start and installs what is missing in one place.
Shots open again for everyone who updated from an older version, and the viewer is colour managed
as in Nuke.

## Opening it the first time

- **Windows**: 1.2.5 is not signed yet, so Windows may ask once: **More info** › **Run anyway**.
  Code signing with a free certificate for open-source projects from SignPath Foundation is being
  set up ([code signing policy](CODE_SIGNING_POLICY.md)); the first signed release follows once
  SignPath has approved RUDRA.
- **macOS**: unchanged. Right-click RUDRA.app › **Open** › **Open** (or System Settings ›
  Privacy & Security › **Open Anyway**).

Download it only from this GitHub page; the checksums are listed with each release.

## Check this computer

New in 1.2.5. It opens after the welcome on the first start, and after each update; the shield
button in the toolbar (with a dot when something wants a look) and **Help › Check this computer…**
open it any time.

- **Can this computer run RUDRA?** The system, the GPU and its memory, memory, free disk where
  models are kept, ffmpeg for movies, and whether the model list could be read: each line OK, a
  note, a warning or a blocker, in plain words.
- **What is missing, in one place.** The learned ACES 2.0 model is ticked: **Install selected**
  downloads it (and anything else you tick) while you work. AI fill's model, and its base model
  and Python (**Set up…**), are offered only where the GPU can run them (a 16 GB NVIDIA card; on
  a Mac, 16 GB of memory); on a smaller card the window says why. RTX 50 series cards get the
  CUDA 12.8 build of PyTorch they need.
- **AI fill no longer stops on "not installed".** When a part is missing, Deliver says which one
  in plain words and opens Check this computer, where it is set up.
- **Copy report** puts the whole check on the clipboard for a support request.

## Fixed: an old Beta 2 model opened instead of RUDRA ACES 2

1.0.0 to 1.0.2 put the Beta 2 model (`sdr2hdr_image_v8`) beside the app, and installing a newer
version over it left it there, so a start could open it. RUDRA no longer opens a Beta 2 model by
itself, wherever it is; Check this computer lists it, and **Move aside** puts it in a
`.retired-beta2` folder. The 1.2.5 installer removes the ones beside the app.

## Fixed: blank window buttons

On some Windows machines the menu, minimise, maximise and close buttons at the top of the window
were blank: they were characters of a Windows font that was not used. They are drawn now, like the
other toolbar icons.

## Fixed: nothing opened after updating to 1.2.4

If an older version had **Reconstruct › Use the model** turned on, 1.2.4 refused every shot:
**Open…** did nothing, and the Log repeated "Open a model package first, or turn off
Reconstruct > Use the model." That setting runs an optional reconstruction model, and since
1.2.0 there has been none to download, so the shot waited for one that never came, while the
learned ACES 2.0 model sat ready.

In 1.2.5:

- When no reconstruction model is installed, RUDRA turns **Use the model** off, says so in the
  Log and the status bar, and opens the shot. The learned ACES 2.0 model shows it as before.
- Ticking **Use the model** with nothing to load leaves it off and says why.
- When a shot has to wait for a model that is still loading, the status bar says so, not only
  the Log.

## The viewer is colour managed, as in Nuke

The viewer toolbar now shows the picture through OpenColorIO, the way Nuke's viewer does, so the
master can be checked without opening Nuke:

- **Views**: SDR sRGB, SDR Rec.709, HDR Rec.2100 PQ 1000 nits, P3-D65 and Raw from the built-in
  ACES 2.0 config (OpenColorIO 2.5.2), and every other display and view of it. **View › Colour
  management (OpenColorIO)…** uses `$OCIO` or your own `config.ocio` instead. On an HDR monitor an HDR view
  shows at its real level. *RUDRA view* (absolute nits, as before) stays the default.
- **ACEScg or ACES2065-1 (AP0)**: the picture as a render to that space holds it; the probe shows
  its values and the view's 10-bit codes.
- **The original beside RUDRA's result**, through an input transform: by default the inverse of
  the ACES 2.0 SDR view, so it looks as it was graded.
- **Range checks**: out of gamut (AP1, Rec.2020, P3-D65 or Rec.709), invalid values (NaN,
  infinite, outside AP0, above half float) and clipped by the view, on the picture and as shares
  of the frame under the range readout.

## Drag and drop

- A file RUDRA cannot open (not a still or a movie) is named in the status bar, instead of
  being added as a frame that fails later.
- A folder, a movie or a project is a shot of its own: dropped together with other items, RUDRA
  opens it and says what it left out.
- A dropped RUDRA ACES 2 folder is used as the learned model, as **Use** in Model packages does.
- The empty viewer in Full now says movies open too.

## File › Model packages

- The empty list at the top now says what it is: the optional reconstruction models. It no
  longer reads "No model package" above models that are installed and checked.
- **Use** on RUDRA ACES 2 shows the model's real state ("RUDRA ACES 2 is in use. ACES model
  ready on cuda…"). Before, it could stay on "Loading RUDRA ACES 2…" for good when the model was
  already loaded.

## Privacy

RUDRA has no account, no telemetry and no analytics. What it connects to, and when, is now
written down in [PRIVACY.md](PRIVACY.md), and the installer shows it on a page of its own.

- The one automatic connection is the update check: at most once a day RUDRA reads its release
  list on GitHub. The installer asks whether to keep it, and the new **Help › Check for updates
  automatically** turns it on or off at any time.
- Everything else (the model list, model downloads, AI fill's setup) happens only when you ask
  for it.

## For developers

- The Windows job of `release.yml` is ready to sign `RUDRA.exe`, `rudra-native.exe` and the
  installer through SignPath, with two signing requests approved by hand
  ([SIGNPATH_SETUP.md](SIGNPATH_SETUP.md), [CODE_SIGNING_POLICY.md](CODE_SIGNING_POLICY.md)).
  Until SignPath's approval and its secret are in place it publishes unsigned, with a warning, as
  1.2.5 is. The `.pfx` secrets are no longer read.
- `PACKAGE_WINDOWS.ps1 -NoArchive` stops after the checked package folder;
  `PACKAGE_WINDOWS_ARCHIVE.ps1` makes its ZIP and installer.
- `rudra-native.exe` has a version resource, and a release's version resources no longer mark it
  as a pre-release.
- New: `app/system_check.cpp` (Check this computer, the Beta 2 retirement); the caption buttons are
  `IconButton` glyphs; `rudra.iss` deletes `{app}\models\sdr2hdr_*` and the other Beta 2 folders.
- New tests: `AppModels.ARetiredBeta2ModelIsNeverOpenedByItself`, `AppCheck.TheMachineIsJudgedInPlainWords`,
  `AppCheck.TheWindowOffersWhatIsMissingAndInstallsTheTickedOnes`,
  `AppModels.UseTheModelWithNoPackageAnywhereDoesNotHoldTheShotBack`,
  `AppOpen.ADropSaysWhatItDoesNotOpen`, `AppUpdates.TheDailyCheckIsShownAndCanBeTurnedOff`.
- The colour-managed viewer: `core/view_ocio.hpp` (the transforms and checks, OpenColorIO-free),
  `infer/ocio_viewer.cpp` (configs, OpenColorIO's Vulkan GLSL and CPU processors),
  `display.frag` built again at run time for a view (`QShaderBaker`; `rudra_render_gpu` now links
  `Qt6::ShaderTools`), `app/colour_view.cpp`. Spec: `docs/view.spec.md` section 15. Tests:
  `ViewOcio.*`, `OcioViewer.*` (with OpenColorIO), `AppColourView.*`;
  `scripts\TEST_OCIO_VIEWER.ps1` turns OpenColorIO on in the app build and runs them.
- The website guide (`tools/build_user_guide.py`) has a **Privacy & signing** page written from
  `docs/CODE_SIGNING_POLICY.md` and `docs/PRIVACY.md`; `RELEASE_1_2_5.ps1 -Tag` rebuilds it.
