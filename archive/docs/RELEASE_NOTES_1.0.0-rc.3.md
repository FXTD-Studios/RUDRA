# RUDRA 1.0.0 RC 3

**SDR in, scene-linear HDR out.** RUDRA turns SDR stills and footage into
scene-referred HDR in three steps: Look, Check, Deliver. Every master says what
is in it, including what was invented.

> **Unsigned builds.** The Windows and macOS builds of this release candidate
> are **not code-signed**. Check each download against its `.sha256` file.
>
> - **Windows 10 / 11 (x64):** SmartScreen shows *Windows protected your PC*.
>   Click **More info › Run anyway**. For the ZIP, right-click it before
>   extracting, choose **Properties** and tick **Unblock**.
> - **macOS 13+ (Apple silicon):** the first open says RUDRA *cannot be
>   verified*. Go to **System Settings › Privacy & Security** and click **Open
>   Anyway**, or run `xattr -dr com.apple.quarantine /Applications/RUDRA.app`
>   once.
>
> Signed builds come with 1.0.

## Download

| File | For |
|---|---|
| `RUDRA-1.0.0-rc.3-windows-x64-setup.exe` | Windows installer (per user, no administrator prompt) |
| `RUDRA-1.0.0-rc.3-windows-x64.zip` | Windows portable: extract the folder, run `RUDRA.exe` |
| `RUDRA-1.0.0-rc.3-macos-arm64.dmg` | macOS, Apple silicon: drag RUDRA to Applications |
| `*.sha256` | the SHA-256 of each file |

User guide: https://www.fxtdstudios.com/rudra-docs/

## New since RC 2

- **Simple workspace.** **1 Look**, **2 Check**, **3 Deliver**. Hold **B** for
  the original SDR. One **View** menu holds the wipe, what changed, invented
  pixels and false colour. **Full** keeps every control. First-run tips; Help ›
  Show tips brings them back.
- **The Deliver tab, rebuilt.** Learned ACES 2.0 or analytic, for a still, a
  sequence or one MOV.
  - EXR float or half in ACES 2065-1 or ACEScg, DPX 10-bit, ProRes 422 HQ / 4444
    or uncompressed RGB MOV.
  - LogC4, ACEScct or the ACES 2.0 HDR 1000-nit PQ output.
  - Nuke-style paths; nothing is ever overwritten.
- **Batch queue** (Ctrl+Shift+B). Shots, sequences and movies, each with its own
  settings and range. It survives a restart.
- **Conform.** CMX 3600 EDL import, matched by clip name or reel, with handles
  and per-shot MaxCLL / MaxFALL.
- **For the colourist.** Source timecode and reel go into the EXR and MOV. Scopes
  on the learned route, trims, and ACES 2.0 SDR / HDR monitoring.
- **Model downloads in the app.** Each download is SHA-256 checked; a new About
  window.
- **New look.** A new icon and start screen.
- **AI fill (experimental, off by default; needs Python, the Wan2.1-VACE-1.3B
  base model and a 16 GB GPU).** It fills only fully clipped highlights:
  - The model draws the shape.
  - A small, separately tested estimator sets the brightness.
  - **Highlight brightness** (±3 stops) lets you move it.
  - No 8-bit code of the SDR view changes, and every invented pixel is marked
    in `rudra.generated`.
  - It passed every registered gate on 25 new scenes. Clipped highlights land
    0.13 stops closer to the truth than the exact inverse (95 % CI 0.04–0.23),
    with much more detail when graded down 3–6 stops.
  - What it draws is plausible, not recovered.

## Known

- Not code-signed (see above).
- macOS: ffmpeg is not in the DMG yet: `brew install ffmpeg@6` for movies.
- The learned ACES 2.0 model is a preview. The analytic reconstruction stays the
  default.
- The AI fill is not bundled; see the README's *AI fill* section to set it up.

Licence: code Apache 2.0; each model keeps its own licence (LICENSE-weights).
