"""Build the RUDRA user guide (website\\rudra-docs\\index.html) for a release (29 Oct 2026).

    python tools/build_user_guide.py --version 1.0.0-rc.2 --label "1.0.0 RC 2" --out ../website/rudra-docs
    python tools/build_user_guide.py --version 1.0.0 --label 1.0.0 --tag v1.0.0 --release --out ../website/rudra-docs

It keeps the Train page and the Help answers from the page already in --out
(--source to read another) and writes every other page from here. --tag points
the download buttons at that release's installer and ZIP; without it they go
to the releases page. --release drops the release-candidate note.
"""
import argparse
import datetime
import pathlib
import re

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--version", required=True, help="the app's version, as in its file names (1.0.0, 1.0.0-rc.2)")
_ap.add_argument("--label", help="how the guide writes it (default: --version)")
_ap.add_argument("--tag", default="", help="the GitHub release tag the downloads point at (v1.0.0)")
_ap.add_argument("--release", action="store_true", help="a release, not a candidate")
_today = datetime.date.today()
_ap.add_argument("--updated", default=f"{_today.day} {_today:%B %Y}", help="the date the guide says it was updated")
_ap.add_argument("--out", required=True, type=pathlib.Path, help="the rudra-docs folder")
_ap.add_argument("--source", type=pathlib.Path, help="the page to keep Train and Help from (default: <out>/index.html)")
ARGS = _ap.parse_args()
ARGS.label = ARGS.label or ARGS.version
SRC_TAG = ARGS.tag or 'main'   # the source the training page clones and links to
src = (ARGS.source or ARGS.out / "index.html").read_text(encoding="utf-8")
_t = src.index('<section class="page" id="training"')
training = src[_t:src.index('<section class="page"', _t + 10)]   # up to the next page, whichever it is
# The training page keeps its content; its eyebrow number moves with the new order.
training = training.replace('04 / Optional advanced workflow', '07 / Optional advanced workflow')
training = re.sub(r'git clone --branch v[0-9][^ ]* ', f'git clone --branch {SRC_TAG} ', training)
training = re.sub(r'/blob/v[0-9][^/"]*/', f'/blob/{SRC_TAG}/', training)
training = training.replace('The desktop download already includes a model.', 'The app works without a model, and the learned ACES 2.0 model is a free download inside it.')
training = training.replace('Keep the included model until the candidate proves useful.', 'Keep the learned ACES 2.0 model until the candidate proves useful.')
training = training.replace('Compare against both the analytic baseline and the shipped model.', 'Compare against both the analytic baseline and the learned ACES 2.0 model.')
training = training.replace('Keep the included model available for comparison', 'Keep the learned ACES 2.0 model available for comparison')

VER = ARGS.label
UPDATED = ARGS.updated
REL = (f'https://github.com/FXTD-Studios/RUDRA/releases/tag/{ARGS.tag}' if ARGS.tag
       else 'https://github.com/FXTD-Studios/RUDRA/releases')
SETUP = (f'https://github.com/FXTD-Studios/RUDRA/releases/download/{ARGS.tag}/RUDRA-{ARGS.version}-windows-x64-setup.exe'
         if ARGS.tag else REL)
ZIP = (f'https://github.com/FXTD-Studios/RUDRA/releases/download/{ARGS.tag}/RUDRA-{ARGS.version}-windows-x64.zip'
       if ARGS.tag else REL)
DMG = (f'https://github.com/FXTD-Studios/RUDRA/releases/download/{ARGS.tag}/RUDRA-{ARGS.version}-macos-arm64.dmg'
       if ARGS.tag else REL)

head = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Learn to install RUDRA {VER}, turn SDR shots into HDR with the Look, Check, Deliver workflow, download models, and deliver ACES masters, HDR10 and HLG."><link rel="canonical" href="https://www.fxtdstudios.com/rudra-docs/"><title>RUDRA | User guide | FXTD Studios</title><link rel="stylesheet" href="style.css"><script src="guide.js" defer></script></head><body>
<header><a class="logo" href="https://www.fxtdstudios.com" aria-label="FXTD Studios">FXTD<sup>®</sup></a><div class="headlinks"><a href="https://www.fxtdstudios.com">FXTDSTUDIOS.COM</a><a href="https://www.fxtdstudios.com/research-lab">TOOLS</a><a href="https://www.fxtdstudios.com/rudra">RUDRA</a><a class="button" href="{REL}">Download {VER} ↗</a></div></header>
<div class="layout"><aside><div class="eyebrow">Artist documentation</div><div class="sidebar-title">RUDRA / GUIDE</div><input id="search" type="search" aria-label="Find a topic" placeholder="Find a topic…">
<nav aria-label="Documentation">
<div class="navgroup">GET STARTED</div>
<a href="#introduction" data-keywords="about hdr sdr overview what's new"><span>01</span>Introduction</a>
<a href="#installation" data-keywords="windows mac linux download setup install"><span>02</span>Installation</a>
<a href="#models" data-keywords="model download package hugging face learned aces"><span>03</span>Models</a>
<a href="#workspace" data-keywords="simple full tips look check deliver view menu b key"><span>04</span>The workspace</a>
<a href="#first-render" data-keywords="open image look trims strength peak exposure contrast"><span>05</span>Your first HDR</a>
<a href="#export" data-keywords="deliver render aces exr hdr10 hlg prores batch queue file path"><span>06</span>Deliver &amp; colour</a>
<div class="navgroup">GO FURTHER</div>
<a href="#training" data-keywords="train dataset model gpu resume checkpoint"><span>07</span>Train your model</a>
<a href="#whats-new" data-keywords="changes release notes version"><span>08</span>What’s new</a>
<a href="#help" data-keywords="faq troubleshooting licence license support about"><span>09</span>Help &amp; answers</a>
</nav><p id="empty" hidden>No matching topic.</p>
<div class="sidebar-foot"><span class="dot"></span>Desktop {VER}<br>Windows x64 · macOS<br><br>Apache 2.0 licence<br>Updated {UPDATED}<br><br><a class="textlink" href="https://www.fxtdstudios.com">www.fxtdstudios.com</a></div></aside>
<main><div class="breadcrumb">FXTD TOOLS &nbsp;/&nbsp; RUDRA &nbsp;/&nbsp; <b id="crumb">INTRODUCTION</b></div>
'''

RC_NOTE = ('' if ARGS.release else f'<div class="callout"><strong>About this release</strong>{VER} is a release candidate. Check the release notes and your organisation’s security policy.</div>')
intro = f'''<section class="page active" id="introduction" data-title="Introduction">
<div class="hero"><div class="eyebrow">A practical guide for artists</div><h1>SDR in.<br>HDR out.</h1>
<p class="lede">RUDRA turns SDR images and footage into HDR in three steps: <strong>Look</strong>, <strong>Check</strong>, <strong>Deliver</strong>. It works out of the box with its analytic reconstruction; the optional learned ACES 2.0 model rebuilds more of the light the SDR clipped. Every master says exactly what is in it.</p>
<div class="actions"><a href="#installation" class="button">Install RUDRA <span>↗</span></a><a href="#first-render" class="button outline">Make your first HDR →</a></div>
<div class="meta"><span>DESKTOP {VER.upper()}</span><span>NO MODEL NEEDED TO START</span><span>ACES EXR · HDR10 · HLG</span></div></div>
<h2>What is RUDRA?</h2>
<p>SDR stores a limited brightness range: bright skies, lamps and reflections clip to white. RUDRA reads the SDR through the curve it was made with and inverts it, filling the clipped highlights; the learned ACES 2.0 model, a free download, rebuilds what the inverse alone cannot. Everything the SDR already holds is kept.</p>
<div class="flow" aria-label="Workflow: Look, Check, Deliver"><div class="flow-box"><b>1 / Look</b><small>Open a shot, shape its HDR</small></div><div class="flow-arrow">→</div><div class="flow-box"><b>2 / Check</b><small>Read it on the scopes</small></div><div class="flow-arrow">→</div><div class="flow-box"><b>3 / Deliver</b><small>ACES master, HDR10 or HLG</small></div></div>
<div class="callout"><strong>Estimation, not a perfect restoration.</strong>Once detail is clipped away there is no unique original to recover. RUDRA marks the pixels it invented, so you can see where it estimated and where it kept your source. Always compare.</div>
<h2>Choose where to start.</h2>
<div class="grid">
<a class="card" href="#first-render"><span class="number">FOR ARTISTS <span class="arrow">↗</span></span><h3>Open a shot</h3><p>Install, download the model, and follow the four first-run tips. No training needed.</p></a>
<a class="card" href="#export"><span class="number">FOR DELIVERY <span class="arrow">↗</span></span><h3>Deliver a master</h3><p>A grading master for your colourist, an HDR10 movie, or broadcast HLG — one click each.</p></a>
<a class="card" href="#training"><span class="number">FOR CUSTOM WORK <span class="arrow">↗</span></span><h3>Train on your footage</h3><p>Prepare HDR reference data, train a candidate, and measure whether it improves your shots.</p></a>
</div>
<p class="small">This guide covers the native desktop app, {VER}. See <a class="textlink" href="#whats-new">What’s new</a> for the changes since Beta 2.</p>
</section>
'''

install = f'''<section class="page" id="installation" data-title="Installation">
<div class="eyebrow">02 / Set up your workspace</div><h1>Install. Open.<br>Start creating.</h1>
<p class="lede">Install the app, then download a model from inside it if you want one. Still images need nothing else; movies need FFmpeg, which both packages include.</p>
<div class="actions"><a class="button" href="{SETUP}">Windows installer ↓</a><a class="button outline" href="{DMG}">macOS (Apple silicon) ↓</a></div>
<p class="section-label">Windows 10 / 11 · x64</p>
<div class="steps">
<div class="step"><div><h3>Download the installer</h3><p>Get the {VER} setup file from the official release. A SHA-256 file sits beside each download.</p><a class="button" href="{SETUP}">Windows installer ↓</a></div></div>
<div class="step"><div><h3>Install, then open RUDRA</h3><p>Follow the installer and launch RUDRA from the Start menu. The builds are not code-signed yet: if SmartScreen appears, choose <strong>More info → Run anyway</strong>. The first-run check looks at your display and GPU.</p></div></div>
<div class="step"><div><h3>Add the learned model (optional)</h3><p>RUDRA works without it. For more of the clipped highlights, choose <strong>Download the model…</strong> on the Look tab, or <strong>File → Model packages… → Download</strong>. See <a class="textlink" href="#models">Models</a>.</p></div></div>
<div class="step"><div><h3>Follow the tips</h3><p>A welcome card explains the three steps; the first time you open a shot, four short tips point at each part. <strong>Help → Show tips</strong> brings them back.</p></div></div>
</div>
<h2>What your computer needs</h2>
<table><thead><tr><th>Part</th><th>For</th><th>Notes</th></tr></thead><tbody>
<tr><td>NVIDIA GPU (CUDA) or any DirectX 12 GPU (DirectML)</td><td>The learned ACES 2.0 model</td><td>CUDA is fastest. Without a GPU it runs on the CPU, slowly. The GPU also draws the HDR viewer.</td></tr>
<tr><td>16 GB GPU</td><td>AI fill (experimental)</td><td>Only for the optional generative pass; everything else runs without it.</td></tr>
<tr><td>HDR display (optional)</td><td>Seeing HDR 1000 as it is</td><td>On an SDR screen, choose the <strong>SDR 100</strong> monitor; the master is unchanged.</td></tr>
<tr><td>FFmpeg / FFprobe</td><td>Movies in and out</td><td>Included in both packages; or install it and put it on the PATH.</td></tr>
</tbody></table>
{RC_NOTE}
<p class="section-label">macOS · Apple silicon</p>
<div class="steps">
<div class="step"><div><h3>Download the DMG</h3><p>Get the {VER} DMG from the release, open it and drag <strong>RUDRA</strong> to Applications.</p><a class="button" href="{DMG}">macOS DMG ↓</a></div></div>
<div class="step"><div><h3>Open it the first time</h3><p>The build is not notarised yet. If macOS blocks it, open <strong>System Settings → Privacy &amp; Security</strong> and click <strong>Open Anyway</strong>.</p></div></div>
</div>
<p class="small">On macOS, 1.0 runs the analytic reconstruction with every Look, Check and Deliver tool. The learned ACES 2.0 model and AI fill are Windows features for now.</p>
<p>Prefer a portable app? Download the <a class="textlink" href="{ZIP}">Windows ZIP</a>, extract the whole folder, and open <strong>RUDRA.exe</strong>.</p>
<a class="button" href="#models">Next: models →</a>
</section>
'''

models = '''<section class="page" id="models" data-title="Models">
<div class="eyebrow">03 / Optional</div><h1>Add<br>a model.</h1>
<p class="lede">RUDRA starts on its analytic reconstruction, which needs no model. The learned models download separately, so a new model never needs a new install: RUDRA lists them, downloads one, checks it, and keeps it ready.</p>
<div class="steps">
<div class="step"><div><h3>Open Model packages</h3><p>Choose <strong>File → Model packages…</strong>. The <strong>Installed</strong> list shows the models on this computer; <strong>Available</strong> lists the ones you can download, with their size and what they are for.</p></div></div>
<div class="step"><div><h3>Download</h3><p>Click <strong>Download</strong> beside a model. RUDRA shows the progress, checks every file against its SHA-256, and only then adds it. A failed or cancelled download leaves nothing half-installed; you can resume it.</p></div></div>
<div class="step"><div><h3>Use it</h3><p>Click <strong>Use</strong>. The model loads once and stays loaded; the status bar says when it is ready and on which device (for example “learned ACES 2.0 (cuda)”).</p></div></div>
</div>
<table><thead><tr><th>Model</th><th>What it does</th><th>Use it for</th></tr></thead><tbody>
<tr><td>Learned ACES 2.0</td><td>Inverts the ACES 2.0 SDR view and rebuilds clipped highlights</td><td>Most shots, once downloaded (about 2 MB). Apache 2.0.</td></tr>
<tr><td>Analytic (built in)</td><td>The inverse of the curve that made the SDR, with the clipped highlights filled</td><td>Every shot, out of the box. The default until a learned model is installed.</td></tr>
<tr><td>AI fill v3 (experimental)</td><td>A generative pass for large blown regions: the RUDRA LoRA on Wan2.1-VACE-1.3B draws the shape, a small estimator sets the brightness</td><td>Hero shots, reviewed frame by frame. Needs the Wan2.1-VACE-1.3B base (from Wan-AI) and a 16 GB GPU. Apache 2.0.</td></tr>
</tbody></table>
<details><summary>Where are the models kept?</summary><p>In your user folder (on Windows, <strong>%LOCALAPPDATA%\\FXTD Studios\\RUDRA\\models</strong>). <strong>Show in folder</strong> opens it. To move them, choose another folder in Model packages; RUDRA searches every folder listed there.</p></details>
<details><summary>Can I use a model I trained?</summary><p>Yes. Export it as a package (a folder with <strong>manifest.json</strong>) and choose <strong>Add folder…</strong> in Model packages. See <a class="textlink" href="#training">Train your model</a>.</p></details>
<details><summary>No internet on this computer?</summary><p>Download the package on another computer from the model page, copy the folder across, and use <strong>Add folder…</strong>.</p></details>
<div class="callout"><strong>Model terms.</strong>The app and the 1.0 models are Apache 2.0 (<a class="textlink" href="https://huggingface.co/fxtdstudios/rudra-models">fxtdstudios/rudra-models</a>); each model shows its licence before you download it. Each one shows its licence in Model packages before you download it.</div>
</section>
'''

workspace = '''<section class="page" id="workspace" data-title="The workspace">
<div class="eyebrow">04 / Find your way</div><h1>Three steps.<br>Nothing else.</h1>
<p class="lede">RUDRA opens in <strong>Simple</strong>: the three steps and the controls most shots need. <strong>Full</strong> shows every control. Switch with the Simple / Full buttons in the toolbar, or <strong>Advanced →</strong> on the Look tab.</p>
<table><thead><tr><th>Tab</th><th>What is on it</th></tr></thead><tbody>
<tr><td>1 Look</td><td>The monitor (SDR 100 or HDR 1000) and four trims: Strength, Peak, Exposure, Contrast. Copy look, Paste, Reset.</td></tr>
<tr><td>2 Check</td><td>The frame in four numbers — Peak, Average, Brighter than SDR white, Outside P3-D65 — and the scopes beside the picture. Grab a still to compare.</td></tr>
<tr><td>3 Deliver</td><td>Grading master, HDR10 movie, Broadcast HDR or Custom…, the file, and Render.</td></tr>
</tbody></table>
<h2>The viewer bar</h2>
<table><thead><tr><th>Control</th><th>What it does</th></tr></thead><tbody>
<tr><td>Hold <strong>B</strong></td><td>Shows the original SDR while you hold the key; let go for the HDR.</td></tr>
<tr><td>View ▾</td><td>HDR, Original SDR, Wipe, What changed, Invented pixels, False colour.</td></tr>
<tr><td>Wipe (<strong>W</strong>)</td><td>The original SDR on the left, RUDRA on the right; the bar says which side is which. Drag the seam, or move it with ← → (Shift for fine steps). It stays where you leave it while you change the peak or the exposure. <strong>W</strong> or <strong>Esc</strong> closes it.</td></tr>
<tr><td>Fit / 100%</td><td>Fit the frame to the window, or see every pixel. Scroll to zoom, middle-drag to pan.</td></tr>
<tr><td>Scopes</td><td>Show or hide the waveform, histogram, vectorscope and RGB parade beside the picture.</td></tr>
</tbody></table>
<h2>What Full adds</h2>
<p>The analytic route and its Source curves and calibration, region exposure (the Grade tab), the probe, collections, the colour pipeline strip, and every delivery option: format, colour, bit depth, codec, start frame, timecode and reel.</p>
<h2>Useful keys</h2>
<table><thead><tr><th>Key</th><th>Does</th></tr></thead><tbody>
<tr><td>Space</td><td>Play / pause</td></tr><tr><td>, &nbsp; .</td><td>Previous / next frame</td></tr>
<tr><td>B (hold)</td><td>Original SDR</td></tr><tr><td>W</td><td>Wipe</td></tr><tr><td>F</td><td>Fit</td></tr>
<tr><td>Ctrl+B</td><td>Add the shot to the batch queue</td></tr><tr><td>?</td><td>Every shortcut</td></tr>
</tbody></table>
</section>
'''

first = '''<section class="page" id="first-render" data-title="Your first HDR">
<div class="eyebrow">05 / Learn by doing</div><h1>Your first<br>HDR shot.</h1>
<p class="lede">Start with one still or a short clip. Look, check, deliver — then do the rest of the sequence.</p>
<div class="steps">
<div class="step"><div><h3>Open a shot</h3><p>Drop a frame, a folder of frames or a movie on the window, or click <strong>Choose…</strong>. Stills: PNG, JPEG, TIFF, WebP, BMP. Movies: anything FFmpeg reads. Start from the original file; messaging apps recompress.</p></div></div>
<div class="step"><div><h3>Pick the monitor</h3><p>On <strong>1 Look</strong>, choose <strong>HDR 1000</strong> on an HDR display, or <strong>SDR 100</strong> to judge the shot on an ordinary screen. The monitor changes what you see, never the master.</p></div></div>
<div class="step"><div><h3>Compare with the SDR</h3><p>Hold <strong>B</strong> to see the original SDR; let go for the HDR. <strong>W</strong> (or <strong>View ▾ → Wipe</strong>) puts them side by side, SDR left and RUDRA right, with a seam you can drag; and <strong>Invented pixels</strong> shows where RUDRA estimated what the SDR had lost.</p></div></div>
<div class="step"><div><h3>Trim only what the shot needs</h3><p><strong>Strength</strong> — how far from the plain inverse (0 %) to the model (100 %). <strong>Peak</strong> — a soft limit on the rebuilt highlights. <strong>Exposure</strong> and <strong>Contrast</strong> — the whole master. The trims are remembered per shot; <strong>Copy look</strong> and <strong>Paste</strong> carry them to the next one.</p></div></div>
<div class="step"><div><h3>Check it</h3><p>On <strong>2 Check</strong>, read the four numbers and the scopes. A peak far above your target, or a large share outside P3-D65, deserves a closer look. <strong>Grab still</strong> keeps a reference to compare shots.</p></div></div>
<div class="step"><div><h3>Deliver</h3><p>On <strong>3 Deliver</strong>, pick what you need and click <strong>Render</strong>. <a class="textlink" href="#export">Choose the right delivery →</a></p></div></div>
</div>
<h2>Read the Check tab in plain language.</h2>
<table><thead><tr><th>Reading</th><th>What it tells you</th></tr></thead><tbody>
<tr><td>Peak</td><td>The brightest pixel, in nits. An HDR10 1000-nit master should not need much more.</td></tr>
<tr><td>Average</td><td>The frame’s average light (what MaxFALL counts). A big jump between shots is visible to viewers.</td></tr>
<tr><td>Brighter than SDR white</td><td>The share of the frame above diffuse white (203 nits) — the light the SDR could not hold.</td></tr>
<tr><td>Outside P3-D65</td><td>Colours beyond the gamut HDR10 and Dolby Vision masters are graded in.</td></tr>
<tr><td>Waveform · Histogram · Vectorscope · RGB parade</td><td>Where the light sits across the frame, how much of it sits at each level, colour direction and saturation, and each channel on the PQ scale.</td></tr>
</tbody></table>
<div class="callout"><strong>Your screen and your file are different things.</strong>An SDR monitor cannot show HDR brightness, and a screenshot is not proof the master is right. Review the file in a colour-managed HDR workflow.</div>
</section>
'''

export = '''<section class="page" id="export" data-title="Deliver & colour">
<div class="eyebrow">06 / Deliver your work</div><h1>A master.<br>In your folder.</h1>
<p class="lede">On <strong>3 Deliver</strong>, pick one of four choices, check the file path, and Render. Nothing is ever overwritten.</p>
<table><thead><tr><th>Choose</th><th>You get</th><th>For</th></tr></thead><tbody>
<tr><td>Grading master</td><td>ACES 2065-1 EXR (ST 2065-4: AP0, half float), one per frame</td><td>Your colourist, compositing, an ACES pipeline</td></tr>
<tr><td>HDR10 movie</td><td>HEVC 10-bit MP4, PQ 1000 nits P3-D65, with MaxCLL / MaxFALL and mastering-display metadata (with the learned model; without it, Custom… has the analytic PQ movie)</td><td>Streaming, review on HDR TVs</td></tr>
<tr><td>Broadcast HDR</td><td>HLG ProRes 422 HQ</td><td>Broadcast and live-to-air</td></tr>
<tr><td>Custom…</td><td>Every option: EXR 16/32-bit, ACEScg, DPX 10-bit, PQ 2000/4000, ARRI LogC4, ACEScct, ProRes 4444 / 4444 XQ, SDR 709, Dolby Vision XML</td><td>Anything else</td></tr>
</tbody></table>
<div class="steps">
<div class="step"><div><h3>Name the file</h3><p>One path, as Nuke’s Write takes it: folder, name, <strong>####</strong> for the frame number, extension — for example <strong>D:\\Renders\\SH010\\SH010_hdr.####.exr</strong>. <strong>Browse…</strong> picks it; <strong>Create directories</strong> makes a missing folder.</p></div></div>
<div class="step"><div><h3>Render, or queue</h3><p><strong>Render</strong> writes this shot now. <strong>Add to batch queue</strong> (Ctrl+B) keeps the shot with its look and settings, to render many shots in one go from <strong>Deliver → Batch queue…</strong>.</p></div></div>
<div class="step"><div><h3>Open it where it goes next</h3><p>Every render writes its provenance beside it: the model, the trims, the source, the timecode. Read the EXR as ACES 2065-1 in the receiving app.</p></div></div>
</div>
<details><summary>Start frame, timecode and reel</summary><p>In Full, the Render card sets the first frame number, the start timecode and the reel name. A movie keeps its own timecode and reel unless you type others. They are written into the EXR header, the DPX header and the MOV.</p></details>
<details><summary>AI fill (optional)</summary><p>In Custom…, an EXR still or sequence can get a second, generative pass for large blown regions. It writes a copy beside the master and a review folder; <strong>Review fill…</strong> pages through master, fill and mask, and flags frames that flicker. Review every frame before it reaches a client.</p></details>
<details><summary>Conforming from an EDL</summary><p><strong>Import EDL…</strong> in <strong>Deliver → Batch queue…</strong> takes the edit’s ranges and record timecode into the batch queue, with handles. Retimes (M2) are honoured; short handles are warned about.</p></details>
<div class="callout"><strong>For moving shots</strong>Play the whole result, not a few frames. Watch for flicker, unstable grain and changes around bright edges.</div>
</section>
'''

news = f'''<section class="page" id="whats-new" data-title="What’s new">
<div class="eyebrow">08 / Release notes</div><h1>What’s new<br>in {VER}.</h1>
<h2>In 1.0.1</h2>
<ul>
<li><strong>The first published 1.0 build</strong>, for Windows and, again, macOS (Apple silicon).</li>
<li><strong>Wipe</strong>: the seam stays where you leave it when you change the peak or the exposure, and the bar says which side is which: SDR left, RUDRA right.</li>
<li><strong>Clipped highlights</strong>: a large blown area, like a sun in a hazy sky, now runs toward white at its centre instead of carrying the sky’s colour all the way in. Measured against HDR references, the colour is closer and the brightness unchanged.</li>
<li>The model card, README and installers’ “Read me first” are rewritten for artists.</li>
</ul>
<h2>Since Beta 2</h2>
<ul>
<li><strong>Learned ACES 2.0 model</strong>, a free download inside the app: kept loaded on the GPU, in the viewer and every render. The analytic reconstruction needs no model.</li>
<li><strong>Simple workspace</strong>: Look, Check, Deliver, a first-run welcome and four tips. Full keeps every control.</li>
<li><strong>Model downloads</strong> from inside the app, checked before they are used.</li>
<li><strong>Delivery presets</strong>: Grading master, HDR10 movie, Broadcast HDR; Custom… for everything else.</li>
<li><strong>HDR10 metadata</strong>: MaxCLL / MaxFALL on the PQ output, mastering display, Dolby Vision XML.</li>
<li><strong>PQ 1000 / 2000 / 4000, HLG and SDR 709</strong> outputs; ARRI LogC4 and ACEScct copies; DPX with full headers; ProRes 4444 XQ.</li>
<li><strong>Per-shot looks</strong> with Copy / Paste, and a <strong>batch queue</strong>.</li>
<li><strong>Check tab</strong> and scopes beside the picture, with RGB parade and gamut.</li>
<li><strong>Reference stills</strong>, <strong>EDL conform</strong>, timecode and reel in every render.</li>
<li><strong>Safer renders</strong>: files are written as .part then renamed; quitting asks while a render runs.</li>
<li><strong>Open…</strong> takes movies (MOV, MP4, MXF and more) as well as stills.</li>
<li><strong>Apache 2.0</strong>: the code is now free for commercial use; each model states its own licence.</li>
</ul>
<p class="small">Full notes on the <a class="textlink" href="{REL}">release page</a>.</p>
</section>
'''

hlp = src[src.index('<section class="page" id="help"'):src.index('<footer>')]
hlp = hlp.replace('05 / Keep moving', '09 / Keep moving')
# Replace Beta-2-specific answers with current ones.
hlp = re.sub(r'<details open><summary>The result looks almost unchanged.*?</details>',
 '<details open><summary>The result looks almost unchanged. Is it working?</summary><p>Some shots have little clipping, and RUDRA keeps everything the SDR already holds. Choose <strong>View ▾ → Invented pixels</strong> to see where it rebuilt light, check the status bar says the model is ready, and try <strong>Strength</strong> above 100 %.</p></details>', hlp, flags=re.S)
hlp = re.sub(r'<details><summary>Does Beta 2 contain a new model\?.*?</details>',
 '<details><summary>Where did Reconstruct, Grade and the probe go?</summary><p>They are in <strong>Full</strong>. Click Full in the toolbar, or <strong>Advanced →</strong> on the Look tab. Simple keeps only what most shots need.</p></details>'
 '<details><summary>How do I see the tips again?</summary><p><strong>Help → Show tips</strong>. <strong>Help → Getting started</strong> still has the longer five-page guide.</p></details>'
 '<details><summary>A model download failed.</summary><p>Click <strong>Download</strong> again: it resumes. A file that fails its checksum is deleted and fetched again. Behind a proxy, download on another computer and use <strong>Add folder…</strong>.</p></details>', hlp, flags=re.S)
hlp = re.sub(r'<details><summary>The command-line version says 0.1.0.*?</details>', '', hlp, flags=re.S)
hlp = hlp.replace('Can I use Beta 2 commercially?', 'Can I use RUDRA commercially?')
hlp = re.sub(r'<p>The native code uses PolyForm Noncommercial 1.0.0.*?</p>', '<p>Yes. From 29 October 2026 the RUDRA code is licensed under the Apache License 2.0, which allows commercial use. Each model has its own licence, shown in Model packages before you download it; the Beta 2 weights stay non-commercial because of their training data. Questions: <a class="textlink" href="https://www.fxtdstudios.com">www.fxtdstudios.com</a>.</p>', hlp, flags=re.S)
hlp = hlp.replace(f'<a class="button outline" href="https://github.com/FXTD-Studios/RUDRA/releases/tag/v0.9.0-beta.2">Release notes ↗</a>',
                  f'<a class="button outline" href="{REL}">Release notes ↗</a><a class="button outline" href="https://www.fxtdstudios.com">www.fxtdstudios.com ↗</a>')
hlp = re.sub(r'/blob/v[0-9][^/"]*/STATUS\.md', f'/blob/{SRC_TAG}/docs/STATUS.md', hlp)
hlp = re.sub(r'/blob/v[0-9][^/"]*/', f'/blob/{SRC_TAG}/', hlp)
hlp = hlp.replace('https://huggingface.co/fxtdstudios/RUDRA"', 'https://huggingface.co/fxtdstudios/rudra-models"')
hlp = re.sub(r'(?:\s*&nbsp;\s*/\s*&nbsp;\s*)?<a[^>]*paper/main\.pdf[^>]*>[^<]*</a>', '', hlp)
hlp = hlp.replace(' The desktop app’s Core ML inference is a separate capability.', '')
hlp = hlp.replace('<a class="button outline" href="https://github.com/FXTD-Studios/RUDRA/releases">Release notes ↗</a>', f'<a class="button outline" href="{REL}">Release notes ↗</a>')
hlp = re.sub(r'<p[^>]*>The paper predates[^<]*</p>', '<p class="small">This guide does not claim independent HDR-display or clean-machine certification.</p>', hlp)
if 'Can I use RUDRA on a Mac?' not in hlp:
    hlp = hlp.replace('<details><summary>Can I train on a Mac?', '<details><summary>Can I use RUDRA on a Mac?</summary><p>Yes: the 1.0.1 DMG runs on Apple silicon with the analytic reconstruction and every Look, Check and Deliver tool. The learned ACES 2.0 model and AI fill are Windows features for now.</p></details><details><summary>Can I train on a Mac?', 1)
foot = f'<footer><span>FXTD STUDIOS / RUDRA DOCUMENTATION · <a href="https://www.fxtdstudios.com">WWW.FXTDSTUDIOS.COM</a></span><span>{VER.upper()} · USER GUIDE</span></footer></main></div></body></html>'
out = head + intro + install + models + workspace + first + export + training + news + hlp + foot
(ARGS.out / 'index.html').write_text(out, encoding='utf-8')
print(f"wrote {ARGS.out / 'index.html'} ({len(out)} bytes) for {VER}")
print(len(out))
