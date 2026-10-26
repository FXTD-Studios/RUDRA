<#
.SYNOPSIS
  RUDRA with the learned ACES 2.0 master as one folder for use inside FXTD Studios:
  copy it to any Windows x64 PC (NVIDIA GPU for CUDA; DirectML or CPU otherwise)
  and run RUDRA.exe. No Python, no .venv, no PATH setup, nothing installed.

    powershell -ExecutionPolicy Bypass -File scripts\BUILD_INTERNAL_ACES_BUNDLE.ps1
    powershell -ExecutionPolicy Bypass -File scripts\BUILD_INTERNAL_ACES_BUNDLE.ps1 -Zip
    powershell -ExecutionPolicy Bypass -File scripts\BUILD_INTERNAL_ACES_BUNDLE.ps1 -TestDevice cpu

  Makes dist\internal\RUDRA-ACES-internal-<date>\ from the builds that already exist
  (run RUN_APP_ACES.ps1 first; nothing is rebuilt here):
    RUDRA.exe + Qt (windeployqt)              build\native_gate_b
    rudra-native.exe (ACES v2) + its DLLs      build\native_aces_v2 (OCIO 2.5.2, ORT/DirectML, OpenCV, OpenEXR)
    LibTorch DLLs (CUDA included)              .venv torch\lib
    MSVC runtime                               Visual Studio's VC\Redist
    models\<newest dist\models\aces_v2_*>      the model package
    ffmpeg\ (if tmp\native_deps has it)        movies in the app (GPL build, run as a separate program)
    INTERNAL USE.md, LICENSE, NOTICE, THIRD_PARTY_LICENSES.md
  Then it proves the folder stands alone: with PATH reduced to Windows' own folders it
  asks the bundled engine for its ACES usage and masters two frames.

  INTERNAL USE ONLY. Not signed (SmartScreen warns once), not a public release:
  the model package says release_authorized false and the weights licence is not decided.
#>
param(
    [string]$AppBuild = "build\native_gate_b",
    [string]$AcesBuild = "build\native_aces_v2",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Package = "",
    [string]$TestFrames = "G:\datasets\corpora\aces_consistent_exposures_v2_20261024\val",
    [ValidateSet("cuda", "directml", "cpu")] [string]$TestDevice = "cuda",
    [switch]$NoFfmpeg,
    [switch]$Zip
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent   # the repo this script sits in
Set-Location -LiteralPath $Repo
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

# ---- what goes in ------------------------------------------------------------
$App = Join-Path $AppBuild "app\Release\RUDRA.exe"
$EngineDir = Join-Path $AcesBuild "cli\Release"
$Engine = Join-Path $EngineDir "rudra-native.exe"
foreach ($f in @($App, $Engine)) { if (-not (Test-Path $f)) { Fail "missing $f (run RUN_APP_ACES.ps1 first)" } }
$Cache = Join-Path $AppBuild "CMakeCache.txt"
$QtDir = ((Select-String -Path $Cache -Pattern '^Qt6_DIR:PATH=(.+)$').Matches[0].Groups[1].Value)
$QtBin = (Resolve-Path (Join-Path $QtDir "..\..\..\bin")).Path
$Deploy = Join-Path $QtBin "windeployqt.exe"
if (-not (Test-Path $Deploy)) { Fail "no windeployqt in $QtBin" }
if (-not $Package) {
    $p = Get-ChildItem "dist\models" -Directory -Filter "aces_v2_*" -ErrorAction SilentlyContinue |
         Where-Object { Test-Path (Join-Path $_.FullName "manifest.json") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $p) { Fail "no dist\models\aces_v2_* package" }
    $Package = $p.FullName
}
$Manifest = Get-Content (Join-Path $Package "manifest.json") -Raw | ConvertFrom-Json
if (-not $Manifest.parity_pass) { Fail "the package failed its export parity check" }
$TorchLib = & $Python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $TorchLib)) { Fail "LibTorch's lib folder not found through $Python" }
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = if (Test-Path $vswhere) { & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath }
$crt = if ($vsPath) {
    Get-ChildItem (Join-Path $vsPath "VC\Redist\MSVC") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "^\d+\." } | Sort-Object { [version]$_.Name } -Descending |
        ForEach-Object { Get-ChildItem (Join-Path $_.FullName "x64") -Directory -Filter "Microsoft.VC14*.CRT" -ErrorAction SilentlyContinue } |
        Select-Object -First 1
}
if (-not $crt) { Fail "the MSVC runtime (VC\Redist\MSVC\<version>\x64\Microsoft.VC14*.CRT) was not found" }
$Ff = Get-ChildItem "tmp\native_deps" -Directory -Filter "ffmpeg-*-full_build" -ErrorAction SilentlyContinue | Select-Object -First 1

$Name = "RUDRA-ACES-internal-$(Get-Date -Format 'yyyy-MM-dd')"
$Out = Join-Path $Repo "dist\internal\$Name"
if (Test-Path $Out) { Fail "$Out exists: rename or remove it first" }

# ---- assemble ----------------------------------------------------------------
Say "Assemble $Out"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Copy-Item $App $Out
& $Deploy --release --no-translations (Join-Path $Out "RUDRA.exe") | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "windeployqt" }
# The ACES engine and every DLL beside it (OCIO, OpenEXR, OpenImageIO, ONNX Runtime, DirectML, OpenCV ...).
Copy-Item $Engine $Out
Get-ChildItem $EngineDir -Filter "*.dll" | ForEach-Object {
    $dest = Join-Path $Out $_.Name
    if (-not (Test-Path $dest)) { Copy-Item $_.FullName $dest }
    elseif ((Get-FileHash $dest).Hash -ne (Get-FileHash $_.FullName).Hash) { Write-Host "  kept the app's $($_.Name) (the engine's differs)" -ForegroundColor Yellow }
}
Say "LibTorch DLLs (large: CUDA and cuDNN included)"
Copy-Item (Join-Path $TorchLib "*.dll") $Out -Force
Copy-Item (Join-Path $crt.FullName "*.dll") $Out -Force
$Models = Join-Path $Out "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null
Copy-Item $Package $Models -Recurse
if ($Ff -and -not $NoFfmpeg) {
    $OutFf = Join-Path $Out "ffmpeg"
    New-Item -ItemType Directory -Force -Path $OutFf | Out-Null
    Copy-Item (Join-Path $Ff.FullName "bin\ffmpeg.exe"), (Join-Path $Ff.FullName "bin\ffprobe.exe") $OutFf
    if (Test-Path (Join-Path $Ff.FullName "LICENSE")) { Copy-Item (Join-Path $Ff.FullName "LICENSE") (Join-Path $OutFf "LICENSE.txt") }
}
foreach ($f in @("LICENSE", "NOTICE", "docs\THIRD_PARTY_LICENSES.md")) { if (Test-Path $f) { Copy-Item $f $Out } }
$pkgName = Split-Path $Package -Leaf
$commit = (git -c safe.directory=* rev-parse --short HEAD 2>$null)
@"
# RUDRA with the learned ACES 2.0 master: INTERNAL USE ONLY

FXTD Studios internal build, $(Get-Date -Format 'yyyy-MM-dd'), from commit $commit plus local changes.
Not signed, not a public release. Do not send this folder or its model outside the studio.

## Use
1. Run RUDRA.exe. Windows SmartScreen warns once (unsigned): More info, Run anyway.
2. Deliver > Learned ACES 2.0 master... (Ctrl+Shift+E).
3. Pick a still, a folder of stills or a video (MP4, MOV, MKV, MXF...), a new output folder, and where it runs:
   GPU CUDA (NVIDIA; matches the reference exactly), GPU DirectML (any DX12 GPU; within 0.25 code), or CPU.
4. Start. Each still or video frame becomes a 32-bit float ACEScg EXR with a rudra.reconstructed channel
   (a video gives <name>_000000.exr, <name>_000001.exr, ...),
   plus provenance.json for the run.

The engine also runs on its own:
    rudra-native.exe aces-v2 models\$pkgName <image|folder|video> --out <new folder> --device cuda

## What it is
- Model: $pkgName (checkpoint $($Manifest.checkpoint_sha256.Substring(0,12))), ACES 2.0 SDR 100-nit view
  (OpenColorIO 2.5.2), hybrid64 clipped smoothing, 1-code appearance constraint, no new negative luminance.
- Shadows and everything not clipped: the exact ACES 2.0 inverse. Clipped highlights: rebuilt by the
  model; they are estimates, not recovered camera data. The rudra.reconstructed channel shows where.
- Checked on 42 unseen cases from our own camera footage (18 shots, native exposure +0/+1/+2 stops):
  better than the plain ACES inverse when the master is graded down 3 stops or up 3 stops, and on
  highlight error; at the delivered exposure (0 EV) its highlights sit about 0.5 code from the
  reference against 0.25 for the plain inverse (always within 1 code of the SDR source).
  Not release-qualified: the 0 EV grading check fails.

## Limits
- Videos are read with the bundled ffmpeg (BT.709 YCbCr unless tagged BT.2020); every frame is mastered,
  no audio is carried. HDR (PQ/HLG) sources are refused: this is an SDR-to-HDR master.
- Fully clipped areas are smoothed, not textured. Banding and compression artefacts are not removed.
- Have a colourist check a shot on a calibrated display before it goes to a client.
- The package says release_authorized: false. The weights licence is not decided yet.
"@ | Set-Content -Encoding utf8 (Join-Path $Out "INTERNAL USE.md")

# ---- prove it stands alone ---------------------------------------------------
Say "Self-test with PATH = Windows only"
$savedPath = $env:PATH
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
$ErrorActionPreference = "Continue"
$probe = (& (Join-Path $Out "rudra-native.exe") aces-v2 --help 2>&1 | Out-String)
$probeRc = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($probe -notmatch "usage: rudra-native aces-v2 <package>") {
    $env:PATH = $savedPath
    Write-Host $probe
    Fail ("the bundled engine does not start on its own (exit {0}; -1073741515 = a DLL is missing from the folder)" -f $probeRc)
}
$Self = Join-Path $env:TEMP "rudra_bundle_selftest_$(Get-Random)"
$Frames = Join-Path $Self "frames"
New-Item -ItemType Directory -Force -Path $Frames | Out-Null
Get-ChildItem $TestFrames -Filter "*_source.png" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -First 2 |
    ForEach-Object { Copy-Item $_.FullName $Frames }
if (-not (Get-ChildItem $Frames)) { $env:PATH = $savedPath; Fail "no test frames in $TestFrames" }
$runtime = if ($TestDevice -eq "directml") { "onnxruntime" } else { "libtorch" }
& (Join-Path $Out "rudra-native.exe") aces-v2 (Join-Path $Models $pkgName) $Frames --out (Join-Path $Self "master") `
    --runtime $runtime --device $TestDevice --ocio-threads 8
$rc = $LASTEXITCODE
$env:PATH = $savedPath
if ($rc -ne 0) { Fail "bundled engine master (exit $rc)" }
$prov = Get-Content (Join-Path $Self "master\provenance.json") -Raw | ConvertFrom-Json
if (-not $prov.appearance_constraint_pass) { Fail "self-test frames failed the appearance constraint" }
Remove-Item $Self -Recurse -Force
Write-Host "self-test: $($prov.frames.Count) frames on $TestDevice, appearance pass, no DLLs from outside the folder"

$size = (Get-ChildItem $Out -Recurse | Measure-Object Length -Sum).Sum / 1GB
Say ("Done: {0} ({1:N1} GB)" -f $Out, $size)
if ($Zip) {
    Say "ZIP (slow for a folder this size)"
    $zipPath = "$Out.zip"
    Compress-Archive -Path $Out -DestinationPath $zipPath -CompressionLevel Fastest
    Write-Host ("{0}  SHA-256 {1}" -f $zipPath, (Get-FileHash $zipPath -Algorithm SHA256).Hash)
}
Write-Host "Copy the folder to a studio PC and run RUDRA.exe. Read 'INTERNAL USE.md' first."
