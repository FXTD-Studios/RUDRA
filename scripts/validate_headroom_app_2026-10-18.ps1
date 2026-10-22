<#
.SYNOPSIS
  Windows acceptance for the headroom preview in the app (ledger E1/E2, partly
  E3/E4): build, tests, then the real decoder, viewer and export end to end.

.DESCRIPTION
  Nothing is committed, pushed or published. The evidence goes to
  reports\headroom_app_20261018\, and it should be recorded in
  docs\PRODUCTION_PROGRESS.md before REMAINING_WORK.md changes.
    1. Build RUDRA, rudra_tests and rudra_app_tests (Release) in the existing
       build\native_phase3 tree (Qt 6.8, viewer, ONNX Runtime, OpenCV).
    2. The headroom native tests (driver, export, frames, session), then the
       full native and app suites, each with JSON output.
    3. A v1-contract clip made with FFmpeg: 1920x1080, limited-range Rec.709,
       10-bit 4:4:4 FFV1, 48 frames.
    4. RUDRA.exe --headroom-check on that clip in a real window: paused frame,
       seeks, grade and undo (bit-identical), playback timing, a transactional
       HDR10 export with provenance, viewer upload/present/readback, back to
       legacy.

.EXAMPLE
  .\scripts\validate_headroom_app_2026-10-18.ps1
  .\scripts\validate_headroom_app_2026-10-18.ps1 -SkipBuild -Movie D:\clips\my_v1_clip.mkv
#>
[CmdletBinding()]
param(
    [string]$Build = "build\native_phase3",
    [string]$Movie = "",
    [int]$TimeoutMs = 300000,
    [switch]$SkipBuild,
    [switch]$SkipSuites
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Build = Join-Path $Repo $Build
$Out = Join-Path $Repo "reports\headroom_app_20261018"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
$summary = [ordered]@{ started = (Get-Date).ToString("o"); build = $Build }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Fail "ffmpeg is not on PATH" }
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) { Fail "ffprobe is not on PATH" }

if (-not $SkipBuild) {
    Say "Build (CMake re-runs itself for the new sources)"
    & cmake --build $Build --config Release --parallel --target RUDRA rudra-native rudra_tests rudra_app_tests 2>&1 |
        Tee-Object -FilePath (Join-Path $Out "build.log")
    if ($LASTEXITCODE -ne 0) { Fail "build (see $Out\build.log)" }
}
$App = Join-Path $Build "app\Release\RUDRA.exe"
# rudra_tests.exe has only ONNX Runtime beside it: Qt and OpenCV come from the
# app's deployed folder (without this it exits 0xC0000135, DLL not found).
$env:PATH = (Join-Path $Build "app\Release") + ";" + $env:PATH
$AppTests = Join-Path $Build "app\Release\rudra_app_tests.exe"
$Tests = Join-Path $Build "tests\Release\rudra_tests.exe"
foreach ($exe in $App, $AppTests, $Tests) { if (-not (Test-Path $exe)) { Fail "missing $exe" } }

Say "Headroom native tests"
& $Tests --gtest_filter="HeadroomPreviewDriver.*:HeadroomExport.*:HeadroomFrames.*:HeadroomSession.*" `
    --gtest_output="json:$Out\headroom_native.json" 2>&1 | Tee-Object -FilePath (Join-Path $Out "headroom_native.log")
$summary.headroom_native_exit = $LASTEXITCODE

if (-not $SkipSuites) {
    Say "Full native suite"
    & $Tests --gtest_output="json:$Out\native.json" 2>&1 | Tee-Object -FilePath (Join-Path $Out "native.log") | Out-Null
    $summary.native_exit = $LASTEXITCODE
    Say "Full app suite (offscreen)"
    $env:QT_QPA_PLATFORM = "offscreen"
    & $AppTests --gtest_output="json:$Out\app.json" 2>&1 | Tee-Object -FilePath (Join-Path $Out "app.log") | Out-Null
    $summary.app_exit = $LASTEXITCODE
    Remove-Item Env:\QT_QPA_PLATFORM
}

if (-not $Movie) {
    Say "v1-contract clip"
    # setparams tags the frames themselves: lavfi frames are untagged, and
    # -color_trc alone did not reach the stream (the first run was refused for it).
    $Movie = Join-Path $Out "v1_clip_1080p_48f_tagged.mkv"
    if (-not (Test-Path $Movie)) {
        & ffmpeg -v error -nostdin -f lavfi -i "testsrc2=size=1920x1080:rate=24" `
            -vf "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709" -frames:v 48 -pix_fmt yuv444p10le `
            -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:v ffv1 -level 3 $Movie
        if ($LASTEXITCODE -ne 0) { Fail "could not make the clip" }
    }
}
$summary.movie = $Movie

Say "RUDRA --headroom-check (a real window: decoder, viewer, export)"
$Report = Join-Path $Out ("headroom_check_" + (Get-Date -Format "HHmmss") + ".json")
& $App --headroom-check $Report --movie $Movie --timeout-ms $TimeoutMs 2>&1 | Tee-Object -FilePath (Join-Path $Out "headroom_check.log")
$summary.headroom_check_exit = $LASTEXITCODE
$summary.headroom_check_report = $Report

$summary.finished = (Get-Date).ToString("o")
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Out "summary.json")
Say "Summary"
$summary | Format-Table -AutoSize | Out-String | Write-Host
if ($summary.headroom_native_exit -ne 0 -or $summary.headroom_check_exit -ne 0) {
    Write-Host "Something failed: read $Out (the check's JSON names the step and the reason)." -ForegroundColor Yellow
    exit 1
}
Write-Host "Headroom acceptance PASS on this machine. Record $Out in docs\PRODUCTION_PROGRESS.md." -ForegroundColor Green
