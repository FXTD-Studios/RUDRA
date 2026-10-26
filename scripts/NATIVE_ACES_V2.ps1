<#
.SYNOPSIS
  ACES v2 staged pipeline on Windows: export the package, build the native
  stages with LibTorch + OpenColorIO 2.5.2 + OpenCV, run the unit tests, prove
  C++ == Python stage by stage, then time and run the native master.

    powershell -ExecutionPolicy Bypass -File scripts\NATIVE_ACES_V2.ps1
    powershell -ExecutionPolicy Bypass -File scripts\NATIVE_ACES_V2.ps1 -OcioPrefix C:\deps\ocio-2.5.2
    powershell -ExecutionPolicy Bypass -File scripts\NATIVE_ACES_V2.ps1 -Checkpoint <frozen.pt> -SkipBuild

  OpenColorIO 2.5.2 (the checkpoint contract pins it; configure refuses others):
    -OcioPrefix <install prefix with lib\cmake\OpenColorIO and bin\*.dll>, or
    VCPKG_ROOT set: `vcpkg install opencolorio:x64-windows` is run (classic mode).
  Output: reports\aces_v2_<stamp>\ (parity.json, master\, timing, log). Exit 1 on any failure.
  Signing is NOT part of this script (release step, last).
#>
param(
    [string]$Checkpoint = "G:\rudra_training\partial_pcgrad_duration6400_retry1_20261025\step6400.pt",
    [string]$Frames = "G:\datasets\corpora\aces_consistent_exposures_v2_20261024\val",
    [int]$ParityFrames = 6,
    [string]$Build = "build\native_aces_v2",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$OcioPrefix = "",
    [string]$OpenCvVersion = "4.10.0",
    [string]$OrtVersion = "1.22.0",   # ONNX Runtime DirectML package from tmp\native_deps (NATIVE_GATE_A.ps1 fetches it)
    [string]$Device = "cuda",
    [switch]$SkipBuild,
    [switch]$SkipExport,
    [switch]$InstallBuildTools,
    [switch]$NoLuminanceGuard          # default: appearance constraint also forbids new negative luminance
)
$Guard = @(); if (-not $NoLuminanceGuard) { $Guard = @('--forbid-new-negative-y') }
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent   # the repo this script sits in
Set-Location -LiteralPath $Repo
. (Join-Path $PSScriptRoot "native_toolchain.ps1")
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$Out = Join-Path $Repo "reports\aces_v2_$Stamp"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Deps = Join-Path $Repo "tmp\native_deps"
New-Item -ItemType Directory -Force -Path $Deps | Out-Null
$Package = Join-Path $Repo "dist\models\aces_v2_$((Get-Item $Checkpoint).Directory.Name)"
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; Write-Host "report folder: $Out"; exit 1 }
Start-Transcript -Path (Join-Path $Out "log.txt") | Out-Null

# ---------------------------------------------------------------------------
Say "Python, torch, OCIO"
$info = & $Python -c "import torch, PyOpenColorIO as o; print(torch.__version__); print(torch.utils.cmake_prefix_path); print(o.__version__); print(int(torch.cuda.is_available()))"
if ($LASTEXITCODE -ne 0) { Fail "python with torch and opencolorio not found ($Python)" }
$TorchVersion, $TorchCMake, $PyOcio, $CudaAvail = $info
if ($PyOcio -ne "2.5.2") { Fail "Python OpenColorIO is $PyOcio; the checkpoint contract needs 2.5.2 (pip install opencolorio==2.5.2)" }
$TorchRoot = Split-Path (Split-Path $TorchCMake)
$TorchLib = Join-Path $TorchRoot "lib"
if (-not (Test-Path (Join-Path $TorchRoot "include\torch\script.h"))) { Fail "torch headers not found under $TorchRoot" }
if ($Device -eq "cuda" -and $CudaAvail -ne "1") { Write-Host "no CUDA GPU visible to torch: timing on CPU" -ForegroundColor Yellow; $Device = "cpu" }
Write-Host "torch $TorchVersion ($TorchRoot), Python OCIO $PyOcio, device $Device"

# ---------------------------------------------------------------------------
if (-not $SkipExport) {
    Say "Export the ACES v2 package (G1 TorchScript + manifest; Python staged parity inside)"
    if (Test-Path $Package) { Remove-Item -Recurse -Force $Package }
    # The ONNX export needs onnx (writing) and onnxruntime (CPU check). Install them into this Python if missing;
    # if that is impossible, export without ONNX (the DirectML route is then skipped).
    $NoOnnx = @()
    & $Python -c "import onnx, onnxruntime" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "onnx / onnxruntime missing in ${Python}; installing" -ForegroundColor Yellow
        & $Python -m pip install --disable-pip-version-check onnx onnxruntime
        & $Python -c "import onnx, onnxruntime" 2>$null
        if ($LASTEXITCODE -ne 0) { Write-Host "onnx / onnxruntime unavailable: exporting without ONNX (DirectML route skipped)" -ForegroundColor Yellow; $NoOnnx = @('--no-onnx') }
    }
    $vals = Get-ChildItem $Frames -Filter "*_source.png" | Sort-Object Name | Select-Object -First 2
    & $Python tools\export_native_v2.py --checkpoint $Checkpoint --out $Package --frames @($vals.FullName) @Guard @NoOnnx
    if ($LASTEXITCODE -ne 0) { Fail "export_native_v2.py (TorchScript or staged parity failed)" }
}
if (-not (Test-Path (Join-Path $Package "manifest.json"))) { Fail "no package at $Package" }

# ---------------------------------------------------------------------------
Say "OpenColorIO 2.5.2 (C++)"
if (-not $OcioPrefix) {
    if (-not $env:VCPKG_ROOT) { Fail "pass -OcioPrefix <OCIO 2.5.2 install> or set VCPKG_ROOT" }
    $vcpkg = Join-Path $env:VCPKG_ROOT "vcpkg.exe"
    & $vcpkg install "opencolorio:x64-windows"
    if ($LASTEXITCODE -ne 0) { Fail "vcpkg install opencolorio" }
    $OcioPrefix = Join-Path $env:VCPKG_ROOT "installed\x64-windows"
}
if (-not (Test-Path (Join-Path $OcioPrefix "share\opencolorio")) -and -not (Test-Path (Join-Path $OcioPrefix "lib\cmake\OpenColorIO"))) {
    Fail "no OpenColorIO CMake config under $OcioPrefix"
}

# OpenCV (still decode for the aces-v2 command), same pack as the phase 4 gate.
$CvRoot = Join-Path $Deps "opencv-$OpenCvVersion"
$CvBuild = Join-Path $CvRoot "opencv\build"
if (-not (Test-Path (Join-Path $CvBuild "OpenCVConfig.cmake"))) {
    $exe = Join-Path $Deps "opencv-$OpenCvVersion-windows.exe"
    Invoke-WebRequest -Uri "https://github.com/opencv/opencv/releases/download/$OpenCvVersion/opencv-$OpenCvVersion-windows.exe" -OutFile $exe -UseBasicParsing
    $p = Start-Process -FilePath $exe -ArgumentList @("-o`"$CvRoot`"", "-y") -WindowStyle Hidden -Wait -PassThru
    if ($p.ExitCode -ne 0) { Fail "OpenCV extract" }
    Remove-Item $exe
}
$CvLib = Get-ChildItem (Join-Path $CvBuild "x64") -Directory | Where-Object { $_.Name -match "^vc\d+$" } |
         Sort-Object { [int]($_.Name.Substring(2)) } -Descending | Select-Object -First 1
$CvConfigDir = Join-Path $CvLib.FullName "lib"
$CvDll = Get-ChildItem (Join-Path $CvBuild "x64") -Recurse -Filter "opencv_world*.dll" | Where-Object { $_.Name -notmatch "d\.dll$" } | Select-Object -First 1

$OrtRoot = Join-Path $Deps "ort-root-$OrtVersion"
$OrtArgs = @()
if (Test-Path (Join-Path $OrtRoot "include\onnxruntime_cxx_api.h")) {
    $OrtArgs = @("-DRUDRA_WITH_ONNXRUNTIME=ON", "-DONNXRUNTIME_ROOT=$OrtRoot")
    Write-Host "ONNX Runtime $OrtVersion (DirectML) found: the DirectML route is built and timed too"
} else { Write-Host "no $OrtRoot (run NATIVE_GATE_A.ps1 once to fetch it): DirectML route skipped" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Say "Configure and build (Release): rudra_tests, rudra-native, aces_v2_parity"
    $tc = Find-NativeToolchain
    if (-not $tc -and $InstallBuildTools) { if (-not (Install-NativeBuildTools)) { Fail "Build Tools install" }; $tc = Find-NativeToolchain }
    if (-not $tc) { Fail "no C++ toolchain" }
    $cmake = $tc.CMake
    Reset-StaleCMakeCache $Build $tc.Generator
    & $cmake -S native -B $Build -G $tc.Generator -A x64 `
        -DRUDRA_BUILD_TESTS=ON -DRUDRA_BUILD_CLI=ON -DRUDRA_BUILD_APP=OFF `
        -DRUDRA_WITH_LIBTORCH=ON "-DRUDRA_TORCH_ROOT=$TorchRoot" -DRUDRA_WITH_OCIO=ON `
        -DRUDRA_WITH_OPENCV=ON "-DOpenCV_DIR=$CvConfigDir" "-DCMAKE_PREFIX_PATH=$OcioPrefix" @OrtArgs
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure (OpenColorIO must be exactly 2.5.2)" }
    & $cmake --build $Build --config Release --parallel --target rudra_tests rudra-native aces_v2_parity aces_v2_constraint_parity
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}
$Tests = Join-Path $Build "tests\Release\rudra_tests.exe"
$Cli = Join-Path $Build "cli\Release\rudra-native.exe"
$Par = Get-ChildItem $Build -Recurse -Filter "aces_v2_parity.exe" | Select-Object -First 1
if (-not (Test-Path $Cli) -or -not $Par) { Fail "binaries not built" }
foreach ($d in @((Split-Path $Cli), (Split-Path $Tests), $Par.DirectoryName)) {
    Copy-Item (Join-Path $OcioPrefix "bin\*.dll") $d -Force -ErrorAction SilentlyContinue
    if ($CvDll) { Copy-Item $CvDll.FullName $d -Force }
    if ($OrtArgs.Count -and (Test-Path (Join-Path $Package "rudra_g1.onnx"))) {
        Copy-Item "$OrtRoot\lib\*.dll" $d -Force   # beside the exe: System32 has an older onnxruntime.dll
        $dml = Get-ChildItem $Deps -Recurse -Filter DirectML.dll | Where-Object { $_.FullName -match "x64-win" } | Select-Object -First 1
        if ($dml) { Copy-Item $dml.FullName $d -Force }
    }
}
$env:PATH = "$TorchLib;$env:PATH"

# ---------------------------------------------------------------------------
Say "Unit tests (AcesV2.* and the full suite)"
& $Tests "--gtest_filter=AcesV2.*" | Tee-Object -FilePath (Join-Path $Out "tests_aces_v2.txt")
if ($LASTEXITCODE -ne 0) { Fail "AcesV2 unit tests" }
& $Tests | Out-File (Join-Path $Out "tests_all.txt")
if ($LASTEXITCODE -ne 0) {
    # Re-run only the failed tests once, to separate timing-sensitive tests from real failures.
    # Both results stay in the report folder; a test that fails twice stops the run.
    $failed = Select-String -Path (Join-Path $Out "tests_all.txt") -Pattern '^\[  FAILED  \] (\S+\.\S+) \(' |
              ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique
    if (-not $failed) { Fail "native test suite failed: see tests_all.txt" }
    Write-Host "re-running $($failed.Count) failed test(s) once: $($failed -join ', ')" -ForegroundColor Yellow
    & $Tests "--gtest_filter=$($failed -join ':')" | Tee-Object -FilePath (Join-Path $Out "tests_rerun.txt")
    if ($LASTEXITCODE -ne 0) { Fail "native test suite failed twice: see tests_all.txt and tests_rerun.txt" }
    Write-Host "passed on re-run (timing-sensitive): $($failed -join ', ')" -ForegroundColor Yellow
}

Say "C++ vs Python, stage by stage ($ParityFrames frames)"
$pf = Get-ChildItem $Frames -Filter "*_source.png" | Sort-Object Name | Select-Object -First $ParityFrames
& $Python native\tools\aces_v2_parity\parity_check.py --exe $Par.FullName --checkpoint $Checkpoint `
    --g1 (Join-Path $Package "rudra_g1.ts") --out (Join-Path $Out "parity.json") @Guard @($pf.FullName)
if ($LASTEXITCODE -ne 0) { Fail "native parity (see parity.json)" }

Say "Native master: rudra-native aces-v2 ($Device)"
& $Cli aces-v2 $Package $Frames --out (Join-Path $Out "master") --device $Device --ocio-threads 8
if ($LASTEXITCODE -ne 0) { Fail "rudra-native aces-v2 (exit $LASTEXITCODE; 3 = appearance constraint errors)" }

if ($OrtArgs.Count -and (Test-Path (Join-Path $Package "rudra_g1.onnx"))) {
    Say "Native master through ONNX Runtime + DirectML (tolerance route)"
    & $Cli aces-v2 $Package $Frames --out (Join-Path $Out "master_directml") --runtime onnxruntime --device directml --ocio-threads 8
    if ($LASTEXITCODE -ne 0) { Fail "rudra-native aces-v2 --runtime onnxruntime --device directml" }
}

Say "Native master of a video (every frame, <stem>_<NNNNNN>.exr)"
# ffmpeg and ffprobe: the PATH first, then the pack the app bundles (tmp\native_deps).
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    $ffd = Get-ChildItem "tmp\native_deps" -Directory -Filter "ffmpeg-*-full_build" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ffd) { $env:PATH = "$(Join-Path $ffd.FullName 'bin');$env:PATH" }
}
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    $first = Get-ChildItem $Frames -Filter "*_source.png" | Sort-Object Name | Select-Object -First 1
    $clip = Join-Path $Out "video_smoke.mp4"
    & ffmpeg -v error -y -loop 1 -i $first.FullName -frames:v 12 -r 24 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" `
        -c:v libx264 -crf 12 -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 $clip
    if ($LASTEXITCODE -ne 0) { Fail "ffmpeg could not make the smoke clip" }
    $vout = Join-Path $Out "master_video"
    & $Cli aces-v2 $Package $clip --out $vout --device $Device --ocio-threads 8
    if ($LASTEXITCODE -ne 0) { Fail "rudra-native aces-v2 on a video (exit $LASTEXITCODE)" }
    $n = (Get-ChildItem $vout -Filter "video_smoke_*.exr").Count
    if ($n -ne 12) { Fail "video master wrote $n EXRs, expected 12" }
    Write-Host "video: 12 frames mastered"
} else {
    Write-Host "no ffmpeg on the PATH or in tmp\native_deps: video check skipped" -ForegroundColor Yellow
}

Say "Python export of the same frames (reference timing and EXRs)"
& $Python tools\rudra_aces_export.py $Frames --checkpoint $Checkpoint --out (Join-Path $Out "python_master") `
    --colour-backtrack-all --clipped-smoothing hybrid64 --device $Device --ocio-threads 8 @Guard
if ($LASTEXITCODE -ne 0) { Fail "python reference export" }

Stop-Transcript | Out-Null
Write-Host "`nALL PASSED. Report: $Out" -ForegroundColor Green
Write-Host "Next: open master\*.exr in Resolve and Nuke (ACEScg, float32, rudra.reconstructed channel)."
