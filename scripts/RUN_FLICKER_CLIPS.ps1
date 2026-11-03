<#
.SYNOPSIS
  The T contract on your own footage: renders every movie in a folder through the
  learned ACES 2.0 route (rudra-native aces-v2, EXR masters only), then measures
  flicker on all of them (tools/measure_flicker.py: <= 0.02 EV every transition,
  the Hunyuan48 measure). The roadmap asks for at least 10 real clips: pans, fire,
  handheld, slow fades through clipping.

      powershell -ExecutionPolicy Bypass -File scripts\RUN_FLICKER_CLIPS.ps1 -Clips D:\flicker_clips
      ... -Clips D:\flicker_clips -MaxFrames 96 -Device cuda

  Renders go to <Out>\<clip name>\ and are kept (rerunning skips clips already rendered).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Clips,
    [string]$Out = "G:\rudra_training\flicker_clips_$(Get-Date -Format yyyyMMdd)",
    [string]$Package = "",   # default: where the app downloads it (File > Model packages)
    [string]$Cli = "build\native_aces_v2\cli\Release\rudra-native.exe",
    [string]$Device = "cuda",
    [int]$MaxFrames = 120,
    [string]$Python = ".venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Python)) { Fail "no ${Python}: the repo's .venv runs the measurement and holds the LibTorch the engine was built against (pass -Python)" }
if (-not (Test-Path $Cli)) { Fail "no rudra-native at $Cli (build it with NATIVE_ACES_V2.ps1, or pass -Cli)" }
# The engine's DLLs, as RUN_APP_ACES.ps1 sets them: LibTorch from the .venv's torch\lib first on PATH
# (without it Windows stops the exe silently with -1073741515), then ffmpeg from tmp\native_deps if the PATH has none.
$TorchLib = & $Python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $TorchLib)) { Fail "torch is not importable from $Python" }
$env:PATH = "$TorchLib;$env:PATH"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $ffd = Get-ChildItem "tmp\native_deps" -Directory -Filter "ffmpeg*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ffd) { $env:PATH = "$(Join-Path $ffd.FullName 'bin');$env:PATH" }
}
$ErrorActionPreference = "Continue"   # the engine writes progress to stderr; Windows PowerShell would stop on it
$probe = (& $Cli aces-v2 --help 2>&1 | Out-String)
if ($probe -notmatch "usage: rudra-native aces-v2") { Write-Host $probe; Fail "the engine does not start (exit $LASTEXITCODE; -1073741515 = a DLL was not found)" }
$ErrorActionPreference = "Stop"
if (-not $Package) {   # the app's AppLocalDataLocation\models (Qt: <LOCALAPPDATA>\<organisation>\<application>)
    $Package = @("$env:LOCALAPPDATA\FXTD Studios\RUDRA\models\RUDRA_ACES_2", "$env:LOCALAPPDATA\RUDRA\models\RUDRA_ACES_2", "dist\models\RUDRA_ACES_2") |
        Where-Object { Test-Path "$_\manifest.json" } | Select-Object -First 1
}
if (-not $Package -or -not (Test-Path "$Package\manifest.json")) { Fail "no RUDRA_ACES_2 model package found (pass -Package; File > Model packages shows where it is)" }
Write-Host "Model package: $Package"
$movies = Get-ChildItem $Clips -File | Where-Object { $_.Extension -match '^\.(mov|mp4|mxf|mkv|avi)$' } | Sort-Object Name
if ($movies.Count -eq 0) { Fail "no movies in $Clips" }
if ($movies.Count -lt 10) { Write-Host "Note: $($movies.Count) clips; the roadmap asks for at least 10." -ForegroundColor Yellow }
New-Item -ItemType Directory -Force $Out | Out-Null
$list = @()
foreach ($m in $movies) {
    $dir = Join-Path $Out $m.BaseName
    if (-not (Test-Path (Join-Path $dir "provenance.json"))) {
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }   # a stopped render of this clip
        Write-Host "`n== Render $($m.Name) (first $MaxFrames frames)" -ForegroundColor Cyan
        $ErrorActionPreference = "Continue"
        & $Cli aces-v2 $Package $m.FullName --out $dir --device $Device --exr half --range "0:$($MaxFrames - 1)" 2>&1 | Select-Object -Last 3
        $rc = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($rc -ne 0) { Write-Host "  render failed (exit $rc); skipped" -ForegroundColor Yellow; continue }
    }
    $list += @{ name = $m.BaseName; clip = $m.FullName; masters = $dir; first = 0 }
}
$listFile = Join-Path $Out "clips.json"
[IO.File]::WriteAllText($listFile, (ConvertTo-Json -InputObject @($list) -Depth 4), (New-Object Text.UTF8Encoding $false))
Write-Host "`n== Flicker" -ForegroundColor Cyan
& $Python tools/measure_flicker.py --list $listFile --out (Join-Path $Out "flicker.json")
exit $LASTEXITCODE
