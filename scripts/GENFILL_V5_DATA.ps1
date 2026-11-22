<#
.SYNOPSIS
  AI fill v5 training material (22 Nov 2026): collect CC0 HDR panoramas, render the Blender scenes, build the
  v5 clips. Registration: configs\experiments\research_genfill_v5_data_20261122.json. No training here.

    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V5_DATA.ps1 -Plan        # what it would download, by theme (no download)
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V5_DATA.ps1 -SmokeTest   # a few of each, in *_smoke folders (~15 min)
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V5_DATA.ps1              # all three steps (each resumes / skips what is done)
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V5_DATA.ps1 -Step scenes # one step: collect | scenes | build

  Needs: the repo's .venv, internet (polyhaven.com, ambientcg.com), Blender 4.2 or newer (found in Program Files,
  or -Blender <path to blender.exe>), the GPU free for the scenes and the encode, and about 60 GB free on G:.
#>
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Blender = "",
    [string]$Base = "G:\datasets\models\Wan2.1-VACE-1.3B-diffusers",
    [string]$Material = "G:\datasets\sources\genfill_v5",
    [string]$Clips = "G:\datasets\corpora\research_genfill_clips_v5_20261122",
    [double]$MaxGB = 30,
    [int]$Scenes = 200,
    [int]$Workers = 8,
    [ValidateSet("all", "collect", "scenes", "build")][string]$Step = "all",
    [switch]$Plan,
    [switch]$SmokeTest
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
$Reg = "configs\experiments\research_genfill_v5_data_20261122.json"
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Run([string]$exe, [string[]]$a, [string]$logFile) {
    $ErrorActionPreference = "Continue"
    if ($logFile) { & $exe @a 2>&1 | Tee-Object -FilePath $logFile -Append | Out-Host } else { & $exe @a 2>&1 | Out-Host }
    $rc = $LASTEXITCODE; $ErrorActionPreference = "Stop"
    return $rc
}
function FreeGB([string]$path) {
    $q = Split-Path -Qualifier $path
    return [math]::Round((Get-PSDrive ($q.TrimEnd(':'))).Free / 1GB, 1)
}
if (-not (Test-Path $Python)) { Fail "no $Python (the repo's venv)" }
if (-not (Test-Path $Reg)) { Fail "no registration $Reg" }
if ($SmokeTest) { $Material = "$Material`_smoke"; $Clips = "$Clips`_smoke"; $MaxGB = 1; $Scenes = 5 }
$Skies = Join-Path $Material "manifest.jsonl"
$SceneDir = Join-Path $Material "blender"
Write-Host ("G: free {0} GB (all three steps want about 60)" -f (FreeGB $Material))

# ---- 1. the CC0 panoramas ---------------------------------------------------
if ($Plan -or $Step -in @("all", "collect")) {
    $cargs = @("tools\collect_genfill_v5.py", "--dest", $Material, "--max-gb", $MaxGB)
    if ($SmokeTest) { $cargs += @("--per-theme", "2") }
    if ($Plan) {
        Say "Plan: what the collector would take (nothing downloaded)"
        $rc = Run $Python ($cargs + "--dry-run") ""
        exit $rc
    }
    Say "Collect CC0 panoramas (Poly Haven, ambientCG; up to $MaxGB GB) -> $Material"
    New-Item -ItemType Directory -Force -Path $Material | Out-Null
    $rc = Run $Python $cargs (Join-Path $Material "collect.log")
    if ($rc -ne 0) { Fail "collect (see $Material\collect.log)" }
    if (-not (Test-Path $Skies)) { Fail "nothing collected ($Skies missing)" }
    Write-Host ("panoramas: {0}" -f (Get-Content $Skies).Count)
}

# ---- 2. the Blender scenes --------------------------------------------------
if ($Step -in @("all", "scenes")) {
    if (-not $Blender) {
        $found = @()
        foreach ($root in @("$env:ProgramFiles\Blender Foundation", "${env:ProgramFiles(x86)}\Steam\steamapps\common\Blender")) {
            if (Test-Path $root) { $found += Get-ChildItem -Path $root -Recurse -Filter blender.exe -ErrorAction SilentlyContinue }
        }
        $Blender = ($found | Sort-Object { $_.Directory.Name } -Descending | Select-Object -First 1).FullName
    }
    if (-not $Blender -or -not (Test-Path $Blender)) { Fail "Blender not found: install Blender 4.2+ or pass -Blender <path to blender.exe>" }
    if (-not (Test-Path $Skies)) { Fail "no panoramas yet ($Skies): run -Step collect first" }
    $frames = 33
    $samples = if ($SmokeTest) { 32 } else { 96 }
    Say "Render $Scenes Blender scenes ($frames frames each, $Blender) -> $SceneDir"
    New-Item -ItemType Directory -Force -Path $SceneDir | Out-Null
    $bargs = @("-b", "--factory-startup", "--python-exit-code", "1", "-P", "tools\blender\genfill_v5_scenes.py", "--",
               "--skies", $Skies, "--out", $SceneDir, "--count", $Scenes, "--frames", $frames, "--samples", $samples, "--device", "GPU")
    $rc = Run $Blender $bargs (Join-Path $SceneDir "render.log")
    if ($rc -ne 0) { Fail "Blender (see $SceneDir\render.log)" }
    $man = Join-Path $SceneDir "manifest.jsonl"
    Write-Host ("scenes rendered: {0}" -f $(if (Test-Path $man) { (Get-Content $man).Count } else { 0 }))
}

# ---- 3. the v5 clips --------------------------------------------------------
if ($Step -in @("all", "build")) {
    $env:HF_HOME = "G:\datasets\models\_hf_home"
    if (-not (Test-Path (Join-Path $Base "transformer\config.json"))) { Fail "no base model at $Base" }
    if (Test-Path (Join-Path $Clips "summary.json")) {
        Write-Host "clips already built: $Clips (delete the folder to rebuild)"
    } else {
        $common = @("tools\build_genfill_clips.py", "--out", $Clips, "--registration", $Reg, "--recipe", "v5", "--seed", "20261122",
                    "--v5-hdris", $Skies, "--v5-scenes", (Join-Path $SceneDir "manifest.jsonl"))
        Say "Plan"
        $rc = Run $Python ($common + "--dry-run") ""
        if ($rc -ne 0) { Fail "plan" }
        $resume = Test-Path (Join-Path $Clips "clips.jsonl")
        $bargs = $common + @("--base", $Base, "--workers", $Workers, "--device", "cuda")
        if ($SmokeTest) { $bargs += @("--limit", "12", "--dev-fraction", "0.2") }
        if ($resume) { $bargs += "--resume"; Write-Host "continuing the interrupted build in $Clips" }
        Say "Build and encode the v5 clips ($Workers workers, VAE on cuda) -> $Clips"
        $rc = Run $Python $bargs "$Clips.build.log"
        if ($rc -ne 0) { Fail "clip build (see $Clips.build.log)" }
    }
    $sum = Get-Content (Join-Path $Clips "summary.json") | ConvertFrom-Json
    Write-Host ("clips {0} from {1} sources, {2} scenes; commercial-clean {3}" -f $sum.clips, $sum.sources_used, $sum.scenes_used, $sum.commercial_clean)
    Write-Host ("by source {0}`nby kind {1}`nby theme {2}`noptics {3}" -f ($sum.by_source | ConvertTo-Json -Compress),
        ($sum.by_kind | ConvertTo-Json -Compress), ($sum.by_theme | ConvertTo-Json -Compress), ($sum.optics | ConvertTo-Json -Compress))
    if (-not $sum.commercial_clean) { Fail "a clip without a commercial-clean licence" }
    if ($sum.recipe -ne "v5") { Fail "$Clips is not a v5 store" }
    Write-Host "`nDone. Tell Claude: it reviews the preview clips and the summary before a v5 training run is registered." -ForegroundColor Green
}
