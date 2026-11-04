<#
.SYNOPSIS
  4K and long-clip test of the learned ACES route (4 Nov 2026): can RUDRA render a long 4K
  clip locally on a 16 GB card, without a frame cap (Beeble SwitchHDR 2.0: 480 frames, cloud)?

  Renders -Frames frames of a clip through rudra-native aces-v2 (EXR half masters), samples
  GPU memory every second (nvidia-smi), and reports: seconds per frame (first, middle and last
  tenth, so a slow-down or leak shows), total time, peak VRAM, peak RAM of the engine, frames
  that failed the appearance check, and the run's exit code.

      powershell -ExecutionPolicy Bypass -File scripts\RUN_4K_TEST.ps1 -Clip D:\clips\shot_4k.mov -Frames 1440
      powershell -ExecutionPolicy Bypass -File scripts\RUN_4K_TEST.ps1 -MakeFrom D:\flicker_clips\09V.mp4 -Frames 1440

  -MakeFrom upscales an HD clip to 3840x2160 and loops it to -Frames: a SPEED and STABILITY
  test only (the picture is an upscale); the report says so. 1440 frames = 1 minute at 24 fps.
  The engine processes each frame whole (no tiles), so there are no tile seams to check; the
  clipped-area smoothing radius is fixed in pixels (sigma 64), half as wide relative to a 4K
  picture as to an HD one, which the report notes.
#>
[CmdletBinding()]
param(
    [string]$Clip = "",
    [string]$MakeFrom = "",
    [int]$Frames = 1440,
    [string]$Out = "G:\rudra_training\test_4k_$(Get-Date -Format yyyyMMdd_HHmm)",
    [string]$Package = "",
    [string]$Cli = "build\native_aces_v2\cli\Release\rudra-native.exe",
    [string]$Device = "cuda",
    [string]$Python = ".venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }

# The engine's DLLs, as RUN_FLICKER_CLIPS.ps1 sets them.
if (-not (Test-Path $Cli)) { Fail "no rudra-native at $Cli (build it with NATIVE_ACES_V2.ps1, or pass -Cli)" }
if ((Test-Path $Python) -and ($Cli -like "*.exe")) {
    $TorchLib = & $Python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>$null
    if ($TorchLib -and (Test-Path $TorchLib)) { $env:PATH = "$TorchLib;$env:PATH" }
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $ffd = Get-ChildItem "tmp\native_deps" -Directory -Filter "ffmpeg*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ffd) { $env:PATH = "$(Join-Path $ffd.FullName 'bin');$env:PATH" }
}
if (-not $Package) {
    $Package = @("$env:LOCALAPPDATA\FXTD Studios\RUDRA\models\RUDRA_ACES_2", "$env:LOCALAPPDATA\RUDRA\models\RUDRA_ACES_2", "dist\models\RUDRA_ACES_2") |
        Where-Object { Test-Path "$_\manifest.json" } | Select-Object -First 1
}
if (-not $Package -or -not (Test-Path "$Package\manifest.json")) { Fail "no RUDRA_ACES_2 model package found (pass -Package)" }
New-Item -ItemType Directory -Force $Out | Out-Null

$synthetic = $false
if ($MakeFrom) {
    if (-not (Test-Path $MakeFrom)) { Fail "no clip at $MakeFrom" }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Fail "ffmpeg is needed for -MakeFrom" }
    $Clip = Join-Path $Out "synthetic_4k.mov"
    Say "Making a 3840x2160 test clip of $Frames frames from $MakeFrom (speed and stability only)"
    & ffmpeg -v error -y -stream_loop -1 -i $MakeFrom -frames:v $Frames -vf "scale=3840:2160:flags=lanczos,format=yuv422p10le" -c:v prores_ks -profile:v 3 -an $Clip
    if ($LASTEXITCODE -ne 0) { Fail "ffmpeg could not make the test clip" }
    $synthetic = $true
}
if (-not $Clip -or -not (Test-Path $Clip)) { Fail "give -Clip <a 4K clip> or -MakeFrom <an HD clip>" }

# GPU memory every second, in the background.
$vramLog = Join-Path $Out "vram.csv"
$haveSmi = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$job = $null
if ($haveSmi) {
    $job = Start-Job -ArgumentList $vramLog -ScriptBlock {
        param($log)
        while ($true) { (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) | Add-Content $log; Start-Sleep -Seconds 1 }
    }
}

Say "Rendering $Frames frames of $(Split-Path $Clip -Leaf) on $Device"
$renderDir = Join-Path $Out "masters"
$logFile = Join-Path $Out "engine.log"
$sw = [Diagnostics.Stopwatch]::StartNew()
$ErrorActionPreference = "Continue"
$proc = Start-Process -FilePath $Cli -ArgumentList @("aces-v2", "`"$Package`"", "`"$Clip`"", "--out", "`"$renderDir`"", "--device", $Device,
    "--exr", "half", "--range", "0:$($Frames - 1)") -NoNewWindow -PassThru -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
$peakRamMB = 0
while (-not $proc.HasExited) {
    try { $proc.Refresh(); $peakRamMB = [Math]::Max($peakRamMB, [int]($proc.PeakWorkingSet64 / 1MB)) } catch {}
    Start-Sleep -Seconds 2
}
$proc.WaitForExit()
$exit = $proc.ExitCode
$sw.Stop()
$ErrorActionPreference = "Stop"
if ($job) { Stop-Job $job; Remove-Job $job }

# Per-frame times and checks from the engine's own lines:
#   <source> frame N -> <name>.exr  1234 ms  remaining_display_errors=0 ...
$ms = @(); $failed = 0
foreach ($line in (Get-Content $logFile -ErrorAction SilentlyContinue)) {
    if ($line -match '\s(\d+(?:\.\d+)?) ms\s+remaining_display_errors=(\d+)') {
        $ms += [double]$Matches[1]
        if ([int]$Matches[2] -gt 0) { $failed++ }
    }
}
$n = $ms.Count
function MeanOf($a) { if ($a.Count -eq 0) { return 0 } ; return ($a | Measure-Object -Average).Average }
$tenth = [Math]::Max(1, [int]($n / 10))
$first = if ($n) { MeanOf ($ms[0..([Math]::Min($n, $tenth) - 1)]) } else { 0 }
$mid = if ($n) { MeanOf ($ms[([int]($n / 2) - [int]($tenth / 2))..([int]($n / 2) + [int]($tenth / 2) - 1)]) } else { 0 }
$last = if ($n) { MeanOf ($ms[($n - $tenth)..($n - 1)]) } else { 0 }
$peakVram = 0; $totalVram = 0
if (Test-Path $vramLog) {
    foreach ($l in Get-Content $vramLog) { $p = $l -split ','; if ($p.Count -ge 2) { $peakVram = [Math]::Max($peakVram, [int]$p[0]); $totalVram = [int]$p[1] } }
}
$res = ""
try { $res = (& ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 $Clip) } catch {}

$report = [ordered]@{
    clip = $Clip; resolution = $res; synthetic_upscale = $synthetic; frames_asked = $Frames; frames_rendered = $n
    exit_code = $exit; total_minutes = [Math]::Round($sw.Elapsed.TotalMinutes, 1)
    seconds_per_frame = [ordered]@{ first_tenth = [Math]::Round($first / 1000, 2); middle = [Math]::Round($mid / 1000, 2); last_tenth = [Math]::Round($last / 1000, 2) }
    slowdown_last_vs_first = if ($first -gt 0) { [Math]::Round($last / $first, 2) } else { $null }
    peak_vram_mb = $peakVram; vram_total_mb = $totalVram; peak_engine_ram_mb = $peakRamMB
    frames_failing_appearance = $failed
    note = "Whole-frame processing (no tiles, no seams). The clipped-area smoothing radius is fixed in pixels (sigma 64): at 4K it is half as wide relative to the picture as at HD." +
           $(if ($synthetic) { " SYNTHETIC: an upscaled, looped HD clip; speed and stability only, not picture quality." } else { "" })
}
[IO.File]::WriteAllText((Join-Path $Out "report_4k.json"), ($report | ConvertTo-Json -Depth 4), (New-Object Text.UTF8Encoding $false))

Say "Result"
"{0} frames of {1} at {2}{3}" -f $n, $Frames, $res, $(if ($synthetic) { "  (synthetic upscale: speed and stability only)" } else { "" })
"exit code {0}; {1} min in all" -f $exit, $report.total_minutes
"seconds per frame: first tenth {0}, middle {1}, last tenth {2}  (last/first {3})" -f $report.seconds_per_frame.first_tenth, $report.seconds_per_frame.middle, $report.seconds_per_frame.last_tenth, $report.slowdown_last_vs_first
"peak VRAM {0} of {1} MB; peak engine RAM {2} MB" -f $peakVram, $totalVram, $peakRamMB
"frames failing the appearance check: {0}" -f $failed
$ok = ($exit -eq 0 -or $exit -eq 3) -and $n -eq $Frames -and ($report.slowdown_last_vs_first -eq $null -or $report.slowdown_last_vs_first -lt 1.3)
if ($ok) {
    Write-Host "`nPASS: all $Frames frames rendered with no slow-down; the claim 'local, no frame cap' holds for this clip." -ForegroundColor Green
    if ($failed -gt 0) { Write-Host "  but $failed frame(s) failed the appearance check: a reconstruction problem to look at (engine.log)" -ForegroundColor Yellow }
}
else { Write-Host "`nNOT A PASS: see the numbers above and $logFile" -ForegroundColor Yellow }
"report: $(Join-Path $Out 'report_4k.json')"
