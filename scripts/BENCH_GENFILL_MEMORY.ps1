<#
.SYNOPSIS
  How much GPU memory AI fill needs in each memory mode (16 Nov 2026, RUDRA 1.2.5). Runs
  tools/bench_genfill_memory.py with AI fill's own Python: the fill as gated, then each mode under a cap
  that stands in for a 12 GB and a 10 GB card, each output compared with the gated one.

    powershell -ExecutionPolicy Bypass -File scripts\BENCH_GENFILL_MEMORY.ps1 -Master D:\shots\09V_master
    powershell -ExecutionPolicy Bypass -File scripts\BENCH_GENFILL_MEMORY.ps1 -Master ... -Frames 58 -Caps "12,10,8"

  -Master: a folder of RUDRA master EXR frames (a shot AI fill was gated on is best). The report is
  reports\genfill_memory_<stamp>\report.md, with every fill kept beside it to look at.
  Close RUDRA first: the benchmark needs the whole GPU. Nothing is changed in RUDRA or its settings.
#>
param(
    [Parameter(Mandatory = $true)][string]$Master,
    [string]$Caps = "12,10",
    [string]$Modes = "swap,tile,on",
    [int]$Frames = 33,
    [string]$Python = "",
    [string]$Base = "",
    [string]$Lora = "",
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path -LiteralPath $Master)) { Fail "no folder $Master" }
if (Get-Process RUDRA -ErrorAction SilentlyContinue) { Fail "RUDRA is running: close it, the benchmark needs the whole GPU" }
if (-not $Python) {
    # The Python AI fill runs with: the one its setup saved (QSettings genfill/python), else the repository's .venv.
    $saved = (Get-ItemProperty "HKCU:\Software\FXTD Studios\RUDRA\genfill" -Name python -ErrorAction SilentlyContinue).python
    if ($saved -and (Test-Path -LiteralPath $saved)) { $Python = $saved }
    elseif (Test-Path ".venv\Scripts\python.exe") { $Python = ".venv\Scripts\python.exe" }
    else { Fail "no Python for AI fill found: give -Python (the one Deliver > AI fill > Setup shows)" }
}
Write-Host "Python: $Python" -ForegroundColor Cyan
$a = @("tools\bench_genfill_memory.py", "--master", $Master, "--caps", $Caps, "--modes", $Modes, "--frames", "$Frames", "--python", $Python)
if ($Base) { $a += @("--base", $Base) }
if ($Lora) { $a += @("--lora", $Lora) }
if ($Extra) { $a += @("--extra", $Extra) }
& $Python @a
if ($LASTEXITCODE -ne 0) { Fail "the benchmark (see its output above)" }
Write-Host "`nSend the report.md above (and report.json beside it)." -ForegroundColor Green
