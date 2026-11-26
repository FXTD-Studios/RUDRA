<#
.SYNOPSIS
  Start RUDRA on Windows.

    powershell -ExecutionPolicy Bypass -File scripts\RUN_RUDRA.ps1              # the installed RUDRA; installs the latest release first if there is none
    powershell -ExecutionPolicy Bypass -File scripts\RUN_RUDRA.ps1 -Update      # install the latest GitHub release (over the old one), then start it
    powershell -ExecutionPolicy Bypass -File scripts\RUN_RUDRA.ps1 -Dev         # the app built from this repo (build\native_gate_b), not the release

  The release is the installer from github.com/FXTD-Studios/RUDRA/releases/latest, checked against its
  .sha256 before it runs. It installs for this user (no administrator prompt).
#>
param([switch]$Update, [switch]$Dev)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
function Say([string]$m) { Write-Host "== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }

if ($Dev) {
    $exe = Join-Path $Repo "build\native_gate_b\app\Release\RUDRA.exe"
    if (-not (Test-Path $exe)) { Fail "no repo build at $exe (scripts\RUN_APP_ACES.ps1 builds it)" }
    Say "Starting the repo build: $exe"
    Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe)
    exit 0
}

$candidates = @("$env:LOCALAPPDATA\Programs\RUDRA\RUDRA.exe", "$env:ProgramFiles\RUDRA\RUDRA.exe")
$exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($Update -or -not $exe) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Say "Reading the latest release"
    $rel = Invoke-RestMethod "https://api.github.com/repos/FXTD-Studios/RUDRA/releases/latest" -Headers @{ "User-Agent" = "RUDRA-run" }
    $setup = $rel.assets | Where-Object { $_.name -like "RUDRA-*-windows-x64-setup.exe" } | Select-Object -First 1
    $sum = $rel.assets | Where-Object { $_.name -eq "$($setup.name).sha256" } | Select-Object -First 1
    if (-not $setup -or -not $sum) { Fail "no Windows installer (with its .sha256) in $($rel.tag_name)" }
    $dir = Join-Path $env:TEMP "rudra_setup"; New-Item -ItemType Directory -Force $dir | Out-Null
    $file = Join-Path $dir $setup.name
    Say "Downloading $($setup.name) ($([math]::Round($setup.size / 1MB)) MB)"
    Invoke-WebRequest $setup.browser_download_url -OutFile $file -UseBasicParsing
    $want = ((Invoke-WebRequest $sum.browser_download_url -UseBasicParsing).Content.Trim() -split "\s+")[0].ToLower()
    $got = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $want) { Remove-Item $file; Fail "checksum mismatch ($got, expected $want)" }
    Say "Checksum OK. Installing $($rel.tag_name) for this user"
    Get-Process RUDRA -ErrorAction SilentlyContinue | ForEach-Object { Fail "RUDRA is running: close it first" }
    $p = Start-Process -FilePath $file -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru
    if ($p.ExitCode -ne 0) { Fail "the installer exited with $($p.ExitCode)" }
    $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exe) { Fail "installed, but RUDRA.exe was not found in $($candidates -join ' or ')" }
}

Say "Starting $exe"
Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe)
