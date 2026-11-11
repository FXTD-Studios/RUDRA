<#
  AI fill bootstrap for Windows (7 Nov 2026): what "Set up automatically" runs when this computer
  has no Python for the fill. Everything goes under -Root (the app's own data folder), and nothing
  on the system changes (no PATH, no registry):
    Root\uv      uv, the Python installer (one fixed release, downloaded once and checked by SHA-256)
    Root\python  a private Python 3.12 (uv's managed CPython)
    Root\venv    the fill's environment: PyTorch (CUDA 12.8), diffusers, OpenColorIO, OpenEXR, ...
  then rudra.genfill.setup checks the GPU and the disk and downloads Wan2.1-VACE-1.3B into -Base.

    powershell -NoProfile -ExecutionPolicy Bypass -File genfill_bootstrap.ps1 -Root <dir> -Base <dir> -Repo <dir>

  -Repo: the folder that holds rudra\ (the app's python\ folder, or the repository).
  stdout: one JSON object per line for the app; the installers' own progress goes to stderr.
  Exit code: rudra.genfill.setup's (0 ready, 1 failed, 2 no GPU). $env:RUDRA_TORCH_INDEX
  overrides where PyTorch comes from (default https://download.pytorch.org/whl/cu128).
#>
param([Parameter(Mandatory)][string]$Root, [Parameter(Mandatory)][string]$Base, [Parameter(Mandatory)][string]$Repo)
# Continue, not Stop: Windows PowerShell 5.1 turns a native command's stderr line into a terminating
# error under Stop when it is redirected (uv writes its progress there). Every step checks itself.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is 10x slower with its progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
function Emit([hashtable]$o) { [Console]::Out.WriteLine(($o | ConvertTo-Json -Compress)); [Console]::Out.Flush() }
function Step([string]$t) { Emit @{ event = 'step'; step = 'bootstrap'; text = $t } }
function Fail([string]$t) { Emit @{ event = 'done'; ok = $false; error = $t }; exit 1 }

New-Item -ItemType Directory -Force -Path $Root | Out-Null
if (-not (Test-Path (Join-Path $Repo 'rudra\genfill\setup.py'))) { Fail "no rudra\genfill in $Repo" }

# uv is a program this script downloads and runs, so it is one fixed release, checked by its SHA-256
# before it is unpacked (security review, 11 Nov 2026; it was releases/latest, unchecked). The hash is
# the release asset's digest as GitHub publishes it. A newer uv: change both lines together.
$UvVersion = '0.13.7'
$UvSha256 = '0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021'   # uv-x86_64-pc-windows-msvc.zip
$uvDir = Join-Path $Root 'uv'
$uv = Join-Path $uvDir 'uv.exe'
if (-not (Test-Path $uv)) {
    Step 'Downloading uv, the Python installer (about 20 MB)'
    $tmp = Join-Path $Root 'uv-download'
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Force -Path $tmp, $uvDir | Out-Null
    $zip = Join-Path $tmp 'uv.zip'
    try {
        Invoke-WebRequest -ErrorAction Stop -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
    } catch { Fail "could not download uv (network?): $($_.Exception.Message)" }
    $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $UvSha256) {
        Remove-Item -Recurse -Force $tmp
        Fail "the uv download is not the expected file (SHA-256 $got, expected $UvSha256): nothing was installed"
    }
    Expand-Archive -Path $zip -DestinationPath $tmp -Force -ErrorAction Stop
    $found = Get-ChildItem -Path $tmp -Recurse -Filter 'uv.exe' | Select-Object -First 1
    if (-not $found) { Fail 'the uv download has no uv.exe in it' }
    Move-Item $found.FullName $uv -Force -ErrorAction Stop
    Remove-Item -Recurse -Force $tmp
}
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'python'
$env:UV_CACHE_DIR = Join-Path $Root 'cache'
$env:UV_NO_PROGRESS = '1'

$py = Join-Path $Root 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Step 'Installing a private Python 3.12 for the AI fill'
    & $uv venv --seed --python 3.12 (Join-Path $Root 'venv') 2>&1 | ForEach-Object { [Console]::Error.WriteLine("$_") }
    if ($LASTEXITCODE -ne 0) { Fail 'could not make the Python environment' }
}
Emit @{ event = 'bootstrap'; python = $py }

$index = if ($env:RUDRA_TORCH_INDEX) { $env:RUDRA_TORCH_INDEX } else { 'https://download.pytorch.org/whl/cu128' }
& $py -c 'import torch' 2>$null
if ($LASTEXITCODE -ne 0) {
    Step 'Installing PyTorch with CUDA (about 2.5 GB)'
    & $uv pip install --python $py torch --index-url $index 2>&1 | ForEach-Object { [Console]::Error.WriteLine("$_") }
    if ($LASTEXITCODE -ne 0) { Fail 'PyTorch did not install (see the log)' }
}
Step "Installing the AI fill's Python packages"
$pkgs = @('numpy>=1.26,<3', 'opencv-python-headless>=4.9', 'OpenEXR>=3.2', 'opencolorio>=2.5', 'safetensors>=0.4',
          'diffusers>=0.35', 'transformers>=4.45', 'peft>=0.13', 'accelerate>=1.0', 'sentencepiece', 'ftfy',
          'huggingface-hub>=0.25', 'PyYAML>=6', 'tqdm>=4.66')
# Fixed versions for every package (genfill_constraints.txt beside this script): the tested set, not the newest.
$cons = Join-Path $PSScriptRoot 'genfill_constraints.txt'
if (Test-Path $cons) { $pkgs += @('-c', $cons) } else { [Console]::Error.WriteLine("no genfill_constraints.txt beside the bootstrap: installing unpinned versions") }
& $uv pip install --python $py @pkgs 2>&1 | ForEach-Object { [Console]::Error.WriteLine("$_") }
if ($LASTEXITCODE -ne 0) { Fail "the AI fill's packages did not install (see the log)" }

$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Repo;$env:PYTHONPATH" } else { $Repo }
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location -LiteralPath $Repo
& $py -m rudra.genfill.setup --base $Base --json --skip-packages
exit $LASTEXITCODE
