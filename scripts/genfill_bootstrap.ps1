<#
  AI fill bootstrap for Windows (7 Nov 2026): what "Set up automatically" runs for RUDRA's own Python
  (1.3.2: every time, not only when there is none; each step checks itself and is passed over when it is
  done). Everything goes under -Root (the app's own data folder), and nothing on the system changes (no
  PATH, no registry):
    Root\uv                 uv, the package installer (one fixed release, downloaded once, checked by SHA-256)
    Root\python-<version>   a private Python 3.12 (python-build-standalone, one fixed build, checked by SHA-256)
    Root\venv               the fill's environment, made by that Python's own venv: PyTorch (CUDA 12.8),
                            diffusers, OpenColorIO, OpenEXR, ...
  then rudra.genfill.setup checks the GPU and the disk and downloads Wan2.1-VACE-1.3B into -Base (or, with
  -From / -FlorenceFrom, installs Wan2.1-VACE or Florence-2 from a folder or zip, with no download).

    powershell -NoProfile -ExecutionPolicy Bypass -File genfill_bootstrap.ps1 -Root <dir> -Base <dir> -Repo <dir>
               [-From <folder or zip> | -FlorenceFrom <folder or zip> | -CheckOnly]
  -CheckOnly: everything up to the model, then rudra.genfill.setup --check-only (tmp\SETUP_CHECK.ps1 runs it so).

  1.3.2 (end-user report, 27 Nov 2026): no junction anywhere. uv made the Python before, and uv puts its
  Python behind a junction (python\cpython-3.12-windows-x86_64-none, so that a venv follows patch upgrades);
  a computer where Windows does not let a program through a junction a user made stopped the setup at "could
  not make the Python environment" ("The path cannot be traversed because it contains an untrusted mount
  point. (os error 448)"). The Python is downloaded and unpacked here now, the venv is Python's own, and uv
  only installs packages into it (UV_PYTHON_DOWNLOADS=never: it never fetches a Python of its own).

  -Repo: the folder that holds rudra\ (the app's python\ folder, or the repository).
  stdout: one JSON object per line for the app; the installers' own progress goes to stderr.
  Exit code: rudra.genfill.setup's (0 ready, 1 failed, 2 no GPU). $env:RUDRA_TORCH_INDEX
  overrides where PyTorch comes from (default https://download.pytorch.org/whl/cu128).
#>
param([Parameter(Mandatory)][string]$Root, [Parameter(Mandatory)][string]$Base, [Parameter(Mandatory)][string]$Repo,
      [string]$From = '', [string]$FlorenceFrom = '', [switch]$CheckOnly)
# Continue, not Stop: Windows PowerShell 5.1 turns a native command's stderr line into a terminating
# error under Stop when it is redirected (uv writes its progress there). Every step checks itself.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is 10x slower with its progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
# ASCII only, the rest as \uXXXX (1.3.2): the app reads UTF-8 and this console writes its own code page, which
# turned a user name in another alphabet into question marks in the Python's path.
function Emit([hashtable]$o) {
    $j = [regex]::Replace(($o | ConvertTo-Json -Compress), '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
    [Console]::Out.WriteLine($j); [Console]::Out.Flush()
}
# Removes a folder and what is in it, never what a junction in it points to (rd /s does not follow one).
function Remove-Tree([string]$p) { if (Test-Path -LiteralPath $p) { & cmd.exe /d /c rd /s /q $p 2>$null | Out-Null } }
function Free-Gb([string]$p) {
    try { [math]::Round((New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($p))).AvailableFreeSpace / 1GB, 1) } catch { -1 }
}
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
    Step 'Downloading uv, the package installer (about 20 MB)'
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
$env:UV_CACHE_DIR = Join-Path $Root 'cache'
$env:UV_NO_PROGRESS = '1'
$env:UV_PYTHON_DOWNLOADS = 'never'

# The Python: python-build-standalone (what uv itself downloads), one fixed build checked by its SHA-256 (the
# release's SHA256SUMS) before it is unpacked. A newer Python: change the three lines together.
$PyVersion = '3.12.13'
$PyBuild = '20260718'
$PySha256 = '0d422a1439ec308e03f47df551bc30f5994727c456e414b026d202bcda9b7c1c'   # ...-x86_64-pc-windows-msvc-install_only_stripped.tar.gz
$pyHome = Join-Path $Root "python-$PyVersion"
$basePy = Join-Path $pyHome 'python\python.exe'
$venv = Join-Path $Root 'venv'
$py = Join-Path $venv 'Scripts\python.exe'

# A venv that does not start (one behind a junction Windows refuses, one half made) is made again.
$venvWorks = $false
if (Test-Path -LiteralPath $py) {
    & $py -c 'import sys' 2>$null
    $venvWorks = ($LASTEXITCODE -eq 0)
}
if (-not $venvWorks) {
    if (-not (Test-Path -LiteralPath $basePy)) {
        Step "Downloading Python $PyVersion for the AI fill (about 22 MB)"
        $tmp = Join-Path $Root 'python-download'
        Remove-Tree $tmp
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null
        $archive = Join-Path $tmp 'python.tar.gz'
        $url = "https://github.com/astral-sh/python-build-standalone/releases/download/$PyBuild/cpython-$PyVersion%2B$PyBuild-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
        try {
            Invoke-WebRequest -ErrorAction Stop -UseBasicParsing -Uri $url -OutFile $archive
        } catch { Fail "could not download Python (network?): $($_.Exception.Message)" }
        $got = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
        if ($got -ne $PySha256) {
            Remove-Tree $tmp
            Fail "the Python download is not the expected file (SHA-256 $got, expected $PySha256): nothing was installed"
        }
        $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
        & $tar -xzf $archive -C $tmp 2>&1 | ForEach-Object { [Console]::Error.WriteLine("$_") }
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $tmp 'python\python.exe'))) {
            Fail 'could not unpack the Python download (see the log)'
        }
        Remove-Tree $pyHome
        New-Item -ItemType Directory -Force -Path $pyHome | Out-Null
        Move-Item -LiteralPath (Join-Path $tmp 'python') -Destination $pyHome -ErrorAction Stop
        Remove-Tree $tmp
    }
    Step 'Making the Python environment for the AI fill'
    Remove-Tree $venv
    & $basePy -m venv $venv 2>&1 | ForEach-Object { [Console]::Error.WriteLine("$_") }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $py)) { Fail 'could not make the Python environment (see the log)' }
    & $py -c 'import sys' 2>&1 | ForEach-Object { [Console]::Error.WriteLine("$_") }
    if ($LASTEXITCODE -ne 0) { Fail 'the new Python environment does not start (see the log)' }
    # uv's Python of the setups before 1.3.2, junction and all: nothing uses it now.
    Remove-Tree (Join-Path $Root 'python')
}
Emit @{ event = 'bootstrap'; python = $py }

$index = if ($env:RUDRA_TORCH_INDEX) { $env:RUDRA_TORCH_INDEX } else { 'https://download.pytorch.org/whl/cu128' }
& $py -c 'import torch' 2>$null
if ($LASTEXITCODE -ne 0) {
    $free = Free-Gb $Root
    if ($free -ge 0 -and $free -lt 10) {
        Fail "$free GB free on $([System.IO.Path]::GetPathRoot($Root)), where RUDRA's Python for the AI fill goes: PyTorch and the fill's packages need about 10 GB. Free some space, then press Set up automatically again"
    }
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
# 1.2.6: Florence-2 for Suggest a prompt. 1.3.2: or a model from a folder or zip, with no download.
# @( ) keeps it an array: an if that yields one item gives a plain string, and @extra then passed it one
# character at a time (28 Nov: "unrecognized arguments: - - c h e c k - o n l y"; --florence alike).
$extra = @(if ($CheckOnly) { '--check-only' } elseif ($From) { '--from', $From } elseif ($FlorenceFrom) { '--florence-from', $FlorenceFrom } else { '--florence' })
& $py -m rudra.genfill.setup --base $Base --json --skip-packages @extra
exit $LASTEXITCODE
