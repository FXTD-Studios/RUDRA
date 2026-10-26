<#
.SYNOPSIS
  Rebuild the RUDRA app with Deliver > "Learned ACES 2.0 master…", run its tests,
  point it at the ACES v2 engine and model package, and start it.

    powershell -ExecutionPolicy Bypass -File scripts\RUN_APP_ACES.ps1
    powershell -ExecutionPolicy Bypass -File scripts\RUN_APP_ACES.ps1 -SkipBuild      # just start it

  The app runs the masters through the ACES v2 rudra-native (build\native_aces_v2,
  made by NATIVE_ACES_V2.ps1) as a child process. This script writes the app's
  settings (HKCU\Software\FXTD Studios\RUDRA\aces) so the dialog opens ready:
    aces/cli           the ACES v2 rudra-native.exe
    aces/roots         dist\models (where the ACES v2 package is)
    aces/package       the newest dist\models\aces_v2_* package
    aces/path_prepend  LibTorch's lib\ from .venv (its DLLs are not beside that exe)
  Nothing is signed or released here.
#>
param(
    [string]$Build = "build\native_gate_b",
    [string]$AcesBuild = "build\native_aces_v2",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Package = "",
    [switch]$SkipBuild,
    [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent   # the repo this script sits in
Set-Location -LiteralPath $Repo
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

$Cache = Join-Path $Build "CMakeCache.txt"
if (-not (Test-Path $Cache)) { Fail "no app build at $Build (run NATIVE_GATE_B.ps1 once)" }
$cmake = ((Select-String -Path $Cache -Pattern '^CMAKE_COMMAND:INTERNAL=(.+)$').Matches[0].Groups[1].Value)

if (-not $SkipBuild) {
    # Windows cannot replace an exe that is running: ask, never close someone's work.
    $exePath = [IO.Path]::GetFullPath((Join-Path $Repo (Join-Path $Build "app\Release\RUDRA.exe")))
    $open = Get-Process RUDRA -ErrorAction SilentlyContinue | Where-Object { $_.Path -and ([IO.Path]::GetFullPath($_.Path) -eq $exePath) }
    if ($open) { Fail "RUDRA is open from this build (pid $($open.Id -join ', ')): close it, then run this again" }
    Say "Build the app (Release): RUDRA, rudra_app_tests"
    & $cmake --build $Build --config Release --parallel --target RUDRA rudra_app_tests
    if ($LASTEXITCODE -ne 0) { Fail "app build" }
    # The engine too: the app passes --luminance-guard / --no-luminance-guard (26 Oct).
    $AcesCache = Join-Path $AcesBuild "CMakeCache.txt"
    if (Test-Path $AcesCache) {
        Say "Build the ACES v2 engine (Release): rudra-native"
        $acesCmake = ((Select-String -Path $AcesCache -Pattern '^CMAKE_COMMAND:INTERNAL=(.+)$').Matches[0].Groups[1].Value)
        & $acesCmake --build $AcesBuild --config Release --parallel --target rudra-native
        if ($LASTEXITCODE -ne 0) { Fail "ACES v2 engine build" }
    }
}
$App = Join-Path $Build "app\Release\RUDRA.exe"
$AppTests = Join-Path $Build "app\Release\rudra_app_tests.exe"
if (-not (Test-Path $App)) { Fail "no $App" }

if (-not $SkipTests -and (Test-Path $AppTests)) {
    Say "App tests: the ACES master dialog and the menus"
    # The tests need Qt6Test.dll, which is not deployed beside them: put Qt's bin\ first on PATH.
    $QtDir = ((Select-String -Path $Cache -Pattern '^Qt6_DIR:PATH=(.+)$').Matches[0].Groups[1].Value)
    $QtBin = Join-Path $QtDir "..\..\..\bin" | Resolve-Path -ErrorAction SilentlyContinue
    $savedPath = $env:PATH
    if ($QtBin) { $env:PATH = "$($QtBin.Path);$env:PATH" }
    $env:QT_QPA_PLATFORM = "offscreen"
    & $AppTests "--gtest_filter=AcesMaster.*:AppActions.MenubarIsTheSpec:AppActions.PageMenusLabelsHintsChecks:AppActions.EveryPageKeyReachesItsAction"
    $rc = $LASTEXITCODE
    Remove-Item Env:\QT_QPA_PLATFORM
    $env:PATH = $savedPath
    if ($rc -ne 0) { Fail ("app tests (exit {0}; -1073741515 = a DLL was not found)" -f $rc) }
}

Say "Point the app at the ACES v2 engine and package"
$Cli = (Resolve-Path (Join-Path $AcesBuild "cli\Release\rudra-native.exe") -ErrorAction SilentlyContinue)
if (-not $Cli) { Fail "no ACES v2 rudra-native in $AcesBuild (run NATIVE_ACES_V2.ps1 first)" }
if (-not $Package) {
    $p = Get-ChildItem (Join-Path $Repo "dist\models") -Directory -Filter "aces_v2_*" -ErrorAction SilentlyContinue |
         Where-Object { Test-Path (Join-Path $_.FullName "manifest.json") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $p) { Fail "no dist\models\aces_v2_* package (run NATIVE_ACES_V2.ps1 first)" }
    $Package = $p.FullName
}
$TorchLib = & $Python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $TorchLib)) { Fail "LibTorch's lib folder not found through $Python" }

$Key = "HKCU:\Software\FXTD Studios\RUDRA\aces"
New-Item -Path $Key -Force | Out-Null
Set-ItemProperty -Path $Key -Name "cli" -Value $Cli.Path
Set-ItemProperty -Path $Key -Name "package" -Value $Package
New-ItemProperty -Path $Key -Name "roots" -PropertyType MultiString -Value @((Split-Path $Package)) -Force | Out-Null
New-ItemProperty -Path $Key -Name "path_prepend" -PropertyType MultiString -Value @($TorchLib) -Force | Out-Null
# The same check the app makes: the engine, with LibTorch on PATH, prints its aces-v2 usage.
$savedPath = $env:PATH
$env:PATH = "$TorchLib;$env:PATH"
$ErrorActionPreference = "Continue"   # its usage goes to stderr; Windows PowerShell would stop on that
$probe = (& $Cli.Path aces-v2 --help 2>&1 | Out-String)
$probeRc = $LASTEXITCODE
$ErrorActionPreference = "Stop"
$env:PATH = $savedPath
if ($probe -notmatch "usage: rudra-native aces-v2 <package>") {
    Write-Host $probe
    Fail ("the engine did not answer as an ACES v2 rudra-native (exit {0}; -1073741515 = a DLL was not found)" -f $probeRc)
}
if ($probe -notmatch "luminance-guard") { Fail "the engine predates --luminance-guard: run without -SkipBuild to rebuild it" }
$env:PATH = "$TorchLib;$env:PATH"
$ErrorActionPreference = "Continue"
$serve = (& $Cli.Path aces-v2-serve --help 2>&1 | Out-String)
$ErrorActionPreference = "Stop"
$env:PATH = $savedPath
if ($serve -notmatch "usage: rudra-native aces-v2-serve") { Fail "the engine predates aces-v2-serve (the preview and preload): run without -SkipBuild to rebuild it" }
Write-Host "engine   $($Cli.Path)"
Write-Host "package  $Package"
Write-Host "LibTorch $TorchLib"

New-ItemProperty -Path $Key -Name "preload" -PropertyType String -Value "true" -Force | Out-Null
Say "Start RUDRA: the ACES model loads in the background (status bar); Deliver > Learned ACES 2.0 master… or Ctrl+Shift+E"
Start-Process -FilePath (Resolve-Path $App).Path -WorkingDirectory (Split-Path (Resolve-Path $App).Path)
