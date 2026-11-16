<#
.SYNOPSIS
  Local check of the colour-managed viewer (17 Nov 2026, docs/view.spec.md revision 7):
  the picture through an OpenColorIO display and view as Nuke shows it, ACEScg or ACES2065-1,
  the original through an input transform, and the range checks.
  Turns OpenColorIO 2.5.2 on in the app build (build\native_gate_b) when it is off, using the
  OpenColorIO the ACES engine or the package was built with, puts its DLL beside the programs,
  then builds and runs the tests that cover the viewer. -Run then starts the app.

    powershell -ExecutionPolicy Bypass -File scripts\TEST_OCIO_VIEWER.ps1
    powershell -ExecutionPolicy Bypass -File scripts\TEST_OCIO_VIEWER.ps1 -Run

  Nothing is committed, pushed or released here.
#>
param(
    [string]$Build = "build\native_gate_b",
    [string]$Ocio = "",          # an OpenColorIO 2.5.2 install (lib\cmake\OpenColorIO, bin\); found when empty
    [switch]$Run,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

$Cache = Join-Path $Build "CMakeCache.txt"
if (-not (Test-Path $Cache)) { Fail "no app build at $Build (run scripts\NATIVE_GATE_B.ps1 once)" }
$cmake = ((Select-String -Path $Cache -Pattern '^CMAKE_COMMAND:INTERNAL=(.+)$').Matches[0].Groups[1].Value)
if (-not $cmake) { $cmake = "cmake" }

# The OpenColorIO install: given, this build's, the ACES engine's, or the package's.
function Ocio-Root([string]$configDir) {
    if (-not $configDir) { return $null }
    $d = $configDir -replace "/", "\"
    $root = Split-Path (Split-Path (Split-Path $d))   # <root>\lib\cmake\OpenColorIO
    if (Test-Path (Join-Path $root "lib\cmake\OpenColorIO\OpenColorIOConfig.cmake")) { return $root }
    return $null
}
function Cache-Ocio([string]$cache) {
    if (-not (Test-Path $cache)) { return $null }
    $m = Select-String -Path $cache -Pattern '^OpenColorIO_DIR:PATH=(.+)$'
    if ($m) { return Ocio-Root $m.Matches[0].Groups[1].Value }
    return $null
}
$OcioRoot = $null
if ($Ocio) { $OcioRoot = $Ocio }
if (-not $OcioRoot) { $OcioRoot = Cache-Ocio $Cache }
if (-not $OcioRoot) { $OcioRoot = Cache-Ocio "build\native_aces_v2\CMakeCache.txt" }
if (-not $OcioRoot -and (Test-Path "tmp\native_deps\ocio-2.5.2\lib\cmake\OpenColorIO\OpenColorIOConfig.cmake")) {
    $OcioRoot = (Resolve-Path "tmp\native_deps\ocio-2.5.2").Path
}
if (-not $OcioRoot) {
    Fail "no OpenColorIO 2.5.2 found. Build it once with scripts\NATIVE_ACES_V2.ps1 or scripts\PACKAGE_WINDOWS.ps1, or pass -Ocio <install folder>"
}
Write-Host "  OpenColorIO: $OcioRoot"

if (-not (Select-String -Path $Cache -Pattern '^RUDRA_WITH_OCIO:BOOL=ON' -Quiet)) {
    Say "Turn OpenColorIO on in $Build (the release package has it on)"
    & $cmake -S native -B $Build -DRUDRA_WITH_OCIO=ON "-DOpenColorIO_DIR=$OcioRoot\lib\cmake\OpenColorIO"
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure with OpenColorIO" }
}

# The DLL beside the programs that load it (the package's windeployqt step does the same for Qt).
$dll = Get-ChildItem (Join-Path $OcioRoot "bin") -Filter "OpenColorIO*.dll" -ErrorAction SilentlyContinue
if (-not $dll) { Fail "no OpenColorIO DLL in $OcioRoot\bin" }
foreach ($dir in @("app\Release", "tests\Release")) {
    $to = Join-Path $Build $dir
    New-Item -ItemType Directory -Force -Path $to | Out-Null
    foreach ($d in $dll) { Copy-Item $d.FullName $to -Force }
}

Say "Build and run the viewer's tests (NATIVE_TESTS.ps1)"
$t = @("-ExecutionPolicy", "Bypass", "-File", "scripts\NATIVE_TESTS.ps1", "-Build", $Build,
       "-Filter", "ViewOcio.*:OcioViewer.*:ViewRev6.*:Viewer*:AnchorView*:AppColourView.*:AppViewerToolbar.*:AppActions.*:Actions.*")
if ($SkipBuild) { $t += "-SkipBuild" }
& powershell @t
if ($LASTEXITCODE -ne 0) { Fail "tests (see the report in reports\)" }

if ($Run) {
    Say "Start RUDRA (RUN_APP_ACES.ps1)"
    Write-Host "  On the viewer toolbar: the view box (RUDRA view, SDR sRGB, Rec.709, HDR PQ 1000, P3-D65, Raw, then"
    Write-Host "  every display and view of the config), ACEScg / ACES2065-1, Original, Gamut, Invalid. The label at the"
    Write-Host "  end says what the picture goes through. Hover for the probe: the working-space values and the view's"
    Write-Host "  10-bit codes. View > Colour management... picks the built-in config, `$OCIO or a config.ocio."
    Write-Host "  Compare in Nuke: Read the master EXR as ACEScg (or ACES2065-1), viewer process the same display/view."
    & powershell -ExecutionPolicy Bypass -File scripts\RUN_APP_ACES.ps1 -SkipTests
    if ($LASTEXITCODE -ne 0) { Fail "RUN_APP_ACES.ps1" }
}
Write-Host "`nDone." -ForegroundColor Green
