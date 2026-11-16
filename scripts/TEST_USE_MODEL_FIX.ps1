<#
.SYNOPSIS
  Local check of 1.2.5 (16 Nov 2026) before the release:
  - the 1.2.4 feedback fix: "Use the model" on with no reconstruction model package anywhere held
    every Open back, while the learned ACES 2.0 model sat ready;
  - Help > Check for updates automatically (the signing policy's privacy rule).
  Builds rudra_tests and rudra_app_tests in build\native_gate_b and runs the tests that cover them.
  -Repro then sets the tester's state (model/use = true) and starts the app (RUN_APP_ACES.ps1).
  -Package then builds the public package as release.yml does, unsigned: PACKAGE_WINDOWS.ps1
  -PublicRelease -NoArchive, then PACKAGE_WINDOWS_ARCHIVE.ps1 (the ZIP, and the installer with its
  new Privacy page and update-check choice).

    powershell -ExecutionPolicy Bypass -File scripts\TEST_USE_MODEL_FIX.ps1
    powershell -ExecutionPolicy Bypass -File scripts\TEST_USE_MODEL_FIX.ps1 -Repro
    powershell -ExecutionPolicy Bypass -File scripts\TEST_USE_MODEL_FIX.ps1 -Package

  Nothing is committed, pushed or released here.
#>
param(
    [string]$Build = "build\native_gate_b",
    [switch]$Repro,
    [switch]$Package,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

$Cache = Join-Path $Build "CMakeCache.txt"
if (-not (Test-Path $Cache)) { Fail "no app build at $Build (run scripts\NATIVE_GATE_B.ps1 once)" }
if (-not (Select-String -Path $Cache -Pattern '^RUDRA_WITH_OPENCV:BOOL=ON' -Quiet)) {
    Write-Host "  $Build has no still decode (OpenCV): the new test is compiled out. Run scripts\RUN_APP_ACES.ps1 once to turn it on." -ForegroundColor Yellow
}

Say "Build rudra_app_tests and run the model tests (NATIVE_TESTS.ps1)"
$t = @("-ExecutionPolicy", "Bypass", "-File", "scripts\NATIVE_TESTS.ps1", "-Build", $Build,
       "-Filter", "AppModels.*:AppAnalytic.*:AppProject.*:AppUpdates.*:AppActions.*:AppOpen.*:AppCheck.*:Actions.*")
if ($SkipBuild) { $t += "-SkipBuild" }
& powershell @t
if ($LASTEXITCODE -ne 0) { Fail "tests (see the report in reports\)" }

if ($Repro) {
    $key = "HKCU:\Software\FXTD Studios\RUDRA\model"
    $old = (Get-ItemProperty -Path $key -Name use -ErrorAction SilentlyContinue).use
    Say "The tester's state: model/use = true (it was: $old)"
    if (-not (Test-Path $key)) { New-Item -Path $key | Out-Null }
    Set-ItemProperty -Path $key -Name use -Value "true"
    Write-Host "  Expected: RUDRA starts with Reconstruct > Use the model unticked, the Log says it was turned off,"
    Write-Host "  and Open... opens a clip. File > Model packages says the list is optional, and Use on RUDRA ACES 2"
    Write-Host "  says it is in use instead of 'Loading...'."
    Write-Host "  If File > Model packages lists a reconstruction package on this machine, that loads instead and"
    Write-Host "  nothing is turned off: the unit test above covers the empty case." -ForegroundColor DarkGray
    & powershell -ExecutionPolicy Bypass -File scripts\RUN_APP_ACES.ps1 -SkipTests
    if ($LASTEXITCODE -ne 0) { Fail "RUN_APP_ACES.ps1" }
}
if ($Package) {
    Say "The public package, unsigned, as release.yml builds it"
    & powershell -ExecutionPolicy Bypass -File scripts\PACKAGE_WINDOWS.ps1 -PublicRelease -NoArchive
    if ($LASTEXITCODE -ne 0) { Fail "PACKAGE_WINDOWS.ps1 -NoArchive" }
    $dir = Get-ChildItem dist\beta -Directory | Where-Object { $_.Name -match '^RUDRA-1\.2\.5(-.+)?-windows-x64$' } | Select-Object -First 1
    if (-not $dir) { Fail "no dist\beta\RUDRA-1.2.5-windows-x64 folder (is native\CMakeLists.txt at 1.2.5?)" }
    foreach ($f in "RUDRA.exe", "rudra-native.exe") {
        $vi = (Get-Item (Join-Path $dir.FullName $f)).VersionInfo
        Write-Host ("{0,-18} {1} {2}" -f $f, $vi.ProductName, $vi.ProductVersion)
        if ($vi.ProductName -ne "RUDRA" -or -not $vi.ProductVersion) { Fail "$f has no RUDRA version resource (SignPath would refuse it)" }
    }
    & powershell -ExecutionPolicy Bypass -File scripts\PACKAGE_WINDOWS_ARCHIVE.ps1 -Package $dir.FullName
    if ($LASTEXITCODE -ne 0) { Fail "PACKAGE_WINDOWS_ARCHIVE.ps1" }
    $setup = Join-Path $dir.Parent.FullName "$($dir.Name)-setup.exe"
    $vi = (Get-Item $setup).VersionInfo
    Write-Host ("installer          {0} {1}" -f $vi.ProductName, $vi.ProductVersion)
    Write-Host "  Run $setup once: a Privacy page after the licence, and 'Check for updates automatically'"
    Write-Host "  under Privacy on the tasks page. Unticked, HKCU\Software\FXTD Studios\RUDRA\updates\auto is false."
}
Write-Host "`nDone." -ForegroundColor Green
