<#
.SYNOPSIS
  Build and run the native unit tests (rudra_tests, rudra_app_tests) in the
  gate B build, which already has Qt. Re-runs are incremental.

    powershell -ExecutionPolicy Bypass -File scripts\NATIVE_TESTS.ps1
    powershell -ExecutionPolicy Bypass -File scripts\NATIVE_TESTS.ps1 -Filter "SourceCurve.*:Session.*"
    powershell -ExecutionPolicy Bypass -File scripts\NATIVE_TESTS.ps1 -SkipBuild

  Output in reports\native_tests_<stamp>.txt. Exit 1 if a test fails.
#>
param(
    [string]$Build = "build\native_gate_b",
    [string]$Filter = "*",
    [switch]$SkipBuild,
    [switch]$NoApp
)
$ErrorActionPreference = "Stop"
$Repo = "D:\A.I\Devlopments\rudra"
Set-Location -LiteralPath $Repo
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$Report = Join-Path $Repo "reports\native_tests_$Stamp.txt"
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
$log = @()
function Fail([string]$m) {
    if ($log.Count) { $log | Set-Content -Path $Report -Encoding UTF8; Write-Host "report: $Report" }
    Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1
}

if (-not (Test-Path (Join-Path $Build "CMakeCache.txt"))) { Fail "$Build is not configured; run scripts\NATIVE_GATE_B.ps1 first" }
$cache = Get-Content (Join-Path $Build "CMakeCache.txt") -Raw
if ($cache -notmatch "RUDRA_BUILD_TESTS:BOOL=ON") {
    Say "Configure $Build with the tests on (the cache keeps Qt, the generator and the rest)"
    & cmake -S native -B $Build -DRUDRA_BUILD_TESTS=ON -DRUDRA_BUILD_CLI=ON
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure" }
}
if (-not $SkipBuild) {
    Say "Build rudra_tests and rudra_app_tests (Release)"
    $targets = @("rudra_tests"); if (-not $NoApp) { $targets += "rudra_app_tests" }
    & cmake --build $Build --config Release --parallel --target @targets
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}

$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
$log = @("RUDRA native tests, $Stamp, build $Build, filter $Filter", "")
$failed = $false
$Core = Join-Path $Build "tests\Release\rudra_tests.exe"
if (-not (Test-Path $Core)) { Fail "not built: $Core" }
Say "rudra_tests"
$out = & $Core "--gtest_filter=$Filter" 2>&1 | ForEach-Object { "$_" }
$out | Out-Host; $log += "--- rudra_tests"; $log += $out
if ($LASTEXITCODE -ne 0) { $failed = $true }
if (-not $NoApp) {
    $App = Join-Path $Build "app\Release\rudra_app_tests.exe"   # add_executable in app/CMakeLists.txt, so app\, not app\tests\
    if (-not (Test-Path $App)) { Fail "not built: $App" }
    Say "rudra_app_tests (offscreen)"
    $env:QT_QPA_PLATFORM = "offscreen"
    # The Qt the build was configured with: its DLLs and its platform plugins.
    $QtRoot = ([regex]::Match($cache, "Qt6_DIR:PATH=(.*)/lib/cmake/Qt6")).Groups[1].Value -replace "/", "\\"
    if ($QtRoot) { $env:PATH = "$QtRoot\bin;" + $env:PATH; $env:QT_PLUGIN_PATH = "$QtRoot\plugins" }
    $out = & $App "--gtest_filter=$Filter" 2>&1 | ForEach-Object { "$_" }
    $out | Out-Host; $log += ""; $log += "--- rudra_app_tests"; $log += $out
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}
$ErrorActionPreference = $prev
$log | Set-Content -Path $Report -Encoding UTF8
Write-Host "`nreport: $Report" -ForegroundColor Green
if ($failed) { Fail "a test failed (see the report)" }
Write-Host "ALL PASSED" -ForegroundColor Green
