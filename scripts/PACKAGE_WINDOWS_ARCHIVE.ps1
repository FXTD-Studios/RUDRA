<#
.SYNOPSIS
  The ZIP and the installer of a Windows package folder that PACKAGE_WINDOWS.ps1 -NoArchive built
  (1.2.5). release.yml runs it after SignPath has signed RUDRA.exe and rudra-native.exe in that
  folder, so the ZIP and the installer carry the signed files; the installer itself is signed after.

    powershell -ExecutionPolicy Bypass -File scripts\PACKAGE_WINDOWS_ARCHIVE.ps1 -Package dist\beta\RUDRA-1.2.5-windows-x64

  Writes, beside the folder: <name>.zip and <name>-setup.exe, each with its .sha256.
  -NoInstaller makes the ZIP alone. Inno Setup 6 must be installed (PACKAGE_WINDOWS.ps1 installs it
  for this user when it is missing).
#>
param(
    [Parameter(Mandatory = $true)][string]$Package,
    [string]$InnoSetup = "",
    [switch]$NoInstaller,
    [switch]$ModelFreeEvaluation
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
$Repo = Split-Path -Parent $PSScriptRoot

$Out = (Resolve-Path -LiteralPath $Package).Path
$Name = Split-Path $Out -Leaf
if ($Name -notmatch '^RUDRA-(.+)-windows-x64$') { Fail "not a package folder (RUDRA-<version>-windows-x64): $Out" }
$Version = $Matches[1]
foreach ($f in "RUDRA.exe", "rudra-native.exe", "LICENSE", "PRIVACY.txt") {
    if (-not (Test-Path (Join-Path $Out $f))) { Fail "$f missing from $Out" }
}
$exeVersion = (Get-Item (Join-Path $Out "RUDRA.exe")).VersionInfo.ProductVersion
if ($exeVersion -ne $Version) { Fail "RUDRA.exe says $exeVersion, the folder $Version" }
foreach ($f in "RUDRA.exe", "rudra-native.exe") {
    $s = Get-AuthenticodeSignature (Join-Path $Out $f)
    Write-Host ("{0,-18} {1}{2}" -f $f, $s.Status, $(if ($s.SignerCertificate) { "  " + $s.SignerCertificate.Subject } else { "" }))
}

Say "ZIP"
$Zip = "$Out.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path $Out -DestinationPath $Zip -CompressionLevel Optimal
$hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()
"$hash  $Name.zip" | Set-Content -Encoding ascii "$Zip.sha256"
Write-Host "$hash  $Name.zip"

if (-not $NoInstaller) {
    Say "Installer (Inno Setup)"
    $iscc = $null
    foreach ($c in @($InnoSetup, "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
                     "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if ($c -and (Test-Path $c)) { $iscc = $c; break }
    }
    if (-not $iscc) { $iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source }
    if (-not $iscc) { Fail "Inno Setup 6 not found (PACKAGE_WINDOWS.ps1 installs it, or https://jrsoftware.org/isdl.php)" }
    $iss = Join-Path $Repo "native\app\windows\rudra.iss"
    $mode = @(); if ($ModelFreeEvaluation) { $mode = @('/DModelFreeEvaluation=1') }
    & $iscc /Q "/DAppVersion=$Version" "/DSourceDir=$Out" "/DOutputDir=$(Split-Path $Out)" @mode $iss
    if ($LASTEXITCODE -ne 0) { Fail "ISCC" }
    $Setup = Join-Path (Split-Path $Out) "$Name-setup.exe"
    if (-not (Test-Path $Setup)) { Fail "no installer at $Setup" }
    $sh = (Get-FileHash $Setup -Algorithm SHA256).Hash.ToLower()
    "$sh  $Name-setup.exe" | Set-Content -Encoding ascii "$Setup.sha256"
    Write-Host "$sh  $Name-setup.exe"
}
Write-Host "`nBuilt: $Zip"
if (-not $NoInstaller) { Write-Host "       $Setup  (sign it, then write its .sha256 again)" }
