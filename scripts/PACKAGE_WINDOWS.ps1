<#
.SYNOPSIS
  The RUDRA beta for Windows x64: a self-contained folder and its ZIP.

.DESCRIPTION
  Builds RUDRA.exe and rudra-native.exe (Release; ONNX Runtime with DirectML,
  OpenCV still decode, the QRhi viewer on Qt 6.8), then makes
    dist\beta\RUDRA-<version>-windows-x64\
      RUDRA.exe, rudra-native.exe, Qt (windeployqt), the MSVC runtime,
      onnxruntime.dll, DirectML.dll, opencv_world*.dll,
      models\   the package(s) from dist\models
      ffmpeg\   ffmpeg.exe, ffprobe.exe and their LICENSE (gyan.dev "full" build, GPL)
      LICENSE, NOTICE, LICENSE-weights, "Read me first.md"
  dist\beta\RUDRA-<version>-windows-x64.zip, and the installer
  dist\beta\RUDRA-<version>-windows-x64-setup.exe (Inno Setup 6, installed for
  this user from jrsoftware.org when missing; -NoInstaller skips it), each with
  its SHA-256. The MSVC runtime is copied from Visual Studio's VC\Redist folder.

  Needs what NATIVE_PHASE3_EXIT.ps1 needs: Visual Studio 2022 or 2026 (or the
  Build Tools) with the C++ tools, a Python for aqtinstall, and a model package
  (NATIVE_GATE_A.ps1 -Checkpoint checkpoints\sdr2hdr_image_v8.pt exports dist\models\sdr2hdr_image_v8).

  Movies need an ffmpeg with libx265, prores_ks and zscale. The package
  carries one (since beta 6): the gyan.dev "full" build of the version below,
  downloaded once into tmp\native_deps and checked against its pinned SHA-256,
  in <package>\ffmpeg\, which the app and the CLI put first on their PATH
  (platform/tools.hpp; RUDRA_FFMPEG_DIR=path makes them use the PATH's own).
  That build is GPL v3: its LICENSE and README (with where its source is) go
  beside it, and NOTICE says so. -NoFfmpeg leaves it out.

  Signing (optional): with RUDRA_SIGN_PFX (a .pfx code-signing certificate)
  and RUDRA_SIGN_PASSWORD set, RUDRA.exe, rudra-native.exe and the installer
  are signed with signtool and timestamped; without them nothing is signed and
  Windows SmartScreen warns on first start.

.EXAMPLE
  .\scripts\PACKAGE_WINDOWS.ps1
  .\scripts\PACKAGE_WINDOWS.ps1 -Tests        # also build and run the app's Qt tests
  .\scripts\PACKAGE_WINDOWS.ps1 -SkipBuild    # package the last build
  .\scripts\PACKAGE_WINDOWS.ps1 -NoInstaller  # the ZIP alone, without Inno Setup
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$QtVersion = "6.8.3",
    [string]$OrtVersion = "1.22.0",
    [string]$OpenCvVersion = "4.10.0",
    [string]$InnoSetup = "",
    [switch]$SkipBuild,
    [switch]$Tests,
    [switch]$NoInstaller,
    [switch]$InstallBuildTools,
    [switch]$NoFfmpeg,
    [string]$FfmpegVersion = "7.1.1",
    # The pinned build's SHA-256 (GyanD/codexffmpeg release asset, checked 10 Oct 2026).
    # Another -FfmpegVersion needs its own: -FfmpegSha256 <hash>.
    [string]$FfmpegSha256 = "d760e1b3574402ed18b4865851f87d87e73965a982e6453212df8621fed1c508"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Deps = Join-Path $Repo "tmp\native_deps"
$QtRoot = Join-Path $Deps "Qt\$QtVersion\msvc2022_64"
$Build = Join-Path $Repo "build\native_release"
New-Item -ItemType Directory -Force -Path $Deps | Out-Null

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
. (Join-Path $PSScriptRoot "native_toolchain.ps1")

$Models = Join-Path $Repo "dist\models"
if (-not (Get-ChildItem $Models -Recurse -Filter manifest.json -ErrorAction SilentlyContinue)) {
    Fail "no model package in dist\models (run NATIVE_GATE_A.ps1 first)"
}

# ---------------------------------------------------------------------------
Say "Dependencies"
if (-not (Test-Path (Join-Path $QtRoot "bin\qsb.exe"))) {
    & $Python -m pip install --quiet --upgrade aqtinstall
    if ($LASTEXITCODE -ne 0) { Fail "pip install aqtinstall" }
    & $Python -m aqt install-qt windows desktop $QtVersion win64_msvc2022_64 -m qtshadertools -O (Join-Path $Deps "Qt")
    if ($LASTEXITCODE -ne 0) { Fail "aqt install-qt" }
}
Write-Host "Qt at $QtRoot"

function Get-Nupkg($id, $version, $dest) {
    if (-not (Test-Path $dest)) {
        $zip = Join-Path $Deps "$id.$version.zip"
        $url = "https://api.nuget.org/v3-flatcontainer/$($id.ToLower())/$version/$($id.ToLower()).$version.nupkg"
        Write-Host "download $url"
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $dest -Force
        Remove-Item $zip
    }
    return $dest
}
$OrtPkg = Get-Nupkg "Microsoft.ML.OnnxRuntime.DirectML" $OrtVersion (Join-Path $Deps "ort-dml-$OrtVersion")
$nuspec = Get-ChildItem $OrtPkg -Filter *.nuspec | Select-Object -First 1
[xml]$spec = Get-Content $nuspec.FullName
$dmlDep = $spec.SelectNodes("//*[local-name()='dependency' and @id='Microsoft.AI.DirectML']") | Select-Object -First 1
$DmlVersion = if ($dmlDep) { $dmlDep.version.Trim("[]() ").Split(",")[0] } else { "1.15.4" }
$DmlPkg = Get-Nupkg "Microsoft.AI.DirectML" $DmlVersion (Join-Path $Deps "directml-$DmlVersion")
$OrtRoot = Join-Path $Deps "ort-root-$OrtVersion"
New-Item -ItemType Directory -Force -Path "$OrtRoot\include", "$OrtRoot\lib" | Out-Null
Copy-Item "$OrtPkg\build\native\include\*" "$OrtRoot\include\" -Force
Copy-Item "$OrtPkg\runtimes\win-x64\native\*" "$OrtRoot\lib\" -Force
$DmlDll = Get-ChildItem "$DmlPkg\bin\x64-win" -Filter DirectML.dll | Select-Object -First 1
if (-not $DmlDll) { Fail "DirectML.dll not found in $DmlPkg" }
Write-Host "ONNX Runtime $OrtVersion (DirectML), DirectML $DmlVersion"

$CvRoot = Join-Path $Deps "opencv-$OpenCvVersion"
$CvBuild = Join-Path $CvRoot "opencv\build"
if (-not (Test-Path (Join-Path $CvBuild "OpenCVConfig.cmake"))) {
    $exe = Join-Path $Deps "opencv-$OpenCvVersion-windows.exe"
    $url = "https://github.com/opencv/opencv/releases/download/$OpenCvVersion/opencv-$OpenCvVersion-windows.exe"
    Write-Host "download $url"
    Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    $p = Start-Process -FilePath $exe -ArgumentList @("-o`"$CvRoot`"", "-y") -Wait -PassThru
    if ($p.ExitCode -ne 0 -or -not (Test-Path (Join-Path $CvBuild "OpenCVConfig.cmake"))) { Fail "OpenCV extract" }
    Remove-Item $exe
}
$CvLib = Get-ChildItem (Join-Path $CvBuild "x64") -Directory | Where-Object { $_.Name -match "^vc\d+$" } |
         Sort-Object { [int]($_.Name.Substring(2)) } -Descending | Select-Object -First 1
if (-not $CvLib) { Fail "no vcNN folder under $CvBuild\x64" }
$CvConfigDir = Join-Path $CvLib.FullName "lib"
$CvDll = Get-ChildItem (Join-Path $CvBuild "x64") -Recurse -Filter "opencv_world*.dll" |
         Where-Object { $_.Name -notmatch "d\.dll$" } | Select-Object -First 1
if (-not $CvDll) { Fail "opencv_world DLL not found under $CvBuild" }

# ffmpeg (product item 2): the gyan.dev "full" build, pinned by hash.
$FfDir = $null
if (-not $NoFfmpeg) {
    $FfRoot = Join-Path $Deps "ffmpeg-$FfmpegVersion-full_build"
    if (-not (Test-Path (Join-Path $FfRoot "bin\ffmpeg.exe"))) {
        $zip = Join-Path $Deps "ffmpeg-$FfmpegVersion-full_build.zip"
        $url = "https://github.com/GyanD/codexffmpeg/releases/download/$FfmpegVersion/ffmpeg-$FfmpegVersion-full_build.zip"
        Write-Host "download $url"
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($got -ne $FfmpegSha256.ToLower()) {
            Remove-Item $zip -Force
            Fail "ffmpeg $FfmpegVersion SHA-256 is $got, expected $FfmpegSha256 (a different file: not packaged)"
        }
        Expand-Archive -Path $zip -DestinationPath $Deps -Force
        Remove-Item $zip
    }
    foreach ($f in @("bin\ffmpeg.exe", "bin\ffprobe.exe", "LICENSE")) {
        if (-not (Test-Path (Join-Path $FfRoot $f))) { Fail "ffmpeg build incomplete: $f missing in $FfRoot" }
    }
    $FfDir = $FfRoot
    Write-Host "ffmpeg $FfmpegVersion (gyan.dev full build) at $FfRoot"
}

# Signing: signtool from the Windows SDK, with RUDRA_SIGN_PFX / RUDRA_SIGN_PASSWORD.
function Find-SignTool {
    $c = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $kits = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
    if (Test-Path $kits) {
        $t = Get-ChildItem $kits -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -match "\\x64\\" } | Sort-Object FullName -Descending | Select-Object -First 1
        if ($t) { return $t.FullName }
    }
    return $null
}
$Sign = $null
if ($env:RUDRA_SIGN_PFX) {
    if (-not (Test-Path $env:RUDRA_SIGN_PFX)) { Fail "RUDRA_SIGN_PFX names no file: $env:RUDRA_SIGN_PFX" }
    $Sign = Find-SignTool
    if (-not $Sign) { Fail "RUDRA_SIGN_PFX is set but signtool.exe was not found (Windows SDK)" }
    Write-Host "signing with $Sign"
}
function Invoke-Sign([string[]]$files) {
    if (-not $Sign) { return }
    $a = @("sign", "/fd", "SHA256", "/f", $env:RUDRA_SIGN_PFX, "/tr", "http://timestamp.digicert.com", "/td", "SHA256")
    if ($env:RUDRA_SIGN_PASSWORD) { $a += @("/p", $env:RUDRA_SIGN_PASSWORD) }
    & $Sign @a @files
    if ($LASTEXITCODE -ne 0) { Fail "signtool sign" }
    & $Sign verify /pa @files | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "signtool verify" }
}

# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Say "Configure and build (Visual Studio, Release)"
    $tc = Find-NativeToolchain
    if (-not $tc -and $InstallBuildTools) {
        if (-not (Install-NativeBuildTools)) { Fail "Build Tools install" }
        $tc = Find-NativeToolchain
    }
    if (-not $tc) { Fail "no C++ toolchain (see above)" }
    $cmake = $tc.CMake
    Reset-StaleCMakeCache $Build $tc.Generator
    $testsFlag = if ($Tests) { "ON" } else { "OFF" }
    & $cmake -S native -B $Build -G $tc.Generator -A x64 `
        "-DRUDRA_BUILD_TESTS=$testsFlag" -DRUDRA_BUILD_CLI=ON -DRUDRA_BUILD_APP=ON -DRUDRA_BUILD_RENDER=ON `
        -DRUDRA_WITH_ONNXRUNTIME=ON "-DONNXRUNTIME_ROOT=$OrtRoot" -DRUDRA_WITH_LIBTORCH=OFF `
        -DRUDRA_WITH_OPENCV=ON "-DOpenCV_DIR=$CvConfigDir" "-DCMAKE_PREFIX_PATH=$QtRoot"
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure" }
    $targets = @("RUDRA", "rudra-native")
    if ($Tests) { $targets += "rudra_app_tests" }
    & $cmake --build $Build --config Release --parallel --target $targets
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}
$App = Join-Path $Build "app\Release\RUDRA.exe"
$Cli = Join-Path $Build "cli\Release\rudra-native.exe"
foreach ($e in @($App, $Cli)) { if (-not (Test-Path $e)) { Fail "not built: $e" } }

if ($Tests) {
    Say "The app's Qt tests (offscreen)"
    $AppTests = Join-Path $Build "app\Release\rudra_app_tests.exe"
    & (Join-Path $QtRoot "bin\windeployqt.exe") --release --no-translations $AppTests | Out-Null
    $plat = Join-Path (Split-Path $AppTests) "platforms"
    New-Item -ItemType Directory -Force -Path $plat | Out-Null
    Copy-Item (Join-Path $QtRoot "plugins\platforms\qoffscreen.dll") $plat -Force
    Copy-Item "$OrtRoot\lib\*.dll", $DmlDll.FullName, $CvDll.FullName (Split-Path $AppTests) -Force
    $env:QT_QPA_PLATFORM = "offscreen"
    & $AppTests
    $code = $LASTEXITCODE
    Remove-Item Env:QT_QPA_PLATFORM
    if ($code -ne 0) { Fail "app tests (exit $code)" }
}

# ---------------------------------------------------------------------------
$Version = (Get-Item $App).VersionInfo.ProductVersion
if (-not $Version) { Fail "RUDRA.exe has no version resource" }
$Name = "RUDRA-$Version-windows-x64"
$Out = Join-Path $Repo "dist\beta\$Name"
$Zip = "$Out.zip"
Say "Package $Name"
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force }
if (Test-Path $Zip) { Remove-Item $Zip -Force }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Copy-Item $App, $Cli $Out
Invoke-Sign @((Join-Path $Out "RUDRA.exe"), (Join-Path $Out "rudra-native.exe"))
# Qt and its plugins beside the executables.
& (Join-Path $QtRoot "bin\windeployqt.exe") --release --no-translations (Join-Path $Out "RUDRA.exe")
if ($LASTEXITCODE -ne 0) { Fail "windeployqt" }
# The MSVC runtime from Visual Studio's own redistributable folder, so the
# package runs on a PC that never installed it (windeployqt --compiler-runtime
# needs VCINSTALLDIR, which only a Developer prompt sets).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vsPath) { Fail "Visual Studio with the C++ tools not found (the MSVC runtime comes from it)" }
$crt = Get-ChildItem (Join-Path $vsPath "VC\Redist\MSVC") -Directory -ErrorAction SilentlyContinue |
       Where-Object { $_.Name -match "^\d+\." } | Sort-Object { [version]$_.Name } -Descending |
       ForEach-Object { Get-ChildItem (Join-Path $_.FullName "x64") -Directory -Filter "Microsoft.VC14*.CRT" -ErrorAction SilentlyContinue } |
       Select-Object -First 1
if (-not $crt) { Fail "no VC\Redist\MSVC\<version>\x64\Microsoft.VC14*.CRT under $vsPath" }
Copy-Item (Join-Path $crt.FullName "*.dll") $Out -Force
foreach ($d in @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")) {
    if (-not (Test-Path (Join-Path $Out $d))) { Fail "$d missing from the package" }
}
Write-Host "MSVC runtime: $($crt.FullName)"
# ONNX Runtime and DirectML beside the exe: Windows' own older onnxruntime.dll
# in System32 would otherwise be loaded first.
Copy-Item "$OrtRoot\lib\*.dll", $DmlDll.FullName, $CvDll.FullName $Out -Force
$OutModels = Join-Path $Out "models"
New-Item -ItemType Directory -Force -Path $OutModels | Out-Null
Copy-Item "$Models\*" $OutModels -Recurse -Force
Get-ChildItem $OutModels -Recurse -Include *.safetensors, *.config.json | Remove-Item -Force
Copy-Item LICENSE, NOTICE $Out
if (Test-Path checkpoints\LICENSE) { Copy-Item checkpoints\LICENSE (Join-Path $Out "LICENSE-weights") }
if (Test-Path docs\BETA.md) { Copy-Item docs\BETA.md (Join-Path $Out "Read me first.md") }
if (Test-Path docs\USER_GUIDE.md) { Copy-Item docs\USER_GUIDE.md (Join-Path $Out "User guide.md") }
if ($FfDir) {
    $OutFf = Join-Path $Out "ffmpeg"
    New-Item -ItemType Directory -Force -Path $OutFf | Out-Null
    Copy-Item (Join-Path $FfDir "bin\ffmpeg.exe"), (Join-Path $FfDir "bin\ffprobe.exe") $OutFf
    Copy-Item (Join-Path $FfDir "LICENSE") (Join-Path $OutFf "LICENSE.txt")
    if (Test-Path (Join-Path $FfDir "README.txt")) { Copy-Item (Join-Path $FfDir "README.txt") $OutFf }
    @(
        "ffmpeg $FfmpegVersion, the gyan.dev ""full"" build (https://www.gyan.dev/ffmpeg/builds/),",
        "unmodified, from https://github.com/GyanD/codexffmpeg/releases/tag/$FfmpegVersion",
        "(SHA-256 of the archive: $FfmpegSha256).",
        "",
        "It is licensed under the GNU GPL version 3 (LICENSE.txt). RUDRA runs it as a separate",
        "program and is not linked to it. Its source: https://ffmpeg.org/releases/ and the",
        "libraries listed in README.txt; FXTD Studios will also provide the corresponding source",
        "on request (https://github.com/FXTD-Studios/RUDRA/issues) for three years from this release."
    ) | Set-Content -Encoding utf8 (Join-Path $OutFf "SOURCE.txt")
}

Say "Check the package runs from where it is"
$pkg = Get-ChildItem $OutModels -Directory | Where-Object { Test-Path (Join-Path $_.FullName "manifest.json") } | Select-Object -First 1
& (Join-Path $Out "rudra-native.exe") info $pkg.FullName | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "rudra-native in the package cannot read its model" }
if ($FfDir) {
    & (Join-Path $Out "ffmpeg\ffmpeg.exe") -hide_banner -h encoder=prores_ks | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "the packaged ffmpeg does not run" }
}
$check = Join-Path $env:TEMP "rudra-theme-check.json"
$p = Start-Process -FilePath (Join-Path $Out "RUDRA.exe") -ArgumentList @("--theme-check", "`"$check`"") -Wait -PassThru
if ($p.ExitCode -ne 0) { Fail "RUDRA.exe in the package does not start (exit $($p.ExitCode))" }

Say "ZIP"
Compress-Archive -Path $Out -DestinationPath $Zip -CompressionLevel Optimal
$hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()
"$hash  $Name.zip" | Set-Content -Encoding ascii "$Zip.sha256"
Write-Host "$hash  $Name.zip"

# ---------------------------------------------------------------------------
# The installer (Inno Setup 6): RUDRA-<version>-windows-x64-setup.exe, per-user
# by default, Start menu entry, uninstaller, the licence shown first.
if (-not $NoInstaller) {
    Say "Installer (Inno Setup)"
    $iscc = $null
    foreach ($c in @($InnoSetup, "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
                     "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if ($c -and (Test-Path $c)) { $iscc = $c; break }
    }
    if (-not $iscc) { $iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source }
    if (-not $iscc) {
        # Not installed: the official installer, silently and for this user only
        # (no administrator prompt), into %LOCALAPPDATA%\Programs\Inno Setup 6.
        Write-Host "Inno Setup 6 not found; installing it for this user from jrsoftware.org"
        # The GitHub release asset: jrsoftware.org/download.php answers with an
        # HTML page, not the installer.
        $isExe = Join-Path $env:TEMP "innosetup-6.7.3.exe"
        Invoke-WebRequest -Uri "https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe" `
            -OutFile $isExe -UseBasicParsing
        $head = [System.IO.File]::ReadAllBytes($isExe)[0..1]
        if ((Get-Item $isExe).Length -lt 1MB -or $head[0] -ne 0x4D -or $head[1] -ne 0x5A) {
            Fail "the Inno Setup download is not an installer ($isExe)"
        }
        Unblock-File $isExe
        $p = Start-Process -FilePath $isExe -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER") -Wait -PassThru
        Remove-Item $isExe -ErrorAction SilentlyContinue
        $c = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
        if ($p.ExitCode -eq 0 -and (Test-Path $c)) { $iscc = $c }
    }
    if (-not $iscc) {
        Fail "Inno Setup 6 not found and could not be installed (https://jrsoftware.org/isdl.php, or -NoInstaller for the ZIP alone)"
    }
    $iss = Join-Path $Repo "native\app\windows\rudra.iss"
    & $iscc /Q "/DAppVersion=$Version" "/DSourceDir=$Out" "/DOutputDir=$(Split-Path $Out)" $iss
    if ($LASTEXITCODE -ne 0) { Fail "ISCC" }
    $Setup = Join-Path (Split-Path $Out) "$Name-setup.exe"
    if (-not (Test-Path $Setup)) { Fail "no installer at $Setup" }
    Invoke-Sign @($Setup)
    $sh = (Get-FileHash $Setup -Algorithm SHA256).Hash.ToLower()
    "$sh  $Name-setup.exe" | Set-Content -Encoding ascii "$Setup.sha256"
    Write-Host "$sh  $Name-setup.exe"
}
Write-Host "`nBuilt: $Zip"
if (-not $NoInstaller) { Write-Host "       $Setup" }
