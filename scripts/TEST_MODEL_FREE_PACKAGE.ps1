<# Validate a staged analytic package without developer dependencies on PATH.
   Writes only to a fresh report directory; never installs or publishes. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Package,
    [string]$InputClip = "",
    [string]$ReportRoot = ""
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$pkg = (Resolve-Path -LiteralPath $Package).Path
if (-not $ReportRoot) { $ReportRoot = Join-Path $repo 'reports' }
if (-not $InputClip) { $InputClip = Join-Path $repo 'native\tests\golden\video\clips\h264_709.mp4' }
$InputClip = (Resolve-Path -LiteralPath $InputClip).Path
$report = Join-Path $ReportRoot ('model_free_smoke_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $report | Out-Null
if (-not (Test-Path (Join-Path $pkg 'PACKAGE_MODE.txt'))) { throw 'Missing analytic package marker' }
if (Get-ChildItem (Join-Path $pkg 'models') -Force) { throw 'Models directory is not empty' }
$weights = Get-ChildItem $pkg -Recurse -File | Where-Object { $_.Extension -in '.pt','.pth','.onnx','.safetensors' }
if ($weights) { throw 'Unexpected weight files in package' }
$saved = @{}
foreach ($key in 'PATH','PYTHONHOME','PYTHONPATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM','RUDRA_FFMPEG_DIR') {
    $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
try {
    $env:PATH = "$pkg;$env:SystemRoot\System32;$env:SystemRoot"
    foreach ($key in 'PYTHONHOME','PYTHONPATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM','RUDRA_FFMPEG_DIR') {
        [Environment]::SetEnvironmentVariable($key, $null, 'Process')
    }
    $theme = Join-Path $report 'theme.json'
    $p = Start-Process -FilePath (Join-Path $pkg 'RUDRA.exe') -ArgumentList @('--theme-check', "`"$theme`"") -WindowStyle Hidden -Wait -PassThru
    if ($p.ExitCode -ne 0 -or -not (Test-Path $theme)) { throw 'Packaged app startup failed' }
    if (-not (Get-Content $theme -Raw | ConvertFrom-Json).ok) { throw 'Theme check failed' }
    $grade = Join-Path $report 'grade.json'
    # Windows PowerShell 5.1 is available on clean Windows without installing pwsh.
    [IO.File]::WriteAllText($grade, '{"analytic":true,"source_curve":"bt1886"}', (New-Object Text.UTF8Encoding($false)))
    foreach ($format in 'hdr10','prores422hq') {
        $ext = if ($format -eq 'hdr10') { 'mp4' } else { 'mov' }
        $out = Join-Path $report "$format.$ext"
        & (Join-Path $pkg 'rudra-native.exe') video - $inputClip --output $out --grade $grade --format $format --audio none --preset ultrafast *> (Join-Path $report "$format.log")
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { throw "$format analytic delivery failed" }
        $sidecar = Get-Content "$out.json" -Raw | ConvertFrom-Json
        if ($sidecar.reconstruction -ne 'analytic (no model)') { throw 'Unexpected reconstruction mode' }
    }
    @{pass=$true; package=$pkg; weights=0; restricted_path=$env:PATH; checks=@('app theme','analytic HDR10 delivery','analytic ProRes delivery')} |
        ConvertTo-Json | Set-Content (Join-Path $report 'summary.json')
    Write-Output "PASS: $report"
} finally {
    foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
}
