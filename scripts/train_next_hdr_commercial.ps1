[CmdletBinding()]
param(
    [ValidateSet('Check','Start','Foreground','Prepare','Status','Stop')][string]$Action='Status',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunName='fxtd_native1080_hdr_run01',
    [ValidateRange(200,50000)][int]$Steps=20000,
    [ValidateRange(1,24)][int]$MaxHours=24
)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$pythonExe=Join-Path $repoRoot '.venv\Scripts\python.exe'
$runRoot=Join-Path 'G:\rudra_training' $RunName
if ($Action -in @('Status','Stop')) {
    & (Join-Path $PSScriptRoot 'train_fxtd_commercial.ps1') -Action $Action -RunName $RunName
    return
}
& $pythonExe (Join-Path $repoRoot 'pipeline\next_hdr_readiness.py') check
if ($LASTEXITCODE -ne 0) { throw 'Next dataset is not ready. Finish preparation and actual visual review before training.' }
if ($Action -eq 'Check') { return }
if ($Action -in @('Start','Foreground')) {
    $active=@(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -match '(run_\w*commercial|train_sdr2hdr|resumable_training)\.py' -and $_.CommandLine -notmatch '--prepare-only'
    })
    if ($active.Count -gt 0) { throw 'Another HDR trainer is active. Let it finish before starting this next run.' }
}
foreach ($name in @('runner.lock','STOP','checkpoints\config.json')) {
    if (Test-Path -LiteralPath (Join-Path $runRoot $name)) { throw 'Run already started/stopped. This is a scratch launcher; use a fresh RunName.' }
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$runnerArgs=@('-u',(Join-Path $repoRoot 'training\run_next_hdr_commercial.py'),'--run-dir',$runRoot,'--steps',"$Steps",'--max-hours',"$MaxHours")
if ($Action -eq 'Prepare') { $runnerArgs+='--prepare-only' }
if ($Action -eq 'Start') {
    $quoted=($runnerArgs | ForEach-Object { '"'+$_+'"' }) -join ' '
    $child=Start-Process -FilePath $pythonExe -ArgumentList $quoted -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runRoot 'console.log') -RedirectStandardError (Join-Path $runRoot 'errors.log')
    Write-Output "Started process $($child.Id). Run: $runRoot"
} else {
    Push-Location -LiteralPath $repoRoot
    try { & $pythonExe @runnerArgs; if ($LASTEXITCODE -ne 0) { throw "Run preparation/training failed: $LASTEXITCODE" } }
    finally { Pop-Location }
}
