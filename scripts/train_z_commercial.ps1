[CmdletBinding()]
param(
    [ValidateSet('Start','Foreground','Prepare','Status','Stop')][string]$Action = 'Status',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunName = 'fxtd_z_hdr_run01',
    [ValidateRange(200,50000)][int]$Steps = 20000,
    [ValidateRange(1,24)][int]$MaxHours = 8
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoPath '.venv\Scripts\python.exe'
$runPath = Join-Path 'G:\rudra_training' $RunName
if ($Action -in @('Status','Stop')) {
    & (Join-Path $PSScriptRoot 'train_fxtd_commercial.ps1') -Action $Action -RunName $RunName
    return
}
if (-not (Test-Path -LiteralPath 'G:\datasets\corpora\rudra_z_pipeline_20261018_v1\ready.json')) {
    throw 'Expanded dataset is not ready. Check prepare_z_commercial.ps1 -Action Status.'
}
if ($Action -in @('Start','Foreground')) {
    $active = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -match '(run_fxtd_commercial|run_z_commercial|train_sdr2hdr)\.py'
    })
    if ($active.Count -gt 0) { throw 'Another HDR trainer is active. Let it finish before starting this separate run.' }
}
foreach ($name in @('runner.lock','STOP','checkpoints\config.json')) {
    if (Test-Path -LiteralPath (Join-Path $runPath $name)) { throw 'Run already started or stopped. Inspect Status; scratch restart requires a new RunName.' }
}
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$runnerArgs = @('-u',(Join-Path $repoPath 'training\run_z_commercial.py'),
    '--run-dir',$runPath,'--steps',"$Steps",'--max-hours',"$MaxHours")
if ($Action -eq 'Prepare') { $runnerArgs += '--prepare-only' }
if ($Action -eq 'Start') {
    $quotedArgs = ($runnerArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $child = Start-Process -FilePath $pythonPath -ArgumentList $quotedArgs -WorkingDirectory $repoPath `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runPath 'console.log') `
        -RedirectStandardError (Join-Path $runPath 'errors.log')
    Write-Output "Started process $($child.Id). Run: $runPath"
} else {
    & $pythonPath @runnerArgs
    if ($LASTEXITCODE -ne 0) { throw "Training/preparation failed: $LASTEXITCODE" }
}
