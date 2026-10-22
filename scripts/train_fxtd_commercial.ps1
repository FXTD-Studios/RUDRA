[CmdletBinding()]
param(
    [ValidateSet('Start','Foreground','Prepare','Status','Stop')][string]$Action = 'Start',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunName = 'fxtd_hdr_20261018_v1',
    [ValidateRange(200,50000)][int]$Steps = 20000,
    [ValidateRange(1,24)][int]$MaxHours = 8
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runnerPath = Join-Path $repoRoot 'training\run_fxtd_commercial.py'
$runRoot = Join-Path 'G:\rudra_training' $RunName
$statusPath = Join-Path $runRoot 'status.json'
if ($Action -eq 'Status') {
    if (Test-Path -LiteralPath $statusPath) { Get-Content -LiteralPath $statusPath }
    else { Write-Output 'No status yet.' }
    $logPath = Join-Path $runRoot 'console.log'
    if (Test-Path -LiteralPath $logPath) { Get-Content -LiteralPath $logPath -Tail 5 }
    $errorPath = Join-Path $runRoot 'errors.log'
    if (Test-Path -LiteralPath $errorPath) { Get-Content -LiteralPath $errorPath -Tail 5 }
    return
}
if ($Action -eq 'Stop') {
    if (-not (Test-Path -LiteralPath $runRoot)) { throw 'Run does not exist.' }
    Set-Content -LiteralPath (Join-Path $runRoot 'STOP') -Value 'User requested stop'
    Write-Output 'Stop requested. Previously saved checkpoints remain available.'
    return
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'runner.lock')) {
    throw 'This run has a runner lock. Use -Action Status. Do not start a duplicate.'
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'checkpoints\config.json')) {
    throw 'This run already has checkpoints. Guarded resume is not supported; use a new -RunName to restart from scratch.'
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'STOP')) { throw 'This run was stopped. Use a new -RunName.' }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$runnerArgs = @('-u', $runnerPath, '--run-dir', $runRoot, '--steps', "$Steps", '--max-hours', "$MaxHours")
if ($Action -eq 'Prepare') { $runnerArgs += '--prepare-only' }
if ($Action -eq 'Start') {
    $quotedArgs = ($runnerArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $proc = Start-Process -FilePath $pythonExe -ArgumentList $quotedArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runRoot 'console.log') -RedirectStandardError (Join-Path $runRoot 'errors.log')
    Write-Output "Started process $($proc.Id). Run: $runRoot"
    Write-Output "Check progress with: .\scripts\train_fxtd_commercial.ps1 -Action Status -RunName $RunName"
} else {
    & $pythonExe @runnerArgs
    if ($LASTEXITCODE -ne 0) { throw "Training/preparation failed with exit code $LASTEXITCODE" }
}
