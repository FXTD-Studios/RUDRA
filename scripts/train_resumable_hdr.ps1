[CmdletBinding()]
param(
    [ValidateSet('Status','Inspect','Start','Resume','Stop')][string]$Action='Status',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunName,
    # JSON file with {"trainer_args": [...]} (same format as a run's recipe.json). Start only.
    [string]$Recipe,
    [ValidateRange(1,24)][double]$MaxHours=24,
    [ValidateRange(0,50000)][int]$ResumeSaveEvery=200,
    [string]$Checkpoint,
    [switch]$RecoverStaleLock
)
# Resume-capable training for FUTURE runs (training/resumable_training.py).
# Existing scratch runs and their launchers are not changed by this script.
#   Status  : read <run>\checkpoints\resume_status.json and lineage.json
#   Inspect : classify a checkpoint (exact resume state / legacy / weights only)
#   Start   : new run directory only; refuses an existing one
#   Resume  : exact continuation of <run> from its newest VALID state
#   Stop    : graceful; the trainer finishes the step, saves, exits with code 3
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$pythonExe=Join-Path $repoRoot '.venv\Scripts\python.exe'
$trainer=Join-Path $repoRoot 'training\resumable_training.py'
if ($Action -eq 'Inspect') {
    if (-not $Checkpoint) { throw 'Inspect needs -Checkpoint' }
    & $pythonExe $trainer inspect $Checkpoint; return
}
if (-not $RunName) { throw "$Action needs -RunName" }
$runRoot=Join-Path 'G:\rudra_training' $RunName
$ckpt=Join-Path $runRoot 'checkpoints'
if ($Action -eq 'Status') {
    $found=$false
    foreach ($name in @('resume_status.json','lineage.json','train.lock')) {
        $p=Join-Path $ckpt $name
        if (Test-Path -LiteralPath $p) { $found=$true; Write-Output "--- $name"; Get-Content -LiteralPath $p -Raw }
    }
    if (-not $found) {
        # Before the first step the trainer is still hashing the dataset (commercial preflight),
        # and it writes nothing into checkpoints\ until that passes. Show the logs instead.
        Write-Output 'No status yet: the trainer is starting (dataset preflight) or has exited. Latest logs:'
        foreach ($log in @(Get-ChildItem -LiteralPath $runRoot -Filter 'console_*.log' | Sort-Object LastWriteTime | Select-Object -Last 1) +
                         @(Get-ChildItem -LiteralPath $runRoot -Filter 'errors_*.log' | Sort-Object LastWriteTime | Select-Object -Last 1)) {
            if ($log) { Write-Output "--- $($log.Name)"; Get-Content -LiteralPath $log.FullName -Tail 15 }
        }
    }
    return
}
if ($Action -eq 'Stop') {
    if (-not (Test-Path -LiteralPath (Join-Path $ckpt 'run_identity.json'))) {
        # A STOP file written before the trainer creates its run identity would make a fresh
        # start see a non-empty checkpoint folder and refuse. Nothing has been trained yet.
        throw 'Nothing to stop safely yet: the run has not passed preflight (no run_identity.json). Wait for Status to show a step, or end the process if you want to abandon this start.'
    }
    New-Item -ItemType File -Path (Join-Path $ckpt 'STOP') -Force | Out-Null
    Write-Output 'Stop requested; the trainer saves a resume state at the end of its current step.'
    return
}
$active=@(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -match '(run_\w*commercial|train_sdr2hdr|resumable_training)\.py' -and
    $_.CommandLine -notmatch '--prepare-only' -and $_.CommandLine -notmatch ' inspect '
})
if ($active.Count -gt 0) {
    $active | ForEach-Object { Write-Output ("PID {0} (parent {1}): {2}" -f $_.ProcessId, $_.ParentProcessId, $_.CommandLine) }
    throw "Another HDR trainer is active (PID $($active.ProcessId -join ', ')). Let it finish, or stop it if it is a leftover."
}
$stop=Join-Path $ckpt 'STOP'
if ($Action -eq 'Start') {
    if (Test-Path -LiteralPath $runRoot) { throw 'Run directory exists. Start needs a new RunName; use -Action Resume to continue.' }
    if (-not $Recipe) { throw 'Start needs -Recipe (JSON with trainer_args)' }
    $argsList=@((Get-Content -LiteralPath $Recipe -Raw | ConvertFrom-Json).trainer_args)
    $i=[Array]::IndexOf($argsList,'--output-dir')
    if ($i -lt 0) { $argsList+=@('--output-dir',$ckpt) } else { $argsList[$i+1]=$ckpt }
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    Copy-Item -LiteralPath $Recipe -Destination (Join-Path $runRoot 'recipe.json')
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $ckpt 'run_identity.json'))) { throw 'Not a resumable run (no run_identity.json). Existing scratch runs cannot be resumed exactly.' }
    if (Test-Path -LiteralPath $stop) { Move-Item -LiteralPath $stop -Destination (Join-Path $ckpt ("STOP.consumed_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))) }
    $argsList=@((Get-Content -LiteralPath (Join-Path $runRoot 'recipe.json') -Raw | ConvertFrom-Json).trainer_args)
    $i=[Array]::IndexOf($argsList,'--output-dir')
    if ($i -lt 0) { $argsList+=@('--output-dir',$ckpt) } else { $argsList[$i+1]=$ckpt }
    $argsList+='--resume'
    if ($RecoverStaleLock) { $argsList+='--recover-stale-lock' }
}
$argsList+=@('--max-hours',"$MaxHours",'--resume-save-every',"$ResumeSaveEvery")
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$quoted=(@('-u',$trainer)+$argsList | ForEach-Object { '"'+$_+'"' }) -join ' '
$child=Start-Process -FilePath $pythonExe -ArgumentList $quoted -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runRoot "console_$stamp.log") -RedirectStandardError (Join-Path $runRoot "errors_$stamp.log")
Write-Output "Started process $($child.Id) ($Action). Run: $runRoot. Logs: console_$stamp.log / errors_$stamp.log"
