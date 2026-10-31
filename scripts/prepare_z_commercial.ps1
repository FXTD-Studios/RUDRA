param(
    [ValidateSet('Start','Foreground','Resume','Status')]
    [string]$Action = 'Status',
    [string]$OutputDirectory = 'G:\datasets\corpora\rudra_z_pipeline_20261018_v1',
    [switch]$TrainWhenReady
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoPath '.venv\Scripts\python.exe'
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
if ($TrainWhenReady -and $outputPath.TrimEnd('\') -ne 'G:\datasets\corpora\rudra_z_pipeline_20261018_v1') {
    throw 'Automatic training is pinned to the reviewed default corpus. Use a separately reviewed launcher for another output directory.'
}

if ($Action -eq 'Status') {
    $statusPath = Join-Path $outputPath 'status.json'
    if (Test-Path -LiteralPath $statusPath) { Get-Content -LiteralPath $statusPath }
    $logPath = Join-Path $outputPath 'preparation.log'
    if (Test-Path -LiteralPath $logPath) { Get-Content -LiteralPath $logPath -Tail 8 }
    $errorPath = Join-Path $outputPath 'preparation_errors.log'
    if ((Test-Path -LiteralPath $errorPath) -and (Get-Item -LiteralPath $errorPath).Length -gt 0) {
        Get-Content -LiteralPath $errorPath -Tail 8
    }
    exit 0
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
if ($Action -eq 'Start') {
    if (Test-Path -LiteralPath (Join-Path $outputPath 'preparation.lock')) {
        throw 'Preparation lock exists. Inspect the existing process before starting another job.'
    }
    $shellPath = (Get-Process -Id $PID).Path
    $argsForChild = @('-NoProfile','-ExecutionPolicy','Bypass','-File',
        ('"' + $PSCommandPath + '"'),'-Action','Foreground','-OutputDirectory',('"' + $outputPath + '"'))
    if ($TrainWhenReady) { $argsForChild += '-TrainWhenReady' }
    $child = Start-Process -FilePath $shellPath -ArgumentList $argsForChild -WorkingDirectory $repoPath `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'preparation.log') `
        -RedirectStandardError (Join-Path $outputPath 'preparation_errors.log')
    Write-Output "Started preparation process $($child.Id). Output: $outputPath"
    exit 0
}

Push-Location $repoPath
try {
    if (-not (Test-Path -LiteralPath (Join-Path $outputPath 'ready.json'))) {
        $stageName = 'all'
        if ((Test-Path -LiteralPath (Join-Path $outputPath 'plan.json')) -and
            (Test-Path -LiteralPath (Join-Path $outputPath 'qc_summary.json'))) {
            $stageName = 'prepare'
        }
        & $pythonPath -u (Join-Path $repoPath 'pipeline\prepare_z_footage.py') --stage $stageName --out $outputPath
        if ($LASTEXITCODE -ne 0) { throw "Preparation stopped with exit code $LASTEXITCODE" }
        & $pythonPath -u (Join-Path $repoPath 'pipeline\verify_z_footage.py') --out $outputPath
        if ($LASTEXITCODE -ne 0) { throw "Verification stopped with exit code $LASTEXITCODE" }
    }
    if ($TrainWhenReady) {
        $deadline = [DateTime]::UtcNow.AddHours(12)
        do {
            $activeTrainers = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
                $_.CommandLine -match '(run_fxtd_commercial|run_z_commercial|train_sdr2hdr)\.py'
            })
            if ($activeTrainers.Count -eq 0) { break }
            if ([DateTime]::UtcNow -gt $deadline) { throw 'Previous trainer still active after 12 hours. New training was not started.' }
            Start-Sleep -Seconds 15
        } while ($true)
        Write-Output 'Verified dataset ready and previous trainer exited. Starting the separate bounded scratch run.'
        & (Join-Path $PSScriptRoot 'train_z_commercial.ps1') -Action Start -RunName 'fxtd_z_hdr_run01' -Steps 20000 -MaxHours 8
    }
} finally {
    Pop-Location
}
