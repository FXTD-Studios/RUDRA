[CmdletBinding()]
param(
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunName='fxtd_highlight_hdr_run01'
)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$pythonExe=Join-Path $repoRoot '.venv\Scripts\python.exe'
$runRoot=Join-Path 'G:\rudra_training' $RunName
$reportRoot=Join-Path 'G:\rudra_training\production_audits' (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
& $pythonExe (Join-Path $repoRoot 'training\audit_production_checkpoints.py') --run $runRoot --output (Join-Path $reportRoot "$RunName.json")
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint audit failed; do not infer a quality pass.' }
