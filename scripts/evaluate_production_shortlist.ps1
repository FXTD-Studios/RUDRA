[CmdletBinding()]
param(
    [string]$Shortlist='G:\rudra_training\production_audits\20261021_run02_agent\shortlist_20261021.json',
    [string]$Kit='G:\rudra_training\production_eval_20261019\validation_kit',
    [string]$Root='G:\rudra_training\production_eval_20261019\shortlist_run02_20261021',
    [switch]$DryRun
)
# Scores every shortlisted checkpoint with the unchanged production_eval_kit,
# then compares candidates with references. Re-running continues: finished
# evaluations of the same checkpoint bytes are reused, incomplete ones refuse.
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$pythonExe=Join-Path $repoRoot '.venv\Scripts\python.exe'
$runner=@((Join-Path $repoRoot 'training\production_shortlist_eval.py'),'--shortlist',$Shortlist,'--kit',$Kit,'--root',$Root)
if ($DryRun) { $runner+='--dry-run' }
Push-Location -LiteralPath $repoRoot
try {
    & $pythonExe @runner
    if ($LASTEXITCODE -ne 0) { throw 'Shortlist evaluation incomplete. Inspect the error; no quality pass is implied.' }
} finally { Pop-Location }
