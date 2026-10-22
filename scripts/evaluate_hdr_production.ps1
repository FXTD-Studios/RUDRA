[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Kit='G:\rudra_training\production_eval_20261019\validation_kit',
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
& (Join-Path $repoRoot '.venv\Scripts\python.exe') (Join-Path $repoRoot 'training\production_eval_kit.py') evaluate --kit $Kit --checkpoint $Checkpoint --out $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Evaluation incomplete. Inspect the error; no quality pass is implied.' }
