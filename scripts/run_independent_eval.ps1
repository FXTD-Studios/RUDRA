[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('Overlap','SihdrEvaluate','TosBuild','TosEvaluate','Freeze','TestPrepare','TestEvaluate')][string]$Step,
    [string]$Checkpoint,
    [string]$Label,
    [string]$Out,
    [string]$OwnerDecision,
    # Interpreter override (TosBuild needs OpenColorIO >= 2.6 for the frozen ACES 2.1 built-in config)
    [string]$Python
)
# Independent / final evaluation steps after shortlist selection. Each step
# writes to a NEW directory and refuses to overwrite. Protocols were frozen on
# 21 Oct 2026 before any model output existed; do not edit them to fit scores.
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$py=if ($Python) { $Python } else { Join-Path $repoRoot '.venv\Scripts\python.exe' }
$evalRoot='G:\rudra_training\production_eval_20261019'
$proto=Join-Path $evalRoot 'independent_protocol_20261021'
$src='G:\datasets\sources\next_hdr_sources_20261019_v3'
$manifest='G:\rudra_training\fxtd_native1080_hdr_run02\data\manifest.jsonl'
function Need($v,$n){ if (-not $v) { throw "$Step needs -$n" } }
$env:OPENCV_IO_ENABLE_OPENEXR='1'
Push-Location -LiteralPath $repoRoot
try {
    switch ($Step) {
        'Overlap' { & $py pipeline\screen_eval_overlap.py --eval-dir "$src\sihdr\unpacked" --manifest $manifest --out (Join-Path $proto 'sihdr_overlap_screen.json') }
        'SihdrEvaluate' { Need $Checkpoint 'Checkpoint'; Need $Out 'Out'
            & $py training\sihdr_eval.py evaluate --protocol (Join-Path $proto 'sihdr_protocol_v1_1.json') --root "$src\sihdr\unpacked" --checkpoint $Checkpoint --out $Out }
        'TosBuild' { & $py training\tos_temporal_eval.py build --tos "$src\tears_of_steel\exr" --out (Join-Path $evalRoot 'tos_temporal_kit_v1') }
        'TosEvaluate' { Need $Checkpoint 'Checkpoint'; Need $Out 'Out'
            & $py training\tos_temporal_eval.py evaluate --kit (Join-Path $evalRoot 'tos_temporal_kit_v1') --checkpoint $Checkpoint --out $Out }
        'Freeze' { Need $Checkpoint 'Checkpoint'; Need $Label 'Label'
            & $py training\freeze_candidate.py --checkpoint $Checkpoint --selection (Join-Path $evalRoot 'shortlist_run02_20261021\summary.json') --label $Label --out (Join-Path $evalRoot "frozen_candidate_$Label.json") @(if ($OwnerDecision) { '--owner-decision', $OwnerDecision }) }
        'TestPrepare' { & $py training\production_test_kit.py prepare --manifest $manifest --out (Join-Path $evalRoot 'test_kit_v1') }
        'TestEvaluate' { Need $Label 'Label'
            & $py training\production_test_kit.py evaluate --kit (Join-Path $evalRoot 'test_kit_v1') --freeze (Join-Path $evalRoot "frozen_candidate_$Label.json") --out (Join-Path $evalRoot "test_$Label") }
    }
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE); no quality pass is implied." }
} finally { Pop-Location }
