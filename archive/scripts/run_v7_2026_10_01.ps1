# Roadmap 1.4: train v7 (source-curve model) and score it against the N8 gates.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_v7_2026_10_01.ps1 -Stage smoke   # ~5 min
#   powershell -ExecutionPolicy Bypass -File scripts\run_v7_2026_10_01.ps1 -Stage train   # ~10 h on the 4080
#   powershell -ExecutionPolicy Bypass -File scripts\run_v7_2026_10_01.ps1 -Stage bench   # ~30-60 min
#
# Every stage is resumable: re-running skips what is finished.
#
# v7 trains FROM SCRATCH on rudra_mix_v4c_netflix_20261001, with the v4c recipe
# (same flags, seed, 50k steps) plus --source-curve. Not warm-started from v4c:
# v4c trained on carousel_fireworks, a corpus_v4b TEST scene that is in the
# aces bench and in configs\compare_set_v1.json, and warm-starting would carry
# that into v7. From scratch also makes v7 vs v4c a clean comparison: only the
# data (real SDR added, carousel dropped) and the curve input differ.
#
# Gates (training\cp7_verdicts.py, N8/*) were fixed before this run.
param([ValidateSet("smoke", "train", "bench")][string]$Stage = "smoke", [int]$Steps = 50000)
$ErrorActionPreference = "Continue"
$Repo = "D:\A.I\Devlopments\rudra"
Set-Location -LiteralPath $Repo
$logDir = Join-Path $Repo "reports\logs"; New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Start-Transcript -Path (Join-Path $logDir "v7_$Stage-transcript.log") -Append | Out-Null
$py = if (Test-Path ".venv\Scripts\python.exe") { (Resolve-Path ".venv\Scripts\python.exe").Path } else { "python" }
$env:OPENCV_IO_ENABLE_OPENEXR = "1"

$Mix     = "G:\datasets\corpora\rudra_mix_v4c_netflix_20261001\sdr_hdr_manifest.jsonl"
$Netflix = "G:\datasets\corpora\rudra_netflix_realsdr_20261001\manifest.jsonl"
$V4c     = "G:\datasets\corpora\corpus_v4c\sdr_hdr_manifest.jsonl"
$v4bRoot = @("G:\corpus_v4b", "G:\datasets\corpora\corpus_v4b") | Where-Object { Test-Path (Join-Path $_ "_ingest_config.json") } | Select-Object -First 1
$HoldOut = "E:\RUDRA_v3_20260822\sdr_hdr_manifest.jsonl"
$Out     = "checkpoints\sdr2hdr_image_v7"

function Stage([string]$n) { Write-Host ""; Write-Host ("=" * 72) -ForegroundColor Cyan; Write-Host "  $n" -ForegroundColor Cyan; Write-Host ("=" * 72) -ForegroundColor Cyan }
function Fail([string]$why) { Write-Host "`nSTOPPED: $why" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
function Run([string[]]$argv, [string]$log) {
    & $py @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir $log) | Out-Host
    return $LASTEXITCODE
}
foreach ($p in @($Mix, $Netflix, $V4c, $HoldOut)) { if (-not (Test-Path $p)) { Fail "missing: $p" } }
if (-not $v4bRoot) { Fail "corpus_v4b not found" }

# The v4c recipe (scripts\critical_path_2026-09-23.ps1, Train) plus the curve input.
function TrainArgs([string]$out, [int]$steps) {
    return @("training\train_sdr2hdr.py", "--mode", "image", "--manifest", $Mix, "--output-dir", $out,
             "--steps", "$steps", "--batch-size", "4", "--crop-size", "256", "--base-channels", "32",
             "--lr", "2e-4", "--weight-decay", "1e-4", "--workers", "2", "--best-eval", "hard",
             "--best-metric", "composite_gain", "--best-smoothing", "5", "--eval-every", "500",
             "--save-every", "2000", "--seed", "20260715",
             "--curve-head", "--source-curve", "--source-curve-dropout", "0.3")
}

if ($Stage -eq "smoke") {
    Stage "SMOKE  200 steps of v7 on the mixed manifest (checks loading, labels, memory)"
    $smoke = "checkpoints\_smoke\v7_smoke"
    if (Test-Path $smoke) { Remove-Item $smoke -Recurse -Force }
    $a = (TrainArgs $smoke 200) + @("--eval-every", "100", "--save-every", "200")
    if ((Run $a "v7_smoke_train.log") -ne 0) { Fail "smoke training failed (log: reports\logs\v7_smoke_train.log)" }
    $best = Join-Path $smoke "best.pt"
    if (-not (Test-Path $best)) { Fail "smoke run wrote no best.pt" }
    & $py -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['config']; assert c['source_curve'] and c['curve_head'], c; print('config ok: source_curve', c['source_curve'], 'corpus_ev', c['corpus_ev'])" $best
    if ($LASTEXITCODE -ne 0) { Fail "smoke checkpoint config is wrong" }
    Write-Host "`nSMOKE PASSED. Next: -Stage train" -ForegroundColor Green
}

if ($Stage -eq "train") {
    Stage "TRAIN  v7, $Steps steps from scratch"
    $final = Join-Path $Repo ("$Out\step_{0:D7}.pt" -f $Steps)
    if (Test-Path $final) { Write-Host "already trained" -ForegroundColor DarkGray }
    else {
        $a = TrainArgs $Out $Steps
        $last = Get-ChildItem (Join-Path $Repo $Out) -Filter "step_*.pt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
        if ($last) { $a += @("--resume", $last.FullName); Write-Host "resuming from $($last.Name)" -ForegroundColor Yellow }
        if ((Run $a "v7_train.log") -ne 0) { Fail "training failed (log: reports\logs\v7_train.log)" }
    }
    Write-Host "`nTRAINED. Next: -Stage bench" -ForegroundColor Green
}

if ($Stage -eq "bench") {
    Stage "BENCH  v7 with the true curve and blind, on aces / oog / mix, plus real SDR"
    $ckpt = Join-Path $Out "best.pt"
    if (-not (Test-Path $ckpt)) { Fail "no $ckpt; run -Stage train first" }
    # Each export tree is 2 to 3 GB; the run on 7 Oct 2026 died with D: full.
    $freeGB = [math]::Round((Get-PSDrive -Name ($Repo.Substring(0, 1))).Free / 1GB, 1)
    if ($freeGB -lt 12) { Fail "only $freeGB GB free on $($Repo.Substring(0, 2)); need 12 GB for the remaining exports (delete scored bench trees first)" }
    # bench -> manifest, condition, the TRUE curve of that bench's SDR
    $benches = [ordered]@{
        "aces" = @{ manifest = (Join-Path $v4bRoot "sdr_hdr_manifest.jsonl"); condition = "clean"; curve = "aces" }
        "oog"  = @{ manifest = $HoldOut; condition = "out-of-generator"; curve = "hable" }
        "mix"  = @{ manifest = $V4c; condition = "clean"; curve = "from-manifest" }
        "real" = @{ manifest = $Netflix; condition = "clean"; curve = "" }
    }
    foreach ($b in $benches.Keys) {
        $root = Join-Path $Repo "bench\cp_$b"; $res = Join-Path $root "results"
        New-Item -ItemType Directory -Path $res -Force | Out-Null
        # ref/ and baseline/ are shared by every model's export and reused with
        # --only-test. They are only reusable if complete: on 7 Oct 2026 the real
        # bench's first export died with D: full at frame 166 of 287 (one EXR
        # truncated), the re-run saw ref/ and exported only the test tree, and
        # the scorer failed on the short file. Complete = every export_*.json
        # here agrees on "frames" and ref/ and baseline/ both hold that many files.
        $refDir = Join-Path $root "ref"; $baseDir = Join-Path $root "baseline"
        if (Test-Path $refDir) {
            $expected = @(Get-ChildItem $root -Filter "export_*.json" | ForEach-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).frames }) | Sort-Object -Unique
            $nRef = @(Get-ChildItem $refDir -Recurse -File).Count
            $nBase = if (Test-Path $baseDir) { @(Get-ChildItem $baseDir -Recurse -File).Count } else { 0 }
            if ($expected.Count -ne 1 -or $nRef -ne $expected[0] -or $nBase -ne $expected[0]) {
                Write-Host "incomplete ref/baseline in $root (ref $nRef, baseline $nBase, exports say $($expected -join ',')): re-exporting everything" -ForegroundColor Yellow
                Get-ChildItem $root -Directory | Remove-Item -Recurse -Force
                Get-ChildItem $root -Filter "export_*.json" | Remove-Item -Force
                New-Item -ItemType Directory -Path $res -Force | Out-Null
            }
        }
        $runs = [ordered]@{ "v7_unknown" = "none" }
        if ($benches[$b].curve) { $runs["v7"] = $benches[$b].curve }
        foreach ($tree in $runs.Keys) {
            # Resume on the export's manifest, not the directory: export_bench_pairs
            # writes export_<tree>.json last, so a tree without it is a partial
            # export (7 Oct 2026: aces/v7_unknown was scored on 366 of 537
            # frames after an interrupted first attempt). A partial tree is
            # removed with its scores and exported again.
            $done = Join-Path $root "export_$tree.json"
            if ((Test-Path (Join-Path $root $tree)) -and -not (Test-Path $done)) {
                Write-Host "partial export $b/$tree (no $done): re-exporting" -ForegroundColor Yellow
                Remove-Item (Join-Path $root $tree) -Recurse -Force
                Remove-Item (Join-Path $res "$tree.json"), (Join-Path $res "$tree.csv") -Force -ErrorAction SilentlyContinue
            }
            if (-not (Test-Path $done)) {
                $a = @("training\export_bench_pairs.py", "--checkpoint", $ckpt, "--manifest", $benches[$b].manifest,
                       "--out", $root, "--split", "test", "--condition", $benches[$b].condition,
                       "--test-name", $tree, "--source-curve", $runs[$tree])
                if (Test-Path (Join-Path $root "ref")) { $a += "--only-test" }
                if ((Run $a "v7_export_${b}_$tree.log") -ne 0) { Fail "export $b/$tree failed" }
            }
            foreach ($t in @($tree, "baseline")) {
                $json = Join-Path $res "$t.json"
                if (-not (Test-Path $json)) {
                    if ((Run @("-m", "rudra.delivery.cli", "bench", $root, "--nits-scale", "203", "--test-dir", $t, "--output", $json) "v7_bench_${b}_$t.log") -ne 0) { Fail "bench $b/$t failed" }
                }
            }
        }
    }
    Run @("training\cp7_verdicts.py", "--bench-root", (Join-Path $Repo "bench"), "--out", (Join-Path $logDir "cp_results_v7.json")) "v7_verdicts.log" | Out-Null
    Write-Host "`nverdicts -> reports\logs\v7_verdicts.log (N8 rows), JSON -> reports\logs\cp_results_v7.json" -ForegroundColor Green
}
Stop-Transcript | Out-Null
