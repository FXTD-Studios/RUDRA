# Line G: train v8 (real SDR in the backbone, region-focused loss) and score it against the v8 gate.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_v8_2026_10_08.ps1 -Stage smoke   # ~5 min
#   powershell -ExecutionPolicy Bypass -File scripts\run_v8_2026_10_08.ps1 -Stage train   # ~11 h on the 4080
#   powershell -ExecutionPolicy Bypass -File scripts\run_v8_2026_10_08.ps1 -Stage bench   # ~1.5 h
#
# Every stage is resumable: re-running skips what is finished.
#
# What v8 changes against v7 (claude/rudra_model_v8_plan_2026-10-07.md):
#   data   rudra_mix_v4c_netflix_20261008b: v4c rendered + 5,494 real-SDR train
#          frames (3x v7's 1,830), carousel_fireworks dropped as before
#   target --region-focus 0.5: half the loss on the SDR's clipped/crushed pixels,
#          the rest of the frame held to the inverse (the tone fit is the artist's
#          in 1.0: source curve, calibration, reference match). No curve head, no
#          curve label: v7 showed the label hurt on hable and agx.
#   eval   the whole val split every 2,000 steps (--eval-batches 0), real and
#          rendered reported apart; best.pt picked on it, not on 32 records.
# Same recipe otherwise (v4c: 50k steps, batch 4, crop 256, 32 channels, seed 20260715).
#
# Gate (training\cp8_verdicts.py), written before this run: real Meridian both
# CIs above the inverse; aces/oog/mix not worse; inside clipped pixels at +0, +1,
# +2 EV the error lower than the inverse's, CI above zero.
param([ValidateSet("smoke", "train", "bench")][string]$Stage = "smoke", [int]$Steps = 50000,
      [double]$RegionFocus = 0.5, [string]$Name = "v8")
$ErrorActionPreference = "Continue"
$Repo = "D:\A.I\Devlopments\rudra"
Set-Location -LiteralPath $Repo
$logDir = Join-Path $Repo "reports\logs"; New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Start-Transcript -Path (Join-Path $logDir "${Name}_$Stage-transcript.log") -Append | Out-Null
$py = if (Test-Path ".venv\Scripts\python.exe") { (Resolve-Path ".venv\Scripts\python.exe").Path } else { "python" }
$env:OPENCV_IO_ENABLE_OPENEXR = "1"

$Mix     = "G:\datasets\corpora\rudra_mix_v4c_netflix_20261008b\sdr_hdr_manifest.jsonl"
$Netflix = "G:\datasets\corpora\rudra_netflix_realsdr_20261001\manifest.jsonl"
$V4c     = "G:\datasets\corpora\corpus_v4c\sdr_hdr_manifest.jsonl"
$v4bRoot = @("G:\corpus_v4b", "G:\datasets\corpora\corpus_v4b") | Where-Object { Test-Path (Join-Path $_ "_ingest_config.json") } | Select-Object -First 1
$HoldOut = "E:\RUDRA_v3_20260822\sdr_hdr_manifest.jsonl"
$Out     = "checkpoints\sdr2hdr_image_$Name"

function Stage([string]$n) { Write-Host ""; Write-Host ("=" * 72) -ForegroundColor Cyan; Write-Host "  $n" -ForegroundColor Cyan; Write-Host ("=" * 72) -ForegroundColor Cyan }
function Fail([string]$why) { Write-Host "`nSTOPPED: $why" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
function Run([string[]]$argv, [string]$log) {
    & $py @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir $log) | Out-Host
    return $LASTEXITCODE
}
foreach ($p in @($Mix, $Netflix, $V4c, $HoldOut)) { if (-not (Test-Path $p)) { Fail "missing: $p" } }
if (-not $v4bRoot) { Fail "corpus_v4b not found" }

function TrainArgs([string]$out, [int]$steps) {
    return @("training\train_sdr2hdr.py", "--mode", "image", "--manifest", $Mix, "--output-dir", $out,
             "--steps", "$steps", "--batch-size", "4", "--crop-size", "256", "--base-channels", "32",
             "--lr", "2e-4", "--weight-decay", "1e-4", "--workers", "2", "--best-eval", "hard",
             "--best-metric", "composite_gain", "--best-smoothing", "1", "--eval-every", "2000",
             "--eval-batches", "0", "--save-every", "2000", "--seed", "20260715",
             "--region-focus", "$RegionFocus")
}

if ($Stage -eq "smoke") {
    Stage "SMOKE  200 steps of $Name (loading, the region loss, memory, config)"
    $smoke = "checkpoints\_smoke\${Name}_smoke"
    if (Test-Path $smoke) { Remove-Item $smoke -Recurse -Force }
    $a = (TrainArgs $smoke 200) + @("--eval-every", "100", "--save-every", "200", "--eval-batches", "8")
    if ((Run $a "${Name}_smoke_train.log") -ne 0) { Fail "smoke training failed (log: reports\logs\${Name}_smoke_train.log)" }
    $best = Join-Path $smoke "best.pt"
    if (-not (Test-Path $best)) { Fail "smoke run wrote no best.pt" }
    # The shared bench trees (bench\cp_*) were exported at corpus_ev 0 by v7: v8 must match or its
    # baseline would be a different render of the input.
    & $py -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['config']; assert float(c['region_focus'])>0 and not c.get('curve_head') and not c.get('source_curve'), c; assert float(c['corpus_ev'])==0.0, c['corpus_ev']; print('config ok: region_focus', c['region_focus'], 'corpus_ev', c['corpus_ev'])" $best
    if ($LASTEXITCODE -ne 0) { Fail "smoke checkpoint config is wrong" }
    Write-Host "`nSMOKE PASSED. Next: -Stage train" -ForegroundColor Green
}

if ($Stage -eq "train") {
    Stage "TRAIN  $Name, $Steps steps from scratch, region focus $RegionFocus"
    $final = Join-Path $Repo ("$Out\step_{0:D7}.pt" -f $Steps)
    if (Test-Path $final) { Write-Host "already trained" -ForegroundColor DarkGray }
    else {
        $a = TrainArgs $Out $Steps
        $last = Get-ChildItem (Join-Path $Repo $Out) -Filter "step_*.pt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
        if ($last) { $a += @("--resume", $last.FullName); Write-Host "resuming from $($last.Name)" -ForegroundColor Yellow }
        if ((Run $a "${Name}_train.log") -ne 0) { Fail "training failed (log: reports\logs\${Name}_train.log)" }
    }
    Write-Host "`nTRAINED. Next: -Stage bench" -ForegroundColor Green
}

if ($Stage -eq "bench") {
    Stage "BENCH  $Name on real / aces / oog / mix, and inside the clip at +0, +1, +2 EV"
    $ckpt = Join-Path $Out "best.pt"
    if (-not (Test-Path $ckpt)) { Fail "no $ckpt; run -Stage train first" }
    $freeGB = [math]::Round((Get-PSDrive -Name ($Repo.Substring(0, 1))).Free / 1GB, 1)
    if ($freeGB -lt 15) { Fail "only $freeGB GB free on $($Repo.Substring(0, 2)); need 15 GB (four model trees and three clip benches)" }
    # name -> manifest, condition, extra export args
    $benches = [ordered]@{
        "cp_real"      = @{ manifest = $Netflix; condition = "clean"; extra = @() }
        "cp_aces"      = @{ manifest = (Join-Path $v4bRoot "sdr_hdr_manifest.jsonl"); condition = "clean"; extra = @() }
        "cp_oog"       = @{ manifest = $HoldOut; condition = "out-of-generator"; extra = @() }
        "cp_mix"       = @{ manifest = $V4c; condition = "clean"; extra = @() }
        "cp8_clip_ev0" = @{ manifest = $Netflix; condition = "exposure"; extra = @("--exposure-ev", "0", "--exposure-curve", "aces") }
        "cp8_clip_ev1" = @{ manifest = $Netflix; condition = "exposure"; extra = @("--exposure-ev", "1", "--exposure-curve", "aces") }
        "cp8_clip_ev2" = @{ manifest = $Netflix; condition = "exposure"; extra = @("--exposure-ev", "2", "--exposure-curve", "aces") }
    }
    foreach ($b in $benches.Keys) {
        $root = Join-Path $Repo "bench\$b"; $res = Join-Path $root "results"
        New-Item -ItemType Directory -Path $res -Force | Out-Null
        $clip = $b.StartsWith("cp8_clip")
        # ref/ and baseline/ are shared and reused with --only-test only when complete
        # (7 Oct 2026: a truncated ref/ was scored once). A partial tree is re-exported.
        $refDir = Join-Path $root "ref"; $baseDir = Join-Path $root "baseline"
        if (Test-Path $refDir) {
            $expected = @(Get-ChildItem $root -Filter "export*.json" | ForEach-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).frames }) | Sort-Object -Unique
            $nRef = @(Get-ChildItem $refDir -Recurse -File).Count
            $nBase = if (Test-Path $baseDir) { @(Get-ChildItem $baseDir -Recurse -File).Count } else { 0 }
            if ($expected.Count -ne 1 -or $nRef -ne $expected[0] -or $nBase -ne $expected[0]) {
                Write-Host "incomplete ref/baseline in $root (ref $nRef, baseline $nBase, exports say $($expected -join ',')): re-exporting everything" -ForegroundColor Yellow
                Get-ChildItem $root -Directory | Remove-Item -Recurse -Force
                Get-ChildItem $root -Filter "export*.json" | Remove-Item -Force
                New-Item -ItemType Directory -Path $res -Force | Out-Null
            }
        }
        $done = Join-Path $root "export_$Name.json"
        if ((Test-Path (Join-Path $root $Name)) -and -not (Test-Path $done)) {
            Write-Host "partial export $b/${Name}: re-exporting" -ForegroundColor Yellow
            Remove-Item (Join-Path $root $Name) -Recurse -Force
            Remove-Item (Join-Path $res "$Name.*") -Force -ErrorAction SilentlyContinue
        }
        if (-not (Test-Path $done)) {
            $a = @("training\export_bench_pairs.py", "--checkpoint", $ckpt, "--manifest", $benches[$b].manifest,
                   "--out", $root, "--split", "test", "--condition", $benches[$b].condition,
                   "--test-name", $Name) + $benches[$b].extra
            if (Test-Path $refDir) { $a += "--only-test" } elseif ($clip) { $a += "--write-sdr" }
            if ((Run $a "${Name}_export_$b.log") -ne 0) { Fail "export $b failed" }
        }
        if ($clip -or $b -eq "cp_real") {
            if (-not (Test-Path (Join-Path $res "$Name.regions.csv")) -or -not (Test-Path (Join-Path $res "baseline.regions.csv"))) {
                if ((Run @("training\score_regions.py", $root, "--test-dir", "baseline", $Name) "${Name}_regions_$b.log") -ne 0) { Fail "region scoring $b failed" }
            }
        }
        if (-not $clip) {
            foreach ($t in @($Name, "baseline")) {
                $json = Join-Path $res "$t.json"
                if (-not (Test-Path $json)) {
                    if ((Run @("-m", "rudra.delivery.cli", "bench", $root, "--nits-scale", "203", "--test-dir", $t, "--output", $json) "${Name}_bench_${b}_$t.log") -ne 0) { Fail "bench $b/$t failed" }
                }
            }
        }
    }
    Run @("training\cp8_verdicts.py", "--bench-root", (Join-Path $Repo "bench"), "--candidate", $Name, "--out", (Join-Path $logDir "cp8_results_$Name.json")) "${Name}_verdicts.log" | Out-Null
    Write-Host "`nverdicts -> reports\logs\${Name}_verdicts.log, JSON -> reports\logs\cp8_results_$Name.json" -ForegroundColor Green
}
Stop-Transcript | Out-Null
