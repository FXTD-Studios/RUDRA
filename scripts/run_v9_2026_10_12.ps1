# Line G: train v9 (clipped highlights, at 0 EV) and score it against the v9 gate.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_v9_2026_10_12.ps1 -Stage smoke   # ~5 min
#   powershell -ExecutionPolicy Bypass -File scripts\run_v9_2026_10_12.ps1 -Stage train   # ~4.5 h fine-tune (~11 h with -FromScratch)
#   powershell -ExecutionPolicy Bypass -File scripts\run_v9_2026_10_12.ps1 -Stage bench   # ~1.5 h
#
# Every stage is resumable: re-running skips what is finished.
#
# Why v9 (claude/TASK_CLIPPED_HIGHLIGHTS.md, 12 Oct 2026): v8 at 0 EV with region
# focus did not learn what is above a blown highlight (+0 EV -0.36 stops worse than
# the inverse inside clipped pixels, ties at +1/+2), because its corpus barely clips
# (median clipped fraction ~0 %). Meanwhile the analytic clip fill (core/clip_fill)
# took the clipped error at +1 EV from 0.61 to ~0.30 stops and at +2 EV from 0.91 to
# ~0.55, with no model. v9 changes against v8:
#   data   the same manifest (rudra_mix_v4c_netflix_20261008b, 0 EV), and with
#          probability --exposure-push 0.5 a sample re-rendered from its own HDR at
#          +E EV, E uniform in [0, 2] (the clip benches' ACES render) with its target
#          x 2^E: clipped highlights in about half the batches, exact below the clip.
#   target --region-focus 0.5, as v8.
#   start  v8's best.pt (same 0 EV baseline), 20k steps; -FromScratch for 50k.
#
# Gate (training\cp8_verdicts.py --clip-against clipfill), written before this run:
# as v8's (real Meridian both CIs above the inverse; aces/oog/mix not worse), and
# inside clipped pixels at +0, +1, +2 EV the error lower than the CLIP FILL's (not
# the flat inverse's), CI above zero. A model that only matches the analytic fill
# has nothing to add there.
param([ValidateSet("smoke", "train", "bench")][string]$Stage = "smoke", [int]$Steps = 20000,
      [double]$RegionFocus = 0.5, [double]$ExposurePush = 0.5, [double]$PushMaxEv = 2.0,
      [switch]$FromScratch, [string]$Name = "v9", [string]$Cli = "",
      # Bench a given checkpoint instead of $Out\best.pt (13 Oct 2026: best.pt is selected on
      # an unpushed val split, so it stayed at step 0 = v8's weights), and/or only the clip benches.
      [string]$Checkpoint = "", [switch]$ClipOnly)
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

$V8 = "checkpoints\sdr2hdr_image_v8\best.pt"
if (-not $FromScratch -and -not (Test-Path $V8)) { Fail "missing $V8 (or run with -FromScratch)" }
function TrainArgs([string]$out, [int]$steps) {
    $a = @("training\train_sdr2hdr.py", "--mode", "image", "--manifest", $Mix, "--output-dir", $out,
           "--steps", "$steps", "--batch-size", "4", "--crop-size", "256", "--base-channels", "32",
           "--lr", $(if ($FromScratch) { "2e-4" } else { "1e-4" }), "--weight-decay", "1e-4", "--workers", "2", "--best-eval", "hard",
           "--best-metric", "composite_gain", "--best-smoothing", "1", "--eval-every", "2000",
           "--eval-batches", "0", "--save-every", "2000", "--seed", "20261012",
           "--region-focus", "$RegionFocus", "--exposure-push", "$ExposurePush", "--exposure-push-max-ev", "$PushMaxEv")
    if (-not $FromScratch) { $a += @("--init-checkpoint", $V8) }
    return $a
}

# rudra-native with still decode, for the clip fill's bench trees.
function FindCli {
    if ($Cli) { if (Test-Path $Cli) { return (Resolve-Path $Cli).Path } else { Fail "no $Cli" } }
    $found = Get-ChildItem (Join-Path $Repo "build") -Recurse -Filter "rudra-native.exe" -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $found) { Fail "rudra-native.exe not found under build\ (build native with OpenCV, or pass -Cli)" }
    return $found.FullName
}

if ($Stage -eq "smoke") {
    Stage "SMOKE  200 steps of $Name (loading, the exposure push, the region loss, memory, config)"
    $cliExe = FindCli
    # The subcommand exists in this build: on a folder with no ref/ it answers 66; a build
    # without it falls through to the usage, 64.
    & $cliExe bench-clip-fill (Join-Path $env:TEMP "rudra_no_bench") 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 66) { Fail "$cliExe has no bench-clip-fill (rebuild native with OpenCV, or pass -Cli)" }
    $smoke = "checkpoints\_smoke\${Name}_smoke"
    if (Test-Path $smoke) { Remove-Item $smoke -Recurse -Force }
    $a = (TrainArgs $smoke 200) + @("--eval-every", "100", "--save-every", "200", "--eval-batches", "8")
    if ((Run $a "${Name}_smoke_train.log") -ne 0) { Fail "smoke training failed (log: reports\logs\${Name}_smoke_train.log)" }
    $best = Join-Path $smoke "best.pt"
    if (-not (Test-Path $best)) { Fail "smoke run wrote no best.pt" }
    # The shared bench trees (bench\cp_*) were exported at corpus_ev 0 by v7: v8 must match or its
    # baseline would be a different render of the input.
    & $py -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['config']; assert float(c['region_focus'])>0 and float(c['exposure_push'])>0 and not c.get('curve_head') and not c.get('source_curve'), c; assert float(c['corpus_ev'])==0.0, c['corpus_ev']; print('config ok: region_focus', c['region_focus'], 'exposure_push', c['exposure_push'], 'corpus_ev', c['corpus_ev'])" $best
    if ($LASTEXITCODE -ne 0) { Fail "smoke checkpoint config is wrong" }
    Write-Host "`nSMOKE PASSED. Next: -Stage train" -ForegroundColor Green
}

if ($Stage -eq "train") {
    Stage "TRAIN  $Name, $Steps steps $(if ($FromScratch) { 'from scratch' } else { 'from v8' }), region focus $RegionFocus, exposure push $ExposurePush up to +$PushMaxEv EV"
    $final = Join-Path $Repo ("$Out\step_{0:D7}.pt" -f $Steps)
    if (Test-Path $final) { Write-Host "already trained" -ForegroundColor DarkGray }
    else {
        $a = TrainArgs $Out $Steps
        $last = Get-ChildItem (Join-Path $Repo $Out) -Filter "step_*.pt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
        if ($last) {
            # --resume and --init-checkpoint are exclusive: a resumed run already started from v8.
            $a = @($a | Where-Object { $_ -ne "--init-checkpoint" -and $_ -ne $V8 })
            $a += @("--resume", $last.FullName); Write-Host "resuming from $($last.Name)" -ForegroundColor Yellow
        }
        if ((Run $a "${Name}_train.log") -ne 0) { Fail "training failed (log: reports\logs\${Name}_train.log)" }
    }
    Write-Host "`nTRAINED. Next: -Stage bench" -ForegroundColor Green
}

if ($Stage -eq "bench") {
    Stage "BENCH  $Name on real / aces / oog / mix, and inside the clip at +0, +1, +2 EV"
    $ckpt = if ($Checkpoint) { $Checkpoint } else { Join-Path $Out "best.pt" }
    if (-not (Test-Path $ckpt)) { Fail "no $ckpt; run -Stage train first" }
    Write-Host "benching $ckpt as tree '$Name'" -ForegroundColor Yellow
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
    foreach ($b in @($benches.Keys)) {
        if ($ClipOnly -and -not $b.StartsWith("cp8_clip")) { continue }
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
    # The clip fill's trees (the analytic reconstruction's default since 12 Oct 2026),
    # scored the same way: the clip rows of the v9 gate are judged against them.
    $cliExe = FindCli
    foreach ($ev in 0, 1, 2) {
        $root = Join-Path $Repo "bench\cp8_clip_ev$ev"; $res = Join-Path $root "results"
        & $cliExe bench-clip-fill $root --corpus-ev 0 2>&1 | Tee-Object -FilePath (Join-Path $logDir "${Name}_clipfill_ev$ev.log") | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail "bench-clip-fill ev$ev failed" }
        if (-not (Test-Path (Join-Path $res "clipfill.regions.csv"))) {
            if ((Run @("training\score_regions.py", $root, "--test-dir", "clipfill") "${Name}_regions_clipfill_ev$ev.log") -ne 0) { Fail "region scoring clipfill ev$ev failed" }
        }
    }
    Run @("training\cp8_verdicts.py", "--bench-root", (Join-Path $Repo "bench"), "--candidate", $Name, "--clip-against", "clipfill", "--out", (Join-Path $logDir "cp8_results_$Name.json")) "${Name}_verdicts.log" | Out-Null
    Run @("training\cp8_verdicts.py", "--bench-root", (Join-Path $Repo "bench"), "--candidate", "clipfill", "--out", (Join-Path $logDir "cp8_results_clipfill.json")) "clipfill_verdicts.log" | Out-Null
    Write-Host "`nverdicts -> reports\logs\${Name}_verdicts.log (and the clip fill against the inverse in clipfill_verdicts.log), JSON -> reports\logs\cp8_results_$Name.json" -ForegroundColor Green
}
Stop-Transcript | Out-Null
