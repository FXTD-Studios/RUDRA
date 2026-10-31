# v10 (13 Oct 2026): the residual on the Source picker's curve, selected on what the gate scores.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_v10_2026_10_13.ps1 -Stage smoke   # ~10 min
#   powershell -ExecutionPolicy Bypass -File scripts\run_v10_2026_10_13.ps1 -Stage train   # ~5 h fine-tune from v9 step 20,000
#   powershell -ExecutionPolicy Bypass -File scripts\run_v10_2026_10_13.ps1 -Stage bench   # ~2 h
#
# Every stage is resumable: re-running skips what is finished. -ClipOnly benches only
# the clip rows (the cheapest gate to fail first); -Checkpoint benches a given step.
#
# Why (docs/MODEL_PLAN_V10.md): every model so far learned a correction to the ACES
# inverse, while on real SDR the BT.1886 inverse alone is +14 dB over it (R3) and
# is RUDRA 1.0's default. v10 against v9:
#   baseline  the Source picker's curve (--baseline-source): BT.1886 for real SDR and
#             degraded samples, a rendered sample's own curve, BT.1886 for half of
#             those anyway (--baseline-dropout 0.5: the artist who never picks)
#   shadows   --crushed-hold 1.0: changing a crushed black is charged (v8/v9 were
#             1.6 to 2.4 stops worse than the inverse there, 0 of 287 frames better)
#   val       live-action real val (pipeline/build_v10_manifest.py: held-out shots of
#             Nocturne and Sparks; Cosmos Laundromat, an animation, to train)
#   select    --best-metric bench_gain (training/bench_val.py): full-frame PU21 on real
#             and rendered val, the clipped error at +1/+2 EV and the crushed error,
#             against the picker's baseline; v9's best.pt never left step 0
#   start     v9 step 20,000 (it learned the clip at +1/+2 EV), 20k steps, lr 1e-4
#
# Gate (training\cp10_verdicts.py), written before this run: real SDR above the BT.1886
# baseline (PU21 and CVVDP CIs above zero); aces / oog / mix not worse; inside clipped
# pixels at +0, +1, +2 EV better than the CLIP FILL; crushed pixels not worse anywhere.
param([ValidateSet("smoke", "train", "bench")][string]$Stage = "smoke", [int]$Steps = 20000,
      [double]$RegionFocus = 0.5, [double]$ExposurePush = 0.5, [double]$PushMaxEv = 2.0,
      [double]$CrushedHold = 1.0, [double]$BaselineDropout = 0.5,
      [string]$Name = "v10", [string]$Cli = "", [string]$Init = "checkpoints\sdr2hdr_image_v9\step_0020000.pt",
      [string]$Checkpoint = "", [switch]$ClipOnly)
$ErrorActionPreference = "Continue"
$Repo = "D:\A.I\Devlopments\rudra"
Set-Location -LiteralPath $Repo
$logDir = Join-Path $Repo "reports\logs"; New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Start-Transcript -Path (Join-Path $logDir "${Name}_$Stage-transcript.log") -Append | Out-Null
$py = if (Test-Path ".venv\Scripts\python.exe") { (Resolve-Path ".venv\Scripts\python.exe").Path } else { "python" }
$env:OPENCV_IO_ENABLE_OPENEXR = "1"

$Mix     = "G:\datasets\corpora\rudra_mix_v4c_netflix_20261008b\sdr_hdr_manifest.jsonl"
$MixV10  = "G:\datasets\corpora\rudra_mix_v10_20261013\sdr_hdr_manifest.jsonl"
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
if (-not (Test-Path $Init)) { Fail "missing $Init (v9's step 20,000; or pass -Init)" }

# The manifest with live-action real validation (once).
if (-not (Test-Path $MixV10)) {
    Stage "MANIFEST  live-action real val (pipeline\build_v10_manifest.py)"
    if ((Run @("pipeline\build_v10_manifest.py", "--manifest", $Mix, "--out", (Split-Path $MixV10)) "${Name}_manifest.log") -ne 0) { Fail "manifest build failed" }
}

function TrainArgs([string]$out, [int]$steps) {
    return @("training\train_sdr2hdr.py", "--mode", "image", "--manifest", $MixV10, "--output-dir", $out,
             "--steps", "$steps", "--batch-size", "4", "--crop-size", "256", "--base-channels", "32",
             "--lr", "1e-4", "--weight-decay", "1e-4", "--workers", "2", "--best-eval", "hard",
             "--best-metric", "bench_gain", "--best-smoothing", "3", "--bench-val-frames", "48",
             "--eval-every", "2000", "--eval-batches", "0", "--save-every", "2000", "--seed", "20261013",
             "--region-focus", "$RegionFocus", "--exposure-push", "$ExposurePush", "--exposure-push-max-ev", "$PushMaxEv",
             "--baseline-source", "--baseline-dropout", "$BaselineDropout", "--crushed-hold", "$CrushedHold",
             "--init-checkpoint", $Init)
}

function FindCli {
    if ($Cli) { if (Test-Path $Cli) { return (Resolve-Path $Cli).Path } else { Fail "no $Cli" } }
    $found = Get-ChildItem (Join-Path $Repo "build") -Recurse -Filter "rudra-native.exe" -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $found) { Fail "rudra-native.exe not found under build\ (build native with OpenCV, or pass -Cli)" }
    return $found.FullName
}

if ($Stage -eq "smoke") {
    Stage "SMOKE  200 steps of $Name (the picker baseline, the crushed hold, the bench validation, memory, config)"
    $cliExe = FindCli
    & $cliExe bench-clip-fill (Join-Path $env:TEMP "rudra_no_bench") 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 66) { Fail "$cliExe has no bench-clip-fill (rebuild native with OpenCV, or pass -Cli)" }
    $smoke = "checkpoints\_smoke\${Name}_smoke"
    if (Test-Path $smoke) { Remove-Item $smoke -Recurse -Force }
    $a = (TrainArgs $smoke 200)
    $a = @($a | ForEach-Object { $_ })
    $i = [array]::IndexOf($a, "--bench-val-frames"); $a[$i + 1] = "4"
    $a += @("--eval-every", "100", "--save-every", "200")
    if ((Run $a "${Name}_smoke_train.log") -ne 0) { Fail "smoke training failed (log: reports\logs\${Name}_smoke_train.log)" }
    $best = Join-Path $smoke "best.pt"
    if (-not (Test-Path $best)) { Fail "smoke run wrote no best.pt" }
    & $py -c "import torch,sys,json; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['config']; assert c['baseline_source'] and float(c['crushed_hold'])>0 and c['best_metric']=='bench_gain', c; assert float(c['corpus_ev'])==0.0, c['corpus_ev']; ev=[d for d in map(json.loads, open(sys.argv[2])) if 'eval' in d]; m=ev[-1]['eval']; assert m['bench_real_frames']>0 and 'bench_gain' in m, m; print('config ok; bench val at the last eval:', {k: round(v,3) for k,v in m.items() if k.startswith('bench_')})" $best (Join-Path $smoke "train.jsonl")
    if ($LASTEXITCODE -ne 0) { Fail "smoke checkpoint config or bench validation is wrong" }
    Write-Host "`nSMOKE PASSED. Next: -Stage train" -ForegroundColor Green
}

if ($Stage -eq "train") {
    Stage "TRAIN  $Name, $Steps steps from $Init on the picker's baseline, crushed hold $CrushedHold, selected on bench_gain"
    $final = Join-Path $Repo ("$Out\step_{0:D7}.pt" -f $Steps)
    if (Test-Path $final) { Write-Host "already trained" -ForegroundColor DarkGray }
    else {
        $a = TrainArgs $Out $Steps
        $last = Get-ChildItem (Join-Path $Repo $Out) -Filter "step_*.pt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
        if ($last) {
            $a = @($a | Where-Object { $_ -ne "--init-checkpoint" -and $_ -ne $Init })
            $a += @("--resume", $last.FullName); Write-Host "resuming from $($last.Name)" -ForegroundColor Yellow
        }
        if ((Run $a "${Name}_train.log") -ne 0) { Fail "training failed (log: reports\logs\${Name}_train.log)" }
    }
    & $py -c "import torch,sys; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print('best.pt is step', p['step'], 'score', p['best'])" (Join-Path $Out "best.pt")
    Write-Host "`nTRAINED. Next: -Stage bench (or -Stage bench -ClipOnly first)" -ForegroundColor Green
}

if ($Stage -eq "bench") {
    Stage "BENCH  $Name on real / aces / oog / mix against the picker's baseline, and inside the clip at +0, +1, +2 EV against the clip fill"
    $ckpt = if ($Checkpoint) { $Checkpoint } else { Join-Path $Out "best.pt" }
    if (-not (Test-Path $ckpt)) { Fail "no $ckpt; run -Stage train first" }
    & $py -c "import torch,sys; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); c=p.get('config',{}); assert c.get('baseline_source'), 'not a v10 checkpoint'; print('benching step', p.get('step'))" $ckpt
    if ($LASTEXITCODE -ne 0) { Fail "$ckpt is not a v10 checkpoint (baseline_source)" }
    if (-not $Checkpoint -and (& $py -c "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['step'])" $ckpt) -eq "0") {
        Fail "best.pt is step 0 (the init): selection never moved; bench a step with -Checkpoint and look at the bench_* rows in train.jsonl"
    }
    $freeGB = [math]::Round((Get-PSDrive -Name ($Repo.Substring(0, 1))).Free / 1GB, 1)
    if ($freeGB -lt 15) { Fail "only $freeGB GB free on $($Repo.Substring(0, 2)); need 15 GB" }
    # name -> manifest, condition, the picker entry the model rides on, the picker tree it is judged against
    $benches = [ordered]@{
        "cp_real"      = @{ manifest = $Netflix; condition = "clean"; curve = "bt1886"; picker = "bt1886"; extra = @() }
        "cp_aces"      = @{ manifest = (Join-Path $v4bRoot "sdr_hdr_manifest.jsonl"); condition = "clean"; curve = "aces"; picker = $null; extra = @() }
        "cp_oog"       = @{ manifest = $HoldOut; condition = "out-of-generator"; curve = "bt1886"; picker = "bt1886"; extra = @() }
        "cp_mix"       = @{ manifest = $V4c; condition = "clean"; curve = "bt1886"; picker = "bt1886"; extra = @() }
        "cp8_clip_ev0" = @{ manifest = $Netflix; condition = "exposure"; curve = "aces"; picker = $null; extra = @("--exposure-ev", "0", "--exposure-curve", "aces") }
        "cp8_clip_ev1" = @{ manifest = $Netflix; condition = "exposure"; curve = "aces"; picker = $null; extra = @("--exposure-ev", "1", "--exposure-curve", "aces") }
        "cp8_clip_ev2" = @{ manifest = $Netflix; condition = "exposure"; curve = "aces"; picker = $null; extra = @("--exposure-ev", "2", "--exposure-curve", "aces") }
    }
    foreach ($b in @($benches.Keys)) {
        if ($ClipOnly -and -not $b.StartsWith("cp8_clip")) { continue }
        $root = Join-Path $Repo "bench\$b"; $res = Join-Path $root "results"
        New-Item -ItemType Directory -Path $res -Force | Out-Null
        $clip = $b.StartsWith("cp8_clip")
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
        $pickerTree = if ($benches[$b].picker) { "baseline_$($benches[$b].picker)" } else { $null }
        if (-not (Test-Path $done)) {
            $a = @("training\export_bench_pairs.py", "--checkpoint", $ckpt, "--manifest", $benches[$b].manifest,
                   "--out", $root, "--split", "test", "--condition", $benches[$b].condition,
                   "--test-name", $Name, "--baseline-curve", $benches[$b].curve) + $benches[$b].extra
            if ($pickerTree) { $a += @("--picker-baseline", $benches[$b].picker) }
            if (Test-Path $refDir) { $a += "--only-test" } elseif ($clip) { $a += "--write-sdr" }
            if ((Run $a "${Name}_export_$b.log") -ne 0) { Fail "export $b failed" }
        }
        $trees = @($Name) + $(if ($pickerTree) { @($pickerTree) } else { @("baseline") })
        if ($clip -or $b -eq "cp_real") {
            foreach ($t in $trees + @("baseline")) {
                if (-not (Test-Path (Join-Path $res "$t.regions.csv"))) {
                    if ((Run @("training\score_regions.py", $root, "--test-dir", $t) "${Name}_regions_${b}_$t.log") -ne 0) { Fail "region scoring $b/$t failed" }
                }
            }
        }
        if (-not $clip) {
            foreach ($t in $trees + @("baseline") | Sort-Object -Unique) {
                $json = Join-Path $res "$t.json"
                if (-not (Test-Path $json)) {
                    if ((Run @("-m", "rudra.delivery.cli", "bench", $root, "--nits-scale", "203", "--test-dir", $t, "--output", $json) "${Name}_bench_${b}_$t.log") -ne 0) { Fail "bench $b/$t failed" }
                }
            }
        }
    }
    # The clip fill's trees (RUDRA 1.0's reconstruction in clipped highlights): the clip rows are judged against them.
    $cliExe = FindCli
    foreach ($ev in 0, 1, 2) {
        $root = Join-Path $Repo "bench\cp8_clip_ev$ev"; $res = Join-Path $root "results"
        & $cliExe bench-clip-fill $root --corpus-ev 0 2>&1 | Tee-Object -FilePath (Join-Path $logDir "${Name}_clipfill_ev$ev.log") | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail "bench-clip-fill ev$ev failed" }
        if (-not (Test-Path (Join-Path $res "clipfill.regions.csv"))) {
            if ((Run @("training\score_regions.py", $root, "--test-dir", "clipfill") "${Name}_regions_clipfill_ev$ev.log") -ne 0) { Fail "region scoring clipfill ev$ev failed" }
        }
    }
    Run @("training\cp10_verdicts.py", "--bench-root", (Join-Path $Repo "bench"), "--candidate", $Name, "--out", (Join-Path $logDir "cp10_results_$Name.json")) "${Name}_verdicts.log" | Out-Null
    Write-Host "`nverdicts -> reports\logs\${Name}_verdicts.log, JSON -> reports\logs\cp10_results_$Name.json" -ForegroundColor Green
}
Stop-Transcript | Out-Null
