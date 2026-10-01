# RUDRA -- the three review experiments that need no training of the model (1 Oct 2026).
#   pwsh -File D:\A.I\Devlopments\rudra\scripts\review_experiments_2026_10_01.ps1
#   (or double-click scripts\RUN_REVIEW_EXPERIMENTS.bat)
#
#   1. W5  scene-bootstrap confidence intervals on the paper's 429-frame benchmark
#   2. W4  the CVVDP display-model check: describe, synthetic probe, re-score the benchmark
#   3. W1  HDRTV1K standard test: RUDRA, RUDRA-base, the analytic baseline, the app's
#          1,000-nit HDR10 master, and HDRTVDM (CVPR 2023, weights in its repo); HDRTVNet
#          too if its outputs are present
#   4. W6  H7, the per-source exposure control on real SDR (two 64-step corrections)
#
# Everything is resumable: a finished step is skipped. Nothing is committed or promoted.
# Disk: the HDRTV1K step writes ~117 4K 16-bit PNGs per method, about 20 GB in total.
param(
    [string]$Repo      = "D:\A.I\Devlopments\rudra",
    [string]$Bench     = "E:\RUDRA_v3_20260822\bench",
    [string]$HdrTv1k   = "G:\datasets\sources\hdrtv1k\test_set",
    [string]$TvOut     = "G:\datasets\bench\hdrtv1k_20261001",
    [string]$ExtDir    = "G:\datasets\tools",
    [string]$Upgrade   = "G:\datasets\corpora\rudra_upgrade_20260930\manifest.jsonl",
    [string]$Netflix   = "G:\datasets\corpora\rudra_netflix_eval_20261001\manifest.jsonl",
    [string]$HdrtvnetOutputs = "",          # folder of HDRTVNet's 16-bit test outputs, if you have them
    [switch]$SkipHdrtv1k,
    [switch]$SkipH7
)
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $Repo
$logDir = Join-Path $Repo "reports\logs"; New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$py = if (Test-Path ".venv\Scripts\python.exe") { (Resolve-Path ".venv\Scripts\python.exe").Path } else { "python" }
$env:OPENCV_IO_ENABLE_OPENEXR = "1"
function Run([string[]]$argv, [string]$log) {
    & $py @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir $log) | Out-Host
    return $LASTEXITCODE
}
function Say([string]$t, [string]$c = "Cyan") { Write-Host "`n== $t" -ForegroundColor $c }
function Warn([string]$t) { Write-Host "   SKIPPED: $t" -ForegroundColor Yellow }

& $py -c "import pycvvdp, cv2" 2>$null
if ($LASTEXITCODE -ne 0) { Say "installing cvvdp + opencv into the venv" Yellow; & $py -m pip install cvvdp opencv-python | Out-Host }

# ---------------------------------------------------------------- 1. W5
Say "1/4  scene bootstrap on the 429-frame benchmark"
if (Test-Path (Join-Path $Bench "results\clean_baseline.json")) {
    Run @("-m", "training.bench_bootstrap", "--bench", $Bench) "review_bootstrap.log" | Out-Null
} else { Warn "no $Bench\results\clean_baseline.json" }

# ---------------------------------------------------------------- 2. W4
Say "2/4  CVVDP display model"
Run @("-m", "training.cvvdp_display_check", "describe", "--out", (Join-Path $logDir "review_cvvdp_describe.json")) "review_cvvdp_describe.log" | Out-Null
Run @("-m", "training.cvvdp_display_check", "probe", "--out", (Join-Path $logDir "review_cvvdp_probe.md")) "review_cvvdp_probe.log" | Out-Null
$methods = @("v5", "v5_noshadow", "v6", "shadow_v1") | Where-Object { Test-Path (Join-Path $Bench "clean\$_") }
if ($methods -and (Test-Path (Join-Path $Bench "clean\ref"))) {
    Run (@("-m", "training.cvvdp_display_check", "rescore", "--bench", $Bench, "--methods") + $methods) "review_cvvdp_rescore.log" | Out-Null
} else { Warn "no exported trees under $Bench\clean" }

# ---------------------------------------------------------------- 3. W1
if (-not $SkipHdrtv1k) {
    Say "3/4  HDRTV1K standard test"
    $sdr = Join-Path $HdrTv1k "test_sdr"; $hdr = Join-Path $HdrTv1k "test_hdr"
    if (-not (Test-Path $sdr) -or -not (Test-Path $hdr)) { Warn "HDRTV1K test set not found at $HdrTv1k" }
    else {
        New-Item -ItemType Directory -Path $TvOut -Force | Out-Null
        Run @("-m", "training.bench_hdrtv1k", "infer", "--sdr-dir", $sdr, "--out", $TvOut) "review_hdrtv1k_infer.log" | Out-Null

        $tvdm = Join-Path $ExtDir "HDRTVDM"
        if (-not (Test-Path (Join-Path $tvdm "method\params.pth"))) {
            New-Item -ItemType Directory -Path $ExtDir -Force | Out-Null
            git clone https://github.com/AndreGuo/HDRTVDM $tvdm 2>&1 | Out-Host
        }
        & $py -c "import imageio" 2>$null
        if ($LASTEXITCODE -ne 0) { & $py -m pip install imageio | Out-Host }
        if (-not (Test-Path (Join-Path $TvOut "methods\hdrtvdm.run.json"))) {
            if (Test-Path (Join-Path $tvdm "method\params.pth")) {
                Run @("-m", "training.bench_hdrtv1k", "hdrtvdm", "--repo", $tvdm, "--sdr-dir", $sdr, "--out", $TvOut) "review_hdrtv1k_hdrtvdm.log" | Out-Null
            } else { Warn "HDRTVDM clone failed; run without it" }
        }
        if ($HdrtvnetOutputs -and (Test-Path $HdrtvnetOutputs) -and -not (Test-Path (Join-Path $TvOut "methods\hdrtvnet.run.json"))) {
            Run @("-m", "training.bench_hdrtv1k", "import", "--name", "hdrtvnet", "--src", $HdrtvnetOutputs, "--sdr-dir", $sdr,
                  "--out", $TvOut, "--note", "HDRTVNet (Chen et al. ICCV 2021), authors' AGCM+LE+HG test outputs") "review_hdrtv1k_hdrtvnet.log" | Out-Null
        }
        Run @("-m", "training.bench_hdrtv1k", "score", "--gt-dir", $hdr, "--out", $TvOut, "--cvvdp") "review_hdrtv1k_score.log" | Out-Null
    }
}

# ---------------------------------------------------------------- 4. W6 / H7
if (-not $SkipH7) {
    Say "4/4  H7 exposure control on real SDR"
    $proto = @("$Repo\outputs\correction_transfer_20261001_031005\protocol.json",
               "$env:USERPROFILE\.codex\worktrees\e1ef\rudra\outputs\correction_transfer_20261001_031005\protocol.json") |
             Where-Object { Test-Path $_ } | Select-Object -First 1
    $existing = Get-ChildItem -Path (Join-Path $Repo "outputs") -Directory -Filter "exposure_control_*" -ErrorAction SilentlyContinue
    if ($existing) { Warn "already run: $($existing[-1].FullName)" }
    elseif (-not $proto) { Warn "R2 protocol.json (correction_transfer_20261001_031005) not found" }
    elseif (-not (Test-Path $Upgrade) -or -not (Test-Path $Netflix)) { Warn "real-SDR manifests not found" }
    else {
        $out = Join-Path $Repo ("outputs\exposure_control_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
        Run @("-m", "training.audit_exposure_control", "--manifest", $Upgrade, "--eval", $Upgrade, $Netflix,
              "--labels", "hdrtv1k_val", "netflix", "--source-protocol", $proto, "--out", $out) "review_h7.log" | Out-Null
    }
}

Write-Host @"

Done. Read, in this order:
  reports\logs\review_cvvdp_probe.md           does a 2x error above 1,500 nits score 10 JOD?
  $Bench\results\display_check\summary.md      does the clean JOD loss grow on a 10,000-nit display?
  $Bench\results\bootstrap.md                  scene-level CIs for every headline gain
  $TvOut\summary.md                            RUDRA vs baseline vs HDRTVDM on HDRTV1K
  reports\logs\review_h7.log                   H7 verdict: EXPOSURE / SURVIVES / PARTIAL
"@ -ForegroundColor Green
