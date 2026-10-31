<#
.SYNOPSIS
  shadow_v1 on real SDR (Netflix Meridian, 287 test frames), the bench it was
  never run on. Reuses bench\cp_real's ref/ and baseline/ (--only-test), so only
  the shadow_v1 tree is exported. Then PU21/CVVDP, the clipped/crushed region
  scores, and paired CIs against the inverse and against v8.
  Run from a PowerShell with `conda activate comfy`. About 20 minutes.
#>
$ErrorActionPreference = "Continue"
$Repo = "D:\A.I\Devlopments\rudra"
Set-Location -LiteralPath $Repo
$env:OPENCV_IO_ENABLE_OPENEXR = "1"
$logDir = Join-Path $Repo "reports\logs"; New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$py = "python"
$Name = "shadow_v1"
$Ckpt = "checkpoints\sdr2hdr_shadow_v1.pt"
$Netflix = "G:\datasets\corpora\rudra_netflix_realsdr_20261001\manifest.jsonl"
$Root = Join-Path $Repo "bench\cp_real"; $Res = Join-Path $Root "results"

function Fail([string]$why) { Write-Host "`nSTOPPED: $why" -ForegroundColor Red; exit 1 }
function Run([string[]]$argv, [string]$log) {
    & $py @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir $log) | Out-Host
    return $LASTEXITCODE
}
foreach ($p in @($Ckpt, $Netflix, (Join-Path $Root "ref"), (Join-Path $Res "baseline.csv"), (Join-Path $Res "v8.csv"))) {
    if (-not (Test-Path $p)) { Fail "missing: $p" }
}
$nRef = @(Get-ChildItem (Join-Path $Root "ref") -Recurse -File).Count
if ($nRef -ne 287) { Fail "bench\cp_real\ref has $nRef frames, expected 287" }

$done = Join-Path $Root "export_$Name.json"
if ((Test-Path (Join-Path $Root $Name)) -and -not (Test-Path $done)) {
    Write-Host "partial export: re-exporting" -ForegroundColor Yellow
    Remove-Item (Join-Path $Root $Name) -Recurse -Force
    Remove-Item (Join-Path $Res "$Name.*") -Force -ErrorAction SilentlyContinue
}
if (-not (Test-Path $done)) {
    $a = @("training\export_bench_pairs.py", "--checkpoint", $Ckpt, "--manifest", $Netflix, "--out", $Root,
           "--split", "test", "--condition", "clean", "--test-name", $Name, "--only-test")
    if ((Run $a "shadow_v1_real_export.log") -ne 0) { Fail "export failed" }
}
$json = Join-Path $Res "$Name.json"
if (-not (Test-Path $json)) {
    if ((Run @("-m", "rudra.delivery.cli", "bench", $Root, "--nits-scale", "203", "--test-dir", $Name, "--output", $json) "shadow_v1_real_bench.log") -ne 0) { Fail "bench failed" }
}
if ((Run @("training\score_regions.py", $Root, "--test-dir", "baseline", $Name) "shadow_v1_real_regions.log") -ne 0) { Fail "region scoring failed" }

Write-Host "`n== shadow_v1 vs the inverse (positive = shadow_v1 better)" -ForegroundColor Cyan
Run @("training\paired_gate.py", "--a", (Join-Path $Res "$Name.csv"), "--b", (Join-Path $Res "baseline.csv"),
      "--out", (Join-Path $logDir "shadow_v1_real_vs_baseline.json")) "shadow_v1_real_vs_baseline.log" | Out-Null
Write-Host "`n== v8 vs shadow_v1 (positive = v8 better)" -ForegroundColor Cyan
Run @("training\paired_gate.py", "--a", (Join-Path $Res "v8.csv"), "--b", (Join-Path $Res "$Name.csv"),
      "--out", (Join-Path $logDir "v8_vs_shadow_v1_real.json")) "v8_vs_shadow_v1_real.log" | Out-Null
Write-Host "`nDONE. Paste the two blocks above." -ForegroundColor Green
