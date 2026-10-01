"""HDRTV1K standard benchmark: RUDRA, its analytic baseline and published methods.

Answers review point W1/W2 (1 Oct 2026): every result in the paper is on SDR our own
pipeline rendered. This scores RUDRA on the published HDRTV1K test split (117 frames,
3840x2160, YouTube SDR -> HDR10 master) with the metrics that literature reports, next
to the analytic baseline and any published method whose outputs are dropped in.

    # 1. RUDRA, RUDRA-base and the analytic baseline (writes 16-bit PQ PNGs)
    python -m training.bench_hdrtv1k infer --sdr-dir <test_sdr> --out <OUT>

    # 2. a published method with weights in its repo (HDRTVDM, Guo et al. 2023)
    python -m training.bench_hdrtv1k hdrtvdm --repo <HDRTVDM clone> --sdr-dir <test_sdr> --out <OUT>

    # 3. any other method: its 16-bit outputs, one per test frame, same stem
    python -m training.bench_hdrtv1k import --name hdrtvnet --src <dir> --out <OUT>

    # 4. score everything under <OUT>/methods, with bootstrap confidence intervals
    python -m training.bench_hdrtv1k score --gt-dir <test_hdr> --out <OUT> [--cvvdp]

File convention. HDRTV1K's 16-bit HDR PNGs place PQ white at 876 * 74.5234 = 65282.5,
not 65535 (paper, appendix B). Methods trained on those files reproduce that convention,
so RUDRA's outputs are written in it too, and every file is read the same way:

  * PSNR and SSIM are computed on code / 65535, exactly as the literature does, so the
    numbers are comparable with published tables.
  * Delta E ITP (ITU-R BT.2124), PU21-PSNR and CVVDP decode code / 65282.5 through
    ST 2084 to absolute cd/m2, which is what the code values actually mean.

HDR-VDP-3 is MATLAB-only and is not computed. Frames are treated as independent in the
bootstrap unless --clusters maps each stem to a scene; HDRTV1K does not publish one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PQ_WHITE_CODE = 876 * 74.52344347071411   # 65282.5, measured; see the paper's appendix B
U16 = 65535.0

# BT.2124 / BT.2100 ICtCp
_LMS = np.array([[1688, 2146, 262], [683, 2951, 462], [99, 309, 3688]], dtype=np.float64) / 4096.0
_ICTCP = np.array([[2048, 2048, 0], [6610, -13613, 7003], [17933, -17390, -543]], dtype=np.float64) / 4096.0

# Metric name -> True when higher is better.
HIGHER_IS_BETTER = {"psnr_pq": True, "ssim_pq": True, "de_itp": False, "pu21_psnr": True}


# --------------------------------------------------------------------------- io

def read_u16_rgb(path: Path) -> np.ndarray:
    """16-bit RGB image as uint16 (H, W, 3). PNG or TIFF."""
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"{path}: expected 16-bit RGB, got {image.dtype} {image.shape}")
    return np.ascontiguousarray(image[..., 2::-1])


def write_u16_rgb(path: Path, rgb_u16: np.ndarray) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(rgb_u16[..., ::-1])):
        raise OSError(f"could not write {path}")


def read_sdr(path: Path) -> np.ndarray:
    """8-bit (or 16-bit) SDR PNG as float32 RGB in [0, 1]."""
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    scale = 255.0 if image.dtype == np.uint8 else 65535.0
    return np.ascontiguousarray(image[..., 2::-1]).astype(np.float32) / scale


def pq_to_code(pq: np.ndarray) -> np.ndarray:
    """Normalised PQ signal -> uint16 in the HDRTV1K file convention."""
    return np.clip(np.round(np.asarray(pq, np.float64) * PQ_WHITE_CODE), 0, U16).astype(np.uint16)


def code_to_nits(code: np.ndarray) -> np.ndarray:
    from rudra.hdr10 import pq_eotf
    return pq_eotf(np.asarray(code, np.float32) / PQ_WHITE_CODE).astype(np.float64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stems(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff"))


# ---------------------------------------------------------------------- metrics

def psnr_pq(test_u16: np.ndarray, ref_u16: np.ndarray) -> float:
    mse = float(np.mean((test_u16.astype(np.float64) / U16 - ref_u16.astype(np.float64) / U16) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim_pq(test_u16: np.ndarray, ref_u16: np.ndarray, device: str = "cpu") -> float:
    """Gaussian SSIM (11x11, sigma 1.5, K1 0.01, K2 0.03), per channel, data range 1."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(test_u16.astype(np.float32) / U16).permute(2, 0, 1)[:, None].to(device)
    y = torch.from_numpy(ref_u16.astype(np.float32) / U16).permute(2, 0, 1)[:, None].to(device)
    g = torch.exp(-((torch.arange(11, dtype=torch.float32, device=device) - 5) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :])[None, None]
    blur = lambda z: F.conv2d(z, win)  # valid convolution, as in the reference implementation
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean().item())


def ictcp(nits_rgb2020: np.ndarray) -> np.ndarray:
    from rudra.hdr10 import pq_oetf
    lms = np.maximum(nits_rgb2020, 0.0) @ _LMS.T
    return pq_oetf(lms).astype(np.float64) @ _ICTCP.T


def delta_e_itp(test_nits: np.ndarray, ref_nits: np.ndarray) -> float:
    """Mean ITU-R BT.2124 Delta E ITP over pixels."""
    a, b = ictcp(test_nits), ictcp(ref_nits)
    d = a - b
    return float(np.mean(720.0 * np.sqrt(d[..., 0] ** 2 + (0.5 * d[..., 1]) ** 2 + d[..., 2] ** 2)))


def pu21_psnr(test_nits: np.ndarray, ref_nits: np.ndarray) -> float:
    from rudra.delivery.bench import pu_psnr
    return float(pu_psnr(test_nits, ref_nits))


class CVVDP:
    """ColorVideoVDP on PQ-encoded BT.2020 frames, at one or more display peaks.

    Peak 1500 is pycvvdp's own ``standard_hdr_pq``. Any other peak keeps that display's
    geometry and ambient light and changes only the peak, because the display model
    hard-clips at its peak and a 1,500-nit display cannot see errors above it
    (training/cvvdp_display_check.py measures this).
    """

    def __init__(self, peaks: list[float], device: str, scale: float):
        import torch
        import pycvvdp
        import pycvvdp.display_model as dm
        self.torch, self.scale, self.device = torch, float(scale), torch.device(device)
        self.metrics = {}
        for peak in peaks:
            photometry = None
            if float(peak) != 1500.0:
                photometry = dm.vvdp_display_photo_eotf(
                    float(peak), contrast=1_000_000, source_colorspace="BT.2020-PQ",
                    EOTF="PQ", E_ambient=10, k_refl=0.005)
            self.metrics[f"jod_{int(peak)}"] = pycvvdp.cvvdp(
                display_name="standard_hdr_pq", display_photometry=photometry,
                quiet=True, device=self.device)
        for name in self.metrics:
            HIGHER_IS_BETTER[name] = True

    def __call__(self, test_u16: np.ndarray, ref_u16: np.ndarray) -> dict[str, float]:
        torch = self.torch
        import torch.nn.functional as F

        def prep(a):
            t = torch.from_numpy(np.clip(a.astype(np.float32) / PQ_WHITE_CODE, 0, 1)).to(self.device)
            if self.scale != 1.0:
                t = F.interpolate(t.permute(2, 0, 1)[None], scale_factor=self.scale,
                                  mode="area")[0].permute(1, 2, 0)
            return t.contiguous()
        test, ref = prep(test_u16), prep(ref_u16)
        with torch.no_grad():
            return {n: float(m.predict(test, ref, dim_order="HWC")[0]) for n, m in self.metrics.items()}


def score_pair(test_u16, ref_u16, device="cpu", cvvdp: CVVDP | None = None) -> dict[str, float]:
    if test_u16.shape != ref_u16.shape:
        raise ValueError(f"shape mismatch {test_u16.shape} vs {ref_u16.shape}")
    test_nits, ref_nits = code_to_nits(test_u16), code_to_nits(ref_u16)
    out = dict(psnr_pq=psnr_pq(test_u16, ref_u16), ssim_pq=ssim_pq(test_u16, ref_u16, device),
               de_itp=delta_e_itp(test_nits, ref_nits), pu21_psnr=pu21_psnr(test_nits, ref_nits))
    if cvvdp is not None:
        out.update(cvvdp(test_u16, ref_u16))
    return out


# -------------------------------------------------------------------- bootstrap

def bootstrap_ci(values: np.ndarray, clusters: list[str] | None = None, n: int = 10000,
                 seed: int = 20261001, alpha: float = 0.05) -> dict[str, float]:
    """Mean with a percentile bootstrap CI; resamples clusters (scenes) when given."""
    values = np.asarray(values, np.float64)
    keep = np.isfinite(values)
    values = values[keep]
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(values), size=(n, len(values)))
        means = values[idx].mean(1)
        units = len(values)
    else:
        clusters = [c for c, k in zip(clusters, keep) if k]
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(clusters):
            groups.setdefault(c, []).append(i)
        members = [np.array(v) for v in groups.values()]
        sums = np.array([values[m].sum() for m in members])
        counts = np.array([len(m) for m in members])
        pick = rng.integers(0, len(members), size=(n, len(members)))
        means = sums[pick].sum(1) / counts[pick].sum(1)
        units = len(members)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return dict(mean=float(values.mean()), lo=float(lo), hi=float(hi),
                p_positive=float((means > 0).mean()), n=int(len(values)), units=int(units))


def paired_gain(method: dict[str, float], reference: dict[str, float], metric: str) -> float:
    d = method[metric] - reference[metric]
    return d if HIGHER_IS_BETTER.get(metric, True) else -d


# --------------------------------------------------------------------- commands

def run_model(model, x, tile: int, overlap: int, shadow_weight, residual_scale):
    """(hdr, baseline) for one canonical SDR tensor, tiled like rudra.video.Predictor."""
    import torch

    def forward(t):
        o = model(t, preserve_outside=True, recovery_mode="all",
                  shadow_weight=shadow_weight, residual_scale=residual_scale)
        return torch.cat((o.hdr, o.baseline), 1)

    h, w = x.shape[-2:]
    if tile <= 0 or max(h, w) <= tile:
        y = forward(x)
    else:
        def starts(length):
            if length <= tile:
                return [0]
            v = list(range(0, length - tile + 1, tile - overlap))
            if v[-1] != length - tile:
                v.append(length - tile)
            return v
        y = torch.zeros((1, 6, h, w), device=x.device)
        acc = torch.zeros((1, 1, h, w), device=x.device)
        for top in starts(h):
            for left in starts(w):
                t = x[..., top:top + tile, left:left + tile]
                th, tw = t.shape[-2:]
                wy, wx = torch.ones(th, device=x.device), torch.ones(tw, device=x.device)
                fy, fx = min(overlap, th // 2), min(overlap, tw // 2)
                if top and fy:
                    wy[:fy] = torch.linspace(.001, 1, fy, device=x.device)
                if top + th < h and fy:
                    wy[-fy:] = torch.linspace(1, .001, fy, device=x.device)
                if left and fx:
                    wx[:fx] = torch.linspace(.001, 1, fx, device=x.device)
                if left + tw < w and fx:
                    wx[-fx:] = torch.linspace(1, .001, fx, device=x.device)
                blend = (wy[:, None] * wx[None, :])[None, None]
                y[..., top:top + th, left:left + tw] += forward(t) * blend
                acc[..., top:top + th, left:left + tw] += blend
        y = y / acc.clamp_min(1e-6)
    return y[:, :3], y[:, 3:]


def canonical_sdr(rgb: np.ndarray, device):
    """BT.709 full-range SDR -> the BT.2020 sRGB-code tensor RUDRA expects (as rudra/video.py)."""
    import torch
    from rudra.sdr2hdr import canonicalize_sdr, srgb_to_linear, linear_to_srgb
    from rudra.delivery.colorspace import rgb_to_rgb_matrix
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)
    x = canonicalize_sdr(x, "rec709", "full")
    m = torch.as_tensor(rgb_to_rgb_matrix("rec709", "rec2020"), device=device, dtype=x.dtype)
    return linear_to_srgb(torch.einsum("ij,bjhw->bihw", m, srgb_to_linear(x)))


def cmd_infer(args) -> None:
    import torch
    from rudra.hdr10 import pq_oetf, master_to_pq
    from training.train_sdr2hdr import load_image_checkpoint

    device = torch.device(args.device)
    out = Path(args.out)
    frames = stems(Path(args.sdr_dir))[:: args.every][: args.limit or None]
    if not frames:
        raise FileNotFoundError(f"no SDR frames in {args.sdr_dir}")
    runs = [("rudra", Path(args.checkpoint)), ("rudra_base", Path(args.base_checkpoint))]
    runs = [r for r in runs if r[0] in args.models]
    for name, ckpt in runs:
        model, _ = load_image_checkpoint(ckpt, device)
        model.eval()
        peaks = args.master_peak if name == "rudra" else []   # the app's HDR10 master, shipped model only
        targets = [name] + (["baseline"] if name == "rudra" else []) + [f"{name}_m{int(p)}" for p in peaks]
        log = dict(checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt), script_sha256=sha256(Path(__file__)),
                   sdr_dir=str(args.sdr_dir), tile=args.tile, overlap=args.overlap, frames=len(frames),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"), shadow_weights={})
        for i, stem in enumerate(frames):
            if all((out / "methods" / t / f"{stem}.png").exists() for t in targets) and not args.overwrite:
                continue
            src = next(p for p in Path(args.sdr_dir).iterdir() if p.stem == stem)
            with torch.inference_mode():
                x = canonical_sdr(read_sdr(src), device)
                w = model.predict_shadow_weight(x)
                scale = model.predict_residual_scale(x) if hasattr(model, "predict_residual_scale") else None
                hdr, base = run_model(model, x, args.tile, args.overlap,
                                      None if w is None else float(w.item()), scale)
            hdr = hdr[0].permute(1, 2, 0).float().cpu().numpy()
            log["shadow_weights"][stem] = None if w is None else float(w.item())
            write_u16_rgb(out / "methods" / name / f"{stem}.png", pq_to_code(pq_oetf(hdr * 10000.0)))
            if name == "rudra":
                b = base[0].permute(1, 2, 0).float().cpu().numpy()
                write_u16_rgb(out / "methods" / "baseline" / f"{stem}.png", pq_to_code(pq_oetf(b * 10000.0)))
            for p in peaks:
                pq, _ = master_to_pq(hdr, peak_nits=float(p))
                write_u16_rgb(out / "methods" / f"{name}_m{int(p)}" / f"{stem}.png", pq_to_code(pq))
            print(f"[{name}] {i + 1}/{len(frames)} {stem}", flush=True)
        (out / "methods").mkdir(parents=True, exist_ok=True)
        (out / "methods" / f"{name}.run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def collect(src: Path, dst: Path, wanted: list[str]) -> int:
    """Copy one output per wanted stem from ``src`` (any naming that contains the stem)."""
    files = [p for p in src.rglob("*") if p.suffix.lower() in (".png", ".tif", ".tiff")]
    n = 0
    for stem in wanted:
        hits = [p for p in files if p.stem == stem] or \
               [p for p in files if p.stem.startswith(stem + "_") or p.stem.startswith(stem + ".")]
        if len(hits) != 1:
            raise FileNotFoundError(f"{src}: expected one output for {stem}, found {[h.name for h in hits]}")
        rgb = read_u16_rgb(hits[0])
        write_u16_rgb(dst / f"{stem}.png", rgb)
        n += 1
    return n


def cmd_import(args) -> None:
    out = Path(args.out) / "methods" / args.name
    wanted = stems(Path(args.sdr_dir)) if args.sdr_dir else stems(Path(args.src))
    n = collect(Path(args.src), out, wanted)
    (out.parent / f"{args.name}.run.json").write_text(json.dumps(
        dict(source=str(args.src), frames=n, imported=time.strftime("%Y-%m-%d %H:%M:%S"),
             note=args.note), indent=2), encoding="utf-8")
    print(f"imported {n} frames as {args.name}")


def cmd_hdrtvdm(args) -> None:
    """Run HDRTVDM (Guo et al., CVPR 2023, MPL-2.0) with its HDRTV1K-trained params.pth."""
    repo = Path(args.repo).resolve()
    method = repo / "method"
    if not (method / "test.py").exists() or not (method / "params.pth").exists():
        raise FileNotFoundError(f"{method}: need test.py and params.pth (git clone the HDRTVDM repo)")
    sdr = sorted(p.resolve() for p in Path(args.sdr_dir).iterdir() if p.suffix.lower() == ".png")
    sdr = sdr[:: args.every][: args.limit or None]
    raw = Path(args.out).resolve() / "_raw" / "hdrtvdm"
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(sdr), 8):
        cmd = [sys.executable, "test.py", *map(str, sdr[i:i + 8]), "-out", str(raw), "-out_format", "tif"]
        print(" ".join(cmd[:2]), f"... frames {i + 1}-{min(i + 8, len(sdr))}", flush=True)
        subprocess.run(cmd, cwd=method, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    out = Path(args.out) / "methods" / "hdrtvdm"
    n = collect(raw, out, [p.stem for p in sdr])
    (out.parent / "hdrtvdm.run.json").write_text(json.dumps(
        dict(repo=str(repo), commit=commit or None, weights="method/params.pth (HDRTV1K, YouTube degradation)",
             weights_sha256=sha256(method / "params.pth"), frames=n, licence="MPL-2.0",
             note="Outputs taken as written by the authors' test.py; code/65535 read as PQ, "
                  "the same convention as the HDRTV1K files it was trained on."), indent=2), encoding="utf-8")
    print(f"HDRTVDM: {n} frames")


def load_clusters(path: str | None) -> dict[str, str] | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def cmd_score(args) -> None:
    import torch
    out = Path(args.out)
    gt_dir = Path(args.gt_dir)
    methods_dir = out / "methods"
    methods = args.methods or sorted(p.name for p in methods_dir.iterdir() if p.is_dir())
    frames = [s for s in stems(gt_dir)][:: args.every][: args.limit or None]
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    cvvdp = CVVDP(args.cvvdp_peaks, device, args.cvvdp_scale) if args.cvvdp else None
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per: dict[str, dict[str, dict[str, float]]] = {}
    for m in methods:
        cache = results_dir / f"{m}.json"
        done = json.loads(cache.read_text()) if cache.exists() and not args.overwrite else {}
        for i, stem in enumerate(frames):
            if stem in done and (not args.cvvdp or all(k in done[stem] for k in cvvdp.metrics)):
                continue
            test_path = methods_dir / m / f"{stem}.png"
            if not test_path.exists():
                raise FileNotFoundError(f"{m}: missing {test_path.name}")
            gt_path = next(p for p in gt_dir.iterdir() if p.stem == stem)
            done[stem] = score_pair(read_u16_rgb(test_path), read_u16_rgb(gt_path), device, cvvdp)
            if (i + 1) % 10 == 0:
                cache.write_text(json.dumps(done), encoding="utf-8")
                print(f"[{m}] {i + 1}/{len(frames)}", flush=True)
        cache.write_text(json.dumps(done), encoding="utf-8")
        rows = ["frame," + ",".join(sorted(next(iter(done.values())).keys()))]
        rows += [stem + "," + ",".join(f"{done[stem][k]:.6f}" for k in sorted(done[stem])) for stem in frames]
        (results_dir / f"{m}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        per[m] = {s: done[s] for s in frames}
    summary = summarize(per, frames, load_clusters(args.clusters), args.references)
    summary.update(gt_dir=str(gt_dir), frames=len(frames), every=args.every, pq_white_code=PQ_WHITE_CODE,
                   script_sha256=sha256(Path(__file__)), device=device,
                   cvvdp=None if not args.cvvdp else dict(peaks=args.cvvdp_peaks, scale=args.cvvdp_scale,
                                                          display="standard_hdr_pq geometry"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print((out / "summary.md").read_text(encoding="utf-8"))


def summarize(per, frames, clusters, references):
    metrics = sorted(next(iter(next(iter(per.values())).values())).keys())
    cl = [clusters[s] for s in frames] if clusters else None
    table = {m: {k: bootstrap_ci(np.array([per[m][s][k] for s in frames]), cl) for k in metrics} for m in per}
    paired = {}
    for ref in references:
        if ref not in per:
            continue
        paired[ref] = {}
        for m in per:
            if m == ref:
                continue
            paired[ref][m] = {}
            for k in metrics:
                g = np.array([paired_gain(per[m][s], per[ref][s], k) for s in frames])
                ci = bootstrap_ci(g, cl)
                ci["frames_better"] = int((g > 0).sum())
                paired[ref][m][k] = ci
    return dict(metrics=metrics, higher_is_better={k: HIGHER_IS_BETTER.get(k, True) for k in metrics},
                absolute=table, paired=paired,
                bootstrap="percentile, 10000 resamples, seed 20261001, "
                          + ("scene clusters" if clusters else "frames treated as independent"))


def markdown(s) -> str:
    ms = s["metrics"]
    lines = [f"# HDRTV1K test, {s.get('frames')} frames", "",
             "PSNR/SSIM on code/65535 (literature protocol); Delta E ITP, PU21 and JOD on absolute "
             "luminance (code/65282.5 through ST 2084). 95% bootstrap CI in brackets; "
             + s["bootstrap"] + ".", "",
             "| method | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
    for m, row in s["absolute"].items():
        lines.append(f"| {m} | " + " | ".join(
            f"{row[k]['mean']:.4f} [{row[k]['lo']:.4f}, {row[k]['hi']:.4f}]" for k in ms) + " |")
    for ref, rows in s["paired"].items():
        lines += ["", f"## Paired gain over {ref} (positive = better)", "",
                  "| method | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
        for m, row in rows.items():
            lines.append(f"| {m} | " + " | ".join(
                f"{row[k]['mean']:+.4f} [{row[k]['lo']:+.4f}, {row[k]['hi']:+.4f}] {row[k]['frames_better']}/{row[k]['n']}"
                for k in ms) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("infer", help="RUDRA, RUDRA-base and the analytic baseline")
    a.add_argument("--sdr-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--checkpoint", default="checkpoints/sdr2hdr_shadow_v1.pt")
    a.add_argument("--base-checkpoint", default="checkpoints/sdr2hdr_image_v5.pt")
    a.add_argument("--master-peak", type=float, nargs="*", default=[1000.0],
                   help="also write RUDRA through the app's HDR10 mastering at these peaks (empty to skip)")
    a.add_argument("--tile", type=int, default=1024)
    a.add_argument("--overlap", type=int, default=64)
    a.add_argument("--device", default="cuda")
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--every", type=int, default=1, help="every Nth frame (1 = the full published split)")
    a.add_argument("--models", nargs="+", default=["rudra", "rudra_base"], choices=["rudra", "rudra_base"])
    a.add_argument("--overwrite", action="store_true")
    a.set_defaults(func=cmd_infer)

    h = sub.add_parser("hdrtvdm", help="run HDRTVDM from a clone of its repo")
    h.add_argument("--repo", required=True)
    h.add_argument("--sdr-dir", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--limit", type=int, default=0)
    h.add_argument("--every", type=int, default=1)
    h.set_defaults(func=cmd_hdrtvdm)

    i = sub.add_parser("import", help="bring in another method's 16-bit outputs")
    i.add_argument("--name", required=True)
    i.add_argument("--src", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--sdr-dir", help="restrict to these stems")
    i.add_argument("--note", default="")
    i.set_defaults(func=cmd_import)

    s = sub.add_parser("score", help="score every method against the HDR ground truth")
    s.add_argument("--gt-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--methods", nargs="*")
    s.add_argument("--references", nargs="*", default=["baseline", "rudra"])
    s.add_argument("--clusters", help="JSON {stem: scene} for a scene-level bootstrap")
    s.add_argument("--cvvdp", action="store_true")
    s.add_argument("--cvvdp-peaks", type=float, nargs="+", default=[1500.0, 10000.0])
    s.add_argument("--cvvdp-scale", type=float, default=0.5, help="downscale before CVVDP (4K is slow)")
    s.add_argument("--device", default="cuda")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
