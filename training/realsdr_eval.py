"""RUDRA on real SDR, full frame, against its analytic baseline; plus an exposure diagnostic.

Review points W1 and W6 (1 Oct 2026). Everything the paper scores against the analytic
baseline was rendered by our own pipeline. This scores the shipped model, RUDRA-base and
the baseline on real SDR the pipeline never touched, as prepared 1280x720 pairs:

  hdrtv1k_test   HDRTV1K published test split, YouTube SDR (117 frames)
  hdrtv1k_val    HDRTV1K val videos blba / milano / redbull (201 frames)
  netflix        Sol Levante + Sparks, professional SDR grades (78 frames)

and asks the question the real-SDR correction runs (R1-R5) could not: is the model's
shadow error on real SDR a global exposure offset? Per frame it measures the median
log2(target / prediction) luminance ratio separately on interior pixels (SDR luma code
0.15-0.85, the exposure), on target shadows (< 12 nits) and on target highlights
(>= 203 nits). If the shadow ratio equals the interior ratio, aligning exposure removes
it; if the shadows sit apart from the interior, there is a shadow-specific error.
Exposure alignment uses one gain per source fitted on the EVEN frames (by asset id) and
is scored on the ODD frames only.

    python -m training.realsdr_eval --manifest <upgrade manifest> --splits val test \
        --manifest <netflix manifest> --splits val --out <dir> [--cvvdp]

Metrics: PU21-PSNR (the paper's), the PU21 region errors of the R1-R5 runs, Delta E ITP
(BT.2124) and, with --cvvdp, ColorVideoVDP on the paper's display (``jod_paper``,
standard_hdr_linear: 1,500-nit peak, BT.709 primaries) and on a 10,000-nit BT.2020
display (``jod_10k``). No model is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LUMA2020 = np.array([0.2627, 0.6780, 0.0593])
INTERIOR = (0.15, 0.85)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remap(path: str, roots: dict[str, str]) -> str:
    """Manifest paths are Windows paths; map their corpus folder to a local root."""
    p = path.replace("\\", "/")
    for name, local in roots.items():
        if name in p:
            return str(Path(local) / p.split(name, 1)[1].lstrip("/"))
    return path


def label_of(record: dict) -> str:
    a = record["asset_id"]
    if a.startswith("hdrtv1k_"):
        return "hdrtv1k_" + record["split"]
    if a.startswith("netflix_"):
        return "netflix"
    return record["split"]


def ratios(pred_nits, ref_nits, sdr_code, ceiling_nits):
    yp, yt = pred_nits @ LUMA2020, ref_nits @ LUMA2020
    code = sdr_code @ LUMA2020
    valid = (yp > 0.005) & (yt > 0.005) & ~(yt >= ceiling_nits * (1 - 1e-3))
    out = {}
    for name, m in (("interior", (code >= INTERIOR[0]) & (code <= INTERIOR[1])),
                    ("shadow", yt < 12.0), ("mid", (yt >= 12.0) & (yt < 203.0)), ("high", yt >= 203.0)):
        m = m & valid
        out[name] = float(np.median(np.log2(yt[m] / yp[m]))) if m.sum() >= 64 else None
        out[name + "_px"] = int(m.sum())
    return out


def region_errors(pred_nits, ref_nits, ceiling_nits):
    """The R1-R5 PU21 region errors (training/robust_reconstruction.region_pu_risk), numpy."""
    from rudra.delivery.bench import pu21_encode
    ep, et = pu21_encode(pred_nits) / 100.0, pu21_encode(ref_nits) / 100.0   # same units as the runs
    err = (ep - et) ** 2
    cens = ref_nits >= ceiling_nits * (1 - 1e-3)
    allowance = pu21_encode(np.array(min(ceiling_nits * 8, 1e6))) / 100.0
    bounded = np.maximum(et - ep, 0) ** 2 + np.maximum(ep - allowance, 0) ** 2
    err = np.where(cens, bounded, err).mean(-1)
    y = ref_nits @ LUMA2020
    out = {}
    for name, m in (("shadow", y < 12), ("mid", (y >= 12) & (y < 203)), ("high", y >= 203)):
        out["pu_" + name] = float(err[m].mean()) if m.any() else None
    return out


def make_cvvdp(device):
    from training.cvvdp_display_check import make_metrics
    return make_metrics(device, {"jod_paper": None, "jod_10k": (10000.0, "BT.2020-linear")})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--splits", action="append", nargs="+", required=True)
    p.add_argument("--root", action="append", default=[], help="NAME=LOCALDIR to remap manifest paths")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="checkpoints/sdr2hdr_shadow_v1.pt")
    p.add_argument("--base-checkpoint", default="checkpoints/sdr2hdr_image_v5.pt")
    p.add_argument("--cvvdp", action="store_true")
    p.add_argument("--cvvdp-sources", nargs="*", default=None,
                   help="limit CVVDP to these sources (e.g. hdrtv1k_test netflix); default all")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    import torch
    from training.bench_hdrtv1k import delta_e_itp, pu21_psnr
    from training.sdr2hdr_dataset import load_rgb, read_jsonl
    from training.train_sdr2hdr import load_image_checkpoint

    roots = dict(r.split("=", 1) for r in args.root)
    device = torch.device(args.device)
    rows = []
    for manifest, splits in zip(args.manifest, args.splits):
        for r in read_jsonl(manifest):
            if r["split"] in splits and not r.get("synthetic_sdr"):
                rows.append(r | {"_manifest": manifest})
    rows = sorted(rows, key=lambda r: r["asset_id"])[: args.limit or None]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "frames.json"
    frames = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    models = {}
    for name, ck in (("rudra", args.checkpoint), ("rudra_base", args.base_checkpoint)):
        m, _ = load_image_checkpoint(Path(ck), device)
        models[name] = m.eval()
    cv = make_cvvdp(args.device) if args.cvvdp else None

    for i, r in enumerate(rows):
        a = r["asset_id"]
        want_cv = bool(cv) and (args.cvvdp_sources is None or label_of(r) in args.cvvdp_sources)
        if a in frames and (not want_cv or "jod_10k" in frames[a]["rudra"]):
            continue
        sdr = load_rgb(remap(r["sdr_path"], roots), hdr=False)
        ref = load_rgb(remap(r["hdr_path"], roots), hdr=True, ceiling=1e6).astype(np.float64) * 10000.0
        ceiling = float(r.get("ceiling_nits") or 1e12)
        x = torch.from_numpy(np.ascontiguousarray(sdr)).permute(2, 0, 1)[None].float().to(device)
        preds = {}
        with torch.inference_mode():
            for name, m in models.items():
                o = m(x, preserve_outside=True)
                preds[name] = o.hdr[0].permute(1, 2, 0).double().cpu().numpy() * 10000.0
                if name == "rudra":
                    preds["baseline"] = o.baseline[0].permute(1, 2, 0).double().cpu().numpy() * 10000.0
                    w = m.predict_shadow_weight(x)
        rec = dict(source=label_of(r), split=r["split"], shadow_weight=None if w is None else float(w.item()),
                   ref_peak=float((ref @ LUMA2020).max()))
        for name, pred in preds.items():
            s = dict(pu21_psnr=pu21_psnr(pred, ref), de_itp=delta_e_itp(pred, ref))
            s.update(region_errors(pred, ref, ceiling))
            s.update({"k_" + k: v for k, v in ratios(pred, ref, sdr, ceiling).items()})
            if want_cv:
                from training.cvvdp_display_check import jod
                for n, metric in cv.items():
                    s[n] = jod(metric, pred.astype(np.float32), ref.astype(np.float32), args.device)
            rec[name] = s
        frames[a] = rec
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            cache_path.write_text(json.dumps(frames), encoding="utf-8")
            print(f"{i + 1}/{len(rows)} {a}", flush=True)
    cache_path.write_text(json.dumps(frames), encoding="utf-8")
    summary = summarize(frames, args)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


def summarize(frames: dict, args) -> dict:
    from training.bench_hdrtv1k import bootstrap_ci
    sources = sorted({f["source"] for f in frames.values()})
    metrics = [m for m in ("pu21_psnr", "jod_paper", "jod_10k", "de_itp", "pu_shadow", "pu_mid", "pu_high")
               if any(m in f["rudra"] for f in frames.values())]
    better_high = {"pu21_psnr": True, "jod_paper": True, "jod_10k": True}
    out = dict(gains={}, absolute={}, exposure={}, alignment={}, metrics=metrics,
               note="gains are method minus baseline, sign flipped where lower is better; positive = better; "
                    "95% CI frame bootstrap (no scene labels inside a source)")
    for src in sources + ["all"]:
        keys = sorted(a for a, f in frames.items() if src == "all" or f["source"] == src)
        out["gains"][src], out["absolute"][src] = {}, {}
        for meth in ("rudra", "rudra_base", "baseline"):
            out["absolute"][src][meth] = {m: float(np.nanmean([np.nan if frames[a][meth].get(m) is None
                                                                 else frames[a][meth][m] for a in keys] or [np.nan]))
                                          for m in metrics}
        for meth in ("rudra", "rudra_base"):
            out["gains"][src][meth] = {}
            for m in metrics:
                g = []
                for a in keys:
                    t, b = frames[a][meth].get(m), frames[a]["baseline"].get(m)
                    if t is None or b is None:
                        continue
                    g.append((t - b) if better_high.get(m, False) else (b - t))
                if not g:
                    continue
                ci = bootstrap_ci(np.array(g))
                ci["better"] = int(sum(x > 0 for x in g))
                out["gains"][src][meth][m] = ci
        # exposure: median offsets of RUDRA per region, and the shadow-minus-interior gap
        e = {}
        for reg in ("interior", "shadow", "mid", "high"):
            v = [frames[a]["rudra"]["k_" + reg] for a in keys if frames[a]["rudra"]["k_" + reg] is not None]
            e[reg] = dict(median=float(np.median(v)) if v else None,
                          iqr=[float(np.quantile(v, .25)), float(np.quantile(v, .75))] if v else None, n=len(v))
        gap = [frames[a]["rudra"]["k_shadow"] - frames[a]["rudra"]["k_interior"] for a in keys
               if frames[a]["rudra"]["k_shadow"] is not None and frames[a]["rudra"]["k_interior"] is not None]
        e["shadow_minus_interior"] = dict(bootstrap_ci(np.array(gap)) if gap else {},
                                          positive=int(sum(x > 0 for x in gap)), n=len(gap))
        out["exposure"][src] = e
        if src != "all":
            fit, test = keys[0::2], keys[1::2]
            ks = [frames[a]["rudra"]["k_interior"] for a in fit if frames[a]["rudra"]["k_interior"] is not None]
            k = float(np.median(ks))
            rem = [frames[a]["rudra"]["k_shadow"] - k for a in test if frames[a]["rudra"]["k_shadow"] is not None]
            before = [frames[a]["rudra"]["k_shadow"] for a in test if frames[a]["rudra"]["k_shadow"] is not None]
            out["alignment"][src] = dict(k_source=k, fit_frames=len(ks), test_frames=len(test),
                                         shadow_offset_before=bootstrap_ci(np.array(before)) if before else None,
                                         shadow_offset_after=bootstrap_ci(np.array(rem)) if rem else None)
    return out


def markdown(s) -> str:
    L = ["# RUDRA on real SDR (full frame, 1280x720), against the analytic baseline", "", s["note"], ""]
    for src, rows in s["gains"].items():
        L += [f"## {src}", "", "| method | " + " | ".join(s["metrics"]) + " |", "|---|" + "---|" * len(s["metrics"])]
        for meth, r in rows.items():
            L.append(f"| {meth} | " + " | ".join(
                (f"{r[m]['mean']:+.4f} [{r[m]['lo']:+.4f}, {r[m]['hi']:+.4f}] {r[m]['better']}/{r[m]['n']}" if m in r else "n/a")
                for m in s["metrics"]) + " |")
        e = s["exposure"][src]
        L += ["", "RUDRA exposure, median log2(target/prediction) in stops: " + ", ".join(
            f"{reg} {e[reg]['median']:+.3f}" for reg in ("interior", "shadow", "mid", "high") if e[reg]["median"] is not None)]
        g = e["shadow_minus_interior"]
        if g:
            L.append(f"Shadow minus interior: {g['mean']:+.3f} stops [{g['lo']:+.3f}, {g['hi']:+.3f}], positive on {g['positive']}/{g['n']} frames")
        if src in s["alignment"]:
            a = s["alignment"][src]
            if a["shadow_offset_after"]:
                L.append(f"Per-source exposure k = {a['k_source']:+.3f} stops (even frames); odd-frame shadow offset "
                         f"{a['shadow_offset_before']['mean']:+.3f} before, {a['shadow_offset_after']['mean']:+.3f} "
                         f"[{a['shadow_offset_after']['lo']:+.3f}, {a['shadow_offset_after']['hi']:+.3f}] after alignment")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
