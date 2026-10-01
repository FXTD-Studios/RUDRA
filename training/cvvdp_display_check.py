"""Does the metric disagreement survive a display that can show the errors?

Review point W4 (1 Oct 2026). The paper reports that RUDRA-base loses 3.0 dB PU21-PSNR on
clean input for a perceptually invisible -0.046 JOD. Two properties of how JOD was
computed could produce "invisible" on their own:

  1. Display peak. rudra/hdrvdp.py scores with pycvvdp's ``standard_hdr_linear`` display,
     whose photometric model hard-clips at Y_peak = 1500 cd/m2. Any error above 1,500
     nits is invisible to it by construction, and the reference reaches ~10^6 nits.
  2. Primaries. ``standard_hdr_linear`` is defined with BT.709 primaries, while
     rudra/hdrvdp.py hands it Rec.2020-linear RGB. Chroma differences are therefore
     judged in the wrong colour space (identically for test and reference).

PU21 has the mirror problem: it is defined on 0.005 to 10,000 cd/m2 and the harness clamps
there, so errors above 10,000 nits are invisible to PU21-PSNR as well.

Three subcommands, none of which runs a model:

    python -m training.cvvdp_display_check describe     # what the display models are
    python -m training.cvvdp_display_check probe        # synthetic proof of 1 and 2
    python -m training.cvvdp_display_check rescore --bench <bench dir> [--methods v5 shadow_v1]

``rescore`` re-scores the existing benchmark exports (<bench>/<cond>/{ref,baseline,<method>})
under four JOD settings and two PU21 settings and reports each method's paired gain over
the baseline in every one, so the question "does the disagreement survive?" is answered
by one table.

  jod_paper     standard_hdr_linear, as the paper (peak 1500, BT.709 primaries)
  jod_2020      same display, BT.2020-linear primaries (fixes 2 only)
  jod_2020_4k   BT.2020-linear, peak 4000 (a current mastering display)
  jod_2020_10k  BT.2020-linear, peak 10000 (the PQ ceiling; fixes 1 and 2)
  pu21_paper    PU21-PSNR clamped to [0.005, 10000], as the paper
  pu21_in_range PU21-PSNR over pixels whose reference is inside PU21's range only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

VARIANTS = {
    # name: (Y_peak, source colour space); None = pycvvdp's own standard_hdr_linear
    "jod_paper": None,
    "jod_2020": (1500.0, "BT.2020-linear"),
    "jod_2020_4k": (4000.0, "BT.2020-linear"),
    "jod_2020_10k": (10000.0, "BT.2020-linear"),
}


def make_metrics(device, variants=VARIANTS):
    import torch
    import pycvvdp
    import pycvvdp.display_model as dm
    out = {}
    for name, spec in variants.items():
        photometry = None
        if spec is not None:
            peak, space = spec
            photometry = dm.vvdp_display_photo_eotf(peak, contrast=1_000_000, source_colorspace=space,
                                                    EOTF="linear", E_ambient=10, k_refl=0.005)
        out[name] = pycvvdp.cvvdp(display_name="standard_hdr_linear", display_photometry=photometry,
                                  quiet=True, device=torch.device(device))
    return out


def jod(metric, test_nits: np.ndarray, ref_nits: np.ndarray, device) -> float:
    import torch
    t = torch.from_numpy(np.ascontiguousarray(test_nits, dtype=np.float32)).to(device)
    r = torch.from_numpy(np.ascontiguousarray(ref_nits, dtype=np.float32)).to(device)
    with torch.no_grad():
        return float(metric.predict(t, r, dim_order="HWC")[0])


def pu21_variants(test_nits: np.ndarray, ref_nits: np.ndarray) -> dict[str, float]:
    from rudra.delivery.bench import pu21_encode, _PU21_MAX, _PU21_MIN
    peak = float(pu21_encode(np.array(_PU21_MAX)))
    err = (pu21_encode(test_nits) - pu21_encode(ref_nits)) ** 2
    ref_y = ref_nits @ np.array([0.2627, 0.6780, 0.0593])
    inside = (ref_y >= _PU21_MIN) & (ref_y <= _PU21_MAX)
    full = float(err.mean())
    part = float(err[inside].mean()) if inside.any() else float("nan")
    db = lambda mse: float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)
    return dict(pu21_paper=db(full), pu21_in_range=db(part),
                ref_above_10k=float((ref_y > _PU21_MAX).mean()),
                ref_above_1500=float((ref_y > 1500.0).mean()),
                ref_peak=float(ref_y.max()))


# ---------------------------------------------------------------------------

def cmd_describe(args):
    import pycvvdp
    import pycvvdp.display_model as dm
    info = {"pycvvdp": getattr(pycvvdp, "__version__", "unknown")}
    for name in ("standard_hdr_linear", "standard_hdr_pq", "standard_4k"):
        p = dm.vvdp_display_photometry.load(name, [])
        info[name] = {k: v for k, v in vars(p).items()
                      if k in ("Y_peak", "contrast", "E_ambient", "k_refl", "EOTF", "rgb2xyz_list", "full_name")}
    bt709_y = [0.2126729, 0.7151522, 0.072175]
    lin = info["standard_hdr_linear"]
    lin["primaries"] = ("BT.709" if np.allclose(lin["rgb2xyz_list"][1], bt709_y, atol=1e-3) else "other")
    info["what_rudra_feeds_it"] = "Rec.2020 linear RGB in cd/m2 (rudra/hdrvdp.py: _to_rec2020_linear)"
    info["consequence"] = ("values above Y_peak are clipped before comparison; "
                           + ("primaries mismatch (BT.2020 data read as BT.709)" if lin["primaries"] == "BT.709" else ""))
    print(json.dumps(info, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(info, indent=2), encoding="utf-8")


def cmd_probe(args):
    """A 2x error on one bright patch, at increasing patch luminance; and a pure hue change."""
    import torch
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    metrics = make_metrics(device)
    rng = np.random.default_rng(0)
    base = (rng.random((256, 256, 3)) * 50 + 20).astype(np.float32)
    rows = []
    for patch in (200.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0, 8000.0):
        ref = base.copy(); ref[96:160, 96:160] = patch
        test = ref.copy(); test[96:160, 96:160] = patch * 2.0
        rows.append(dict(case=f"patch {patch:.0f} -> {2 * patch:.0f} nits",
                         **{n: jod(m, test, ref, device) for n, m in metrics.items()},
                         **{k: v for k, v in pu21_variants(test.astype(np.float64), ref.astype(np.float64)).items()
                            if k.startswith("pu21")}))
    # Hue: swap a saturated green patch for an equally bright, more saturated one.
    ref = base.copy(); ref[96:160, 96:160] = (20.0, 100.0, 20.0)
    test = base.copy(); test[96:160, 96:160] = (0.0, 100.0, 0.0)
    rows.append(dict(case="saturated green, chroma only",
                     **{n: jod(m, test, ref, device) for n, m in metrics.items()}))
    names = list(metrics) + ["pu21_paper", "pu21_in_range"]
    lines = ["| case | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for r in rows:
        lines.append(f"| {r['case']} | " + " | ".join(
            "" if r.get(n) is None else f"{r[n]:.3f}" for n in names) + " |")
    text = ("10.000 JOD means the display model cannot see the difference at all.\n\n"
            + "\n".join(lines) + "\n")
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


def cmd_rescore(args):
    import torch
    from rudra.delivery.bench import load_frame
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    metrics = make_metrics(device, {k: VARIANTS[k] for k in args.variants})
    bench = Path(args.bench)
    out_dir = bench / "results" / "display_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for cond in args.conditions:
        root = bench / cond
        refs = sorted(p for p in (root / "ref").rglob("*") if p.suffix.lower() in (".exr", ".npy"))
        refs = refs[: args.limit or None]
        per = {}
        for method in ["baseline", *args.methods]:
            cache = out_dir / f"{cond}_{method}.json"
            done = json.loads(cache.read_text()) if cache.exists() else {}
            for i, ref_path in enumerate(refs):
                rel = str(ref_path.relative_to(root / "ref"))
                if rel in done:
                    continue
                test_path = (root / method / ref_path.relative_to(root / "ref"))
                if not test_path.exists():
                    test_path = test_path.with_suffix(".exr" if test_path.suffix == ".npy" else ".npy")
                if not test_path.exists():
                    raise FileNotFoundError(f"{cond}/{method}: missing {rel}")
                ref = load_frame(ref_path, args.nits_scale)
                test = load_frame(test_path, args.nits_scale)
                row = pu21_variants(test, ref)
                for name, m in metrics.items():
                    row[name] = jod(m, test, ref, device)
                done[rel] = row
                if (i + 1) % 20 == 0:
                    cache.write_text(json.dumps(done), encoding="utf-8")
                    print(f"[{cond}/{method}] {i + 1}/{len(refs)}", flush=True)
            cache.write_text(json.dumps(done), encoding="utf-8")
            per[method] = done
        keys = sorted(per["baseline"])
        scenes = [k.replace("\\", "/").split("/")[0] for k in keys]
        from training.bench_hdrtv1k import bootstrap_ci
        report[cond] = {}
        for method in args.methods:
            report[cond][method] = {}
            for name in [*metrics, "pu21_paper", "pu21_in_range"]:
                g = np.array([per[method][k][name] - per["baseline"][k][name] for k in keys])
                ci = bootstrap_ci(g, scenes)
                ci["frames_better"] = int((g > 0).sum())
                report[cond][method][name] = ci
        report[cond]["_reference"] = dict(
            frames=len(keys),
            share_of_pixels_above_1500=float(np.mean([per["baseline"][k]["ref_above_1500"] for k in keys])),
            share_of_pixels_above_10000=float(np.mean([per["baseline"][k]["ref_above_10k"] for k in keys])),
            frames_with_pixels_above_1500=int(sum(per["baseline"][k]["ref_above_1500"] > 0 for k in keys)))
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = [*metrics, "pu21_paper", "pu21_in_range"]
    lines = ["# Paired gain over the analytic baseline under each display / range setting",
             "", "95% CI from a scene bootstrap. JOD columns in JOD, PU21 columns in dB.", ""]
    for cond, rows in report.items():
        ref = rows["_reference"]
        lines += [f"## {cond} ({ref['frames']} frames; {100 * ref['share_of_pixels_above_1500']:.2f}% of reference "
                  f"pixels above 1,500 nits, {100 * ref['share_of_pixels_above_10000']:.3f}% above 10,000; "
                  f"{ref['frames_with_pixels_above_1500']} frames have some)", "",
                  "| method | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for method, r in rows.items():
            if method.startswith("_"):
                continue
            lines.append(f"| {method} | " + " | ".join(
                f"{r[n]['mean']:+.3f} [{r[n]['lo']:+.3f}, {r[n]['hi']:+.3f}]" for n in names) + " |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe"); d.add_argument("--out"); d.set_defaults(func=cmd_describe)
    q = sub.add_parser("probe"); q.add_argument("--out"); q.add_argument("--device", default="cuda")
    q.set_defaults(func=cmd_probe)
    r = sub.add_parser("rescore")
    r.add_argument("--bench", required=True)
    r.add_argument("--methods", nargs="*", default=["v5", "v5_noshadow", "shadow_v1"])
    r.add_argument("--conditions", nargs="*", default=["clean", "hard"])
    r.add_argument("--nits-scale", type=float, default=203.0)
    r.add_argument("--device", default="cuda")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    r.set_defaults(func=cmd_rescore)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
