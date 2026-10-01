"""Scene-level bootstrap confidence intervals for the paper's 429-frame benchmark.

Review point W5 (1 Oct 2026): the 429 held-out frames come from 97 scenes, so frames are
not independent, and "frames won" and mean gains need intervals that resample scenes.
Reads the results ``rudra bench`` already wrote; runs no model.

    python -m training.bench_bootstrap --bench <bench dir>
    python -m training.bench_bootstrap --bench <bench dir> --methods v5 shadow_v1 shadow_s2 shadow_s3

<bench>/results/<cond>_<method>.json holds per-pair rows with ``clip`` (the scene),
``frame``, ``pu_psnr_db`` and ``cvvdp_jod``. Every method is paired with ``baseline`` on
(clip, frame); a missing pair is an error, not a silent drop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.bench_hdrtv1k import bootstrap_ci

METRICS = ("pu_psnr_db", "cvvdp_jod")
DEFAULT_METHODS = ("v5", "v5_noshadow", "v6", "shadow_v1", "shadow_s2", "shadow_s3")


def load(results: Path, cond: str, method: str) -> dict[tuple[str, str], dict]:
    path = results / f"{cond}_{method}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    return {(r["clip"], r["frame"]): r for r in rows}


def paired(results: Path, cond: str, method: str, reference: str = "baseline", n: int = 10000) -> dict:
    ref, test = load(results, cond, reference), load(results, cond, method)
    if set(ref) != set(test):
        raise ValueError(f"{cond}_{method} and {cond}_{reference} cover different frames "
                         f"({len(set(ref) ^ set(test))} differ)")
    keys = sorted(ref)
    scenes = [k[0] for k in keys]
    out = {}
    for metric in METRICS:
        pairs = [(test[k][metric], ref[k][metric]) for k in keys]
        if any(a is None or b is None for a, b in pairs):
            out[metric] = None
            continue
        gain = np.array([a - b for a, b in pairs], dtype=np.float64)
        scene_ci = bootstrap_ci(gain, scenes, n=n)
        frame_ci = bootstrap_ci(gain, None, n=n)
        won = (gain > 0).astype(np.float64)
        won_ci = bootstrap_ci(won, scenes, n=n)
        out[metric] = dict(mean=scene_ci["mean"], scene_ci=[scene_ci["lo"], scene_ci["hi"]],
                           frame_ci=[frame_ci["lo"], frame_ci["hi"]], p_positive=scene_ci["p_positive"],
                           frames_won=int(won.sum()), frames=len(keys),
                           frames_won_share_scene_ci=[won_ci["lo"], won_ci["hi"]],
                           scenes=scene_ci["units"])
    return out


def seed_sweep(results: Path, cond: str, seeds: list[str], n: int = 10000, seed: int = 20261001) -> dict:
    """Mean gain over seeds with a scene bootstrap that resamples scenes jointly for all seeds."""
    ref = load(results, cond, "baseline")
    keys = sorted(ref)
    scenes = sorted({k[0] for k in keys})
    index = {s: [i for i, k in enumerate(keys) if k[0] == s] for s in scenes}
    out = {}
    for metric in METRICS:
        per_seed = []
        for s in seeds:
            t = load(results, cond, s)
            per_seed.append(np.array([t[k][metric] - ref[k][metric] for k in keys], dtype=np.float64))
        g = np.mean(per_seed, axis=0)          # seed-averaged gain per frame
        rng = np.random.default_rng(seed)
        sums = np.array([g[index[s]].sum() for s in scenes])
        counts = np.array([len(index[s]) for s in scenes])
        pick = rng.integers(0, len(scenes), size=(n, len(scenes)))
        means = sums[pick].sum(1) / counts[pick].sum(1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        out[metric] = dict(mean=float(g.mean()), scene_ci=[float(lo), float(hi)],
                           seed_means=[float(p.mean()) for p in per_seed],
                           seed_sd=float(np.std([p.mean() for p in per_seed], ddof=1)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    p.add_argument("--seeds", nargs="*", default=["shadow_v1", "shadow_s2", "shadow_s3"])
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--out", help="default: <bench>/results/bootstrap.{json,md}")
    args = p.parse_args(argv)
    results = Path(args.bench) / "results"
    report = dict(paired={}, seed_sweep={}, resamples=args.resamples,
                  note="percentile bootstrap; scene_ci resamples scenes (clips), frame_ci resamples frames")
    for cond in ("clean", "hard"):
        for m in args.methods:
            if (results / f"{cond}_{m}.json").exists():
                report["paired"][f"{cond}/{m}"] = paired(results, cond, m, n=args.resamples)
            else:
                print(f"skip {cond}_{m}: no results file")
        if all((results / f"{cond}_{s}.json").exists() for s in args.seeds):
            report["seed_sweep"][cond] = seed_sweep(results, cond, args.seeds, n=args.resamples)
    out = Path(args.out) if args.out else results / "bootstrap"
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Paired gain over the analytic baseline, 95% CI (scene bootstrap)", "",
             "| condition / method | PU21 dB | scene CI | frame CI | won | JOD | scene CI |",
             "|---|---|---|---|---|---|---|"]
    for key, r in report["paired"].items():
        d, j = r["pu_psnr_db"], r["cvvdp_jod"]
        jt = ("n/a", "n/a") if j is None else (f"{j['mean']:+.3f}", f"[{j['scene_ci'][0]:+.3f}, {j['scene_ci'][1]:+.3f}]")
        lines.append(f"| {key} | {d['mean']:+.2f} | [{d['scene_ci'][0]:+.2f}, {d['scene_ci'][1]:+.2f}] | "
                     f"[{d['frame_ci'][0]:+.2f}, {d['frame_ci'][1]:+.2f}] | {d['frames_won']}/{d['frames']} | "
                     f"{jt[0]} | {jt[1]} |")
    for cond, r in report["seed_sweep"].items():
        d, j = r["pu_psnr_db"], r["cvvdp_jod"]
        lines += ["", f"Gate, mean of {len(args.seeds)} seeds, {cond}: "
                  f"{d['mean']:+.2f} dB [{d['scene_ci'][0]:+.2f}, {d['scene_ci'][1]:+.2f}] (seed sd {d['seed_sd']:.2f}), "
                  f"{j['mean']:+.3f} JOD [{j['scene_ci'][0]:+.3f}, {j['scene_ci'][1]:+.3f}] (seed sd {j['seed_sd']:.3f})"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
