"""H7: is RUDRA's real-SDR shadow error a global exposure mismatch? Never promotes.

Review point W6 (1 Oct 2026). The analytic baseline multiplies by 2 to undo the -1 EV our
corpus applies before its ACES curve. Real BT.709 SDR was never put through that, so on
real SDR the shipped model may simply sit at the wrong exposure, and a correction bounded
to one stop -- the R2 to R5 runs -- would learn that offset and look like a "systematic
shadow error". That is exactly what invalidated run R0. This experiment controls for it.

Hypothesis. If the shadow error is a global exposure offset, then after each source's
exposure is aligned with ONE gain fitted on frames the evaluation never sees, the same
one-stop correction trained the same way (the R2 recipe) finds little shadow improvement
left to make.

  sources    each --eval manifest's 'val' rows, labelled by --labels (HDRTV1K val,
             Netflix grades, ...). Frames sorted by asset_id; EVEN positions fit the
             exposure, ODD positions are evaluated. No frame is used for both.
  exposure   per frame, k* = median over interior pixels of log2(Y_target / Y_RUDRA)
             (SDR luma code in [0.15, 0.85], target uncensored and above 0.05 nits).
             k_source = median of k* over the source's even frames.
  updates    the SAME 4 real-SDR HDRTV1K frames as R2 (read from --source-protocol)
  arms       control: correction on RUDRA, as R2.
             aligned: correction on RUDRA * 2^k (k_hdrtv1k for the updates, k_source at
             evaluation). 64 steps, AdamW lr 1e-4 wd 0.01, seed 20260930, one stop, both arms.
  scoring    PU21 region errors (target < 12 / 12-203 / >= 203 nits) and the censored log
             error, clean and degraded input, odd frames only; deltas against each arm's
             own teacher, negative = better.

Rule (fixed before running). Let S_c and S_a be the mean shadow delta of the control and
aligned arms over all odd-frame measurements (clean and degraded pooled), and F_a the share
of aligned-arm shadow measurements that improve.
  EXPOSURE      |S_a| < 0.5 |S_c|  or  F_a < 0.75   (most of the error was exposure)
  SURVIVES      |S_a| >= 0.5 |S_c| and F_a >= 0.90  (a shadow error remains after alignment)
  PARTIAL       otherwise
The control arm must reproduce R2 (shadows improve on >= 90% of measurements) or the run is
INVALID. Also reported, not ruled on: k* per source, and what the exposure alignment alone
does to every region.

    python -m training.audit_exposure_control --manifest <upgrade manifest.jsonl> \
        --eval <upgrade manifest.jsonl> <netflix manifest.jsonl> --labels hdrtv1k_val netflix \
        --source-protocol <outputs/correction_transfer_20261001_031005/protocol.json> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SEED = 20260930
STEPS = 64
LR = 1e-4
MAX_STOPS = 1.0
CHECKPOINT = Path("checkpoints/sdr2hdr_shadow_v1.pt")
REGIONS = ("shadow", "mid", "high")
LUMA2020 = (0.2627, 0.6780, 0.0593)
INTERIOR = (0.15, 0.85)


# ------------------------------------------------- scoring (as the R2-R5 harness)

def pu_encode(value):
    """Differentiable PU21 banding+glare on nits (same constants as training/robust_reconstruction.py)."""
    y = value.float().clamp(.005, 10000).pow(.9062562627)
    return 5.963148142 * (((.353487901 + .3734658629 * y) / (1 + 8.277049286e-05 * y)).pow(.09150303166) - .9099517204)


def region_pu_risk(pred, target, ceiling):
    import torch
    from rudra.sdr2hdr import CENSORED_HEADROOM_STOPS
    target, pred = target.float(), pred.float()
    ep, et = pu_encode(pred * 10000), pu_encode(target * 10000)
    error = (ep - et).square()
    if ceiling is not None:
        ceiling = torch.as_tensor(ceiling, device=target.device, dtype=target.dtype).reshape(-1, 1, 1, 1)
        censored = (target >= ceiling * (1 - 1e-3)) & torch.isfinite(ceiling)
        allowance = pu_encode(ceiling * (2 ** CENSORED_HEADROOM_STOPS) * 10000)
        bounded = (et - ep).relu().square() + (ep - allowance).relu().square()
        error = torch.where(censored, bounded, error)
    error = error.mean(1)
    luma = (target * target.new_tensor(LUMA2020)[None, :, None, None]).sum(1) * 10000
    masks = torch.stack([luma < 12, (luma >= 12) & (luma < 203), luma >= 203], 1)
    counts = masks.flatten(2).sum(2)
    return (error[:, None] * masks).flatten(2).sum(2) / counts.clamp_min(1), counts > 0


def reconstruction_risk(pred, target, ceiling):
    from rudra.sdr2hdr import censored_log_error
    return censored_log_error(pred.float(), target.float(), ceiling)[0].flatten(1).mean(1)


def region_paired_objective(predictions, teachers, target, ceiling, harm_weight=4.0):
    """The R2-R5 objective (training/robust_reconstruction.py on the experiment branch)."""
    import torch
    risks = torch.stack([reconstruction_risk(p, target, ceiling) for p in predictions])
    ref = torch.stack([reconstruction_risk(p.detach(), target, ceiling) for p in teachers])
    base = risks.mean(1).mean() + harm_weight * (risks - ref).relu().mean()
    rr = torch.stack([region_pu_risk(p, target, ceiling)[0] for p in predictions])
    rref = torch.stack([region_pu_risk(p.detach(), target, ceiling)[0] for p in teachers])
    present = region_pu_risk(target, target, ceiling)[1][None].expand_as(rr)
    harm = ((rr - rref).relu() * present).sum() / present.sum().clamp_min(1)
    groups = (rr * present).sum((1, 2)) / present.sum((1, 2)).clamp_min(1)
    return base + groups.max() + harm_weight * harm


def named(pred, b) -> dict:
    out = dict(log=reconstruction_risk(pred, b["hdr"], b["ceiling"])[0].item())
    risks, present = region_pu_risk(pred, b["hdr"], b["ceiling"])
    for k, r in enumerate(REGIONS):
        out[r] = risks[0, k].item() if bool(present[0, k]) else None
    return out


def delta(new: dict, old: dict) -> dict:
    return {k: (None if new[k] is None else new[k] - old[k]) for k in new}


# ------------------------------------------------------------------ exposure

def exposure_stops(pred, b) -> float | None:
    """Median log2(target / prediction) luminance ratio over interior, uncensored pixels."""
    import torch
    w = torch.tensor(LUMA2020, device=pred.device)[None, :, None, None]
    yp = (pred.float().clamp_min(0) * w).sum(1)
    yt = (b["hdr"].float().clamp_min(0) * w).sum(1)
    code = (b["clean_sdr"].float() * w).sum(1)
    ceiling = b["ceiling"].reshape(-1, 1, 1)
    ok = (code >= INTERIOR[0]) & (code <= INTERIOR[1]) & (yt > 5e-6) & (yp > 5e-6)
    ok &= ~((yt >= ceiling * (1 - 1e-3)) & torch.isfinite(ceiling))
    if int(ok.sum()) < 64:
        return None
    return float(torch.log2(yt[ok] / yp[ok]).median().item())


class Correction:
    """FrozenCorrection (unrestricted) on a teacher whose output is scaled by 2^k."""

    def __init__(self, teacher, width=16):
        import torch
        from torch import nn
        self.teacher = teacher.eval().requires_grad_(False)
        self.net = nn.Sequential(nn.Conv2d(6, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, 3, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.torch = torch

    def to(self, device):
        self.net.to(device)
        return self

    def base(self, sdr, stops: float):
        with self.torch.no_grad():
            return self.teacher(sdr, preserve_outside=True).hdr * (2.0 ** stops)

    def __call__(self, sdr, stops: float):
        base = self.base(sdr, stops)
        feats = self.torch.cat((sdr, self.torch.log1p(base.clamp_min(0))), 1)
        return base * self.torch.exp2(MAX_STOPS * self.torch.tanh(self.net(feats)))


# ----------------------------------------------------------------------- run

def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verdict(control: list[dict], aligned: list[dict]) -> dict:
    def shadow(ms):
        return [m["deltas"][c]["shadow"] for m in ms for c in ("clean", "degraded")
                if m["deltas"][c]["shadow"] is not None]
    sc, sa = shadow(control), shadow(aligned)
    S_c, S_a = sum(sc) / len(sc), sum(sa) / len(sa)
    F_c, F_a = sum(x < 0 for x in sc) / len(sc), sum(x < 0 for x in sa) / len(sa)
    if F_c < 0.9 or S_c >= 0:
        call = "INVALID"
    elif abs(S_a) < 0.5 * abs(S_c) or F_a < 0.75:
        call = "EXPOSURE"
    elif F_a >= 0.9:
        call = "SURVIVES"
    else:
        call = "PARTIAL"
    return dict(call=call, S_control=S_c, S_aligned=S_a, ratio=(S_a / S_c if S_c else math.nan),
                F_control=F_c, F_aligned=F_a, measurements=len(sa))


def summarize_regions(ms: list[dict]) -> dict:
    out = {}
    for c in ("clean", "degraded"):
        for r in (*REGIONS, "log"):
            xs = [m["deltas"][c][r] for m in ms if m["deltas"][c][r] is not None]
            out[f"{c}_{r}"] = None if not xs else dict(
                mean=sum(xs) / len(xs), worst=max(xs), better=sum(x < 0 for x in xs), n=len(xs))
    return out


def run(args) -> dict:
    import numpy as np
    import torch
    from training.sdr2hdr_dataset import SDRHDRDataset, read_jsonl
    from training.train_sdr2hdr import load_image_checkpoint, seed_everything

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    originals = {"manifest": args.manifest, "eval": list(args.eval)}
    if args.root:
        # Manifests carry absolute Windows paths. Remap their corpus folders to local
        # copies; hashes are still taken of the ORIGINAL manifest files.
        roots = dict(r.split("=", 1) for r in args.root)

        def remapped(path):
            text = Path(path).read_text(encoding="utf-8")
            rows = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                for key in ("sdr_path", "hdr_path"):
                    v = r[key].replace("\\", "/")
                    for name, local in roots.items():
                        if name in v:
                            r[key] = str(Path(local) / v.split(name, 1)[1].lstrip("/"))
                rows.append(json.dumps(r))
            dst = out / ("remapped_" + hashlib.sha256(str(path).encode()).hexdigest()[:8] + ".jsonl")
            dst.write_text("\n".join(rows) + "\n", encoding="utf-8")
            return str(dst)
        args.manifest = remapped(args.manifest)
        args.eval = [remapped(e) for e in args.eval]
    source = json.loads(Path(args.source_protocol).read_text(encoding="utf-8"))
    update_ids = [g if isinstance(g, str) else g["asset_id"] for g in source["groups"]["update"]]
    rows = read_jsonl(args.manifest)
    if source.get("manifest_sha256") and source["manifest_sha256"] != digest(Path(originals["manifest"])):
        raise ValueError("--manifest is not the manifest the R2 run used")
    train = SDRHDRDataset(args.manifest, split="train", crop_size=args.crop, augment=False,
                          deterministic_degradation=True)
    loc = {r["asset_id"]: k for k, r in enumerate(train.records)}
    if any(rows[[r["asset_id"] for r in rows].index(a)].get("synthetic_sdr") for a in update_ids):
        raise ValueError("update frames must be real SDR")

    def batch(ds, k):
        b = ds[k]
        return {key: b[key][None].to(device) for key in ("clean_sdr", "sdr", "hdr", "ceiling")}

    sources = []
    for manifest, label in zip(args.eval, args.labels):
        ds = SDRHDRDataset(manifest, split="val", crop_size=args.crop, augment=False, deterministic_degradation=True)
        order = sorted(range(len(ds.records)), key=lambda k: ds.records[k]["asset_id"])
        if set(ds.records[k]["asset_id"] for k in order) & set(update_ids):
            raise ValueError(f"{label}: an update frame is in the evaluation set")
        orig = originals["eval"][len(sources)]
        sources.append(dict(label=label, ds=ds, fit=order[0::2], test=order[1::2], manifest=str(orig),
                            manifest_sha256=digest(Path(orig))))

    protocol = dict(rule=__doc__.split("Rule (fixed before running).")[1].split("    python -m")[0].strip(),
                    steps=STEPS, lr=LR, seed=SEED, max_stops=MAX_STOPS, crop=args.crop, interior=INTERIOR,
                    updates=update_ids, checkpoint_sha256=digest(CHECKPOINT), script_sha256=digest(Path(__file__)),
                    manifest_sha256=digest(Path(originals["manifest"])), source_protocol=str(args.source_protocol),
                    sources=[{k: s[k] for k in ("label", "manifest", "manifest_sha256")} |
                             dict(fit=[s["ds"].records[k]["asset_id"] for k in s["fit"]],
                                  test=[s["ds"].records[k]["asset_id"] for k in s["test"]]) for s in sources])
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    teacher, _ = load_image_checkpoint(CHECKPOINT, device)
    teacher.eval()

    # Step 1: exposure per frame, per source (fit frames only decide k_source).
    exposure = {}
    with torch.no_grad():
        for s in sources:
            ks = {}
            for k in s["fit"] + s["test"]:
                b = batch(s["ds"], k)
                ks[s["ds"].records[k]["asset_id"]] = exposure_stops(teacher(b["clean_sdr"], preserve_outside=True).hdr, b)
            fit = [ks[s["ds"].records[k]["asset_id"]] for k in s["fit"] if ks[s["ds"].records[k]["asset_id"]] is not None]
            allk = [v for v in ks.values() if v is not None]
            s["k"] = float(np.median(fit))
            exposure[s["label"]] = dict(k_source=s["k"], fit_frames=len(fit), per_frame=ks,
                                        all_median=float(np.median(allk)),
                                        all_iqr=[float(np.quantile(allk, .25)), float(np.quantile(allk, .75))])
            print(f"[{s['label']}] k_source = {s['k']:+.3f} stops (median over {len(fit)} fit frames)", flush=True)
    k_updates = next((s["k"] for s in sources if s["label"].startswith("hdrtv1k")), sources[0]["k"])

    updates = [batch(train, loc[a]) for a in update_ids]
    tests = [(s, s["ds"].records[k]["asset_id"], batch(s["ds"], k)) for s in sources for k in s["test"]]

    # Step 2: what exposure alignment alone does, on the odd frames.
    alignment_only = []
    with torch.no_grad():
        for s, asset, b in tests:
            d = {}
            for c, key in (("clean", "clean_sdr"), ("degraded", "sdr")):
                p = teacher(b[key], preserve_outside=True).hdr
                d[c] = delta(named(p * 2.0 ** s["k"], b), named(p, b))
            alignment_only.append(dict(source=s["label"], asset_id=asset, deltas=d))

    # Step 3: the two arms.
    arms = {}
    for arm in ("control", "aligned"):
        seed_everything(SEED)
        model = Correction(teacher).to(device)
        opt = torch.optim.AdamW(model.net.parameters(), lr=LR, weight_decay=.01)
        k_upd = k_updates if arm == "aligned" else 0.0
        with torch.no_grad():
            refs = [[model.base(b[key], k_upd) for key in ("clean_sdr", "sdr")] for b in updates]
        for step in range(STEPS):
            j = step % len(updates)
            b = updates[j]
            preds = [model(b[key], k_upd) for key in ("clean_sdr", "sdr")]
            loss = region_paired_objective(preds, refs[j], b["hdr"], b["ceiling"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        ms = []
        with torch.no_grad():
            for s, asset, b in tests:
                k_eval = s["k"] if arm == "aligned" else 0.0
                d = {c: delta(named(model(b[key], k_eval), b), named(model.base(b[key], k_eval), b))
                     for c, key in (("clean", "clean_sdr"), ("degraded", "sdr"))}
                ms.append(dict(source=s["label"], asset_id=asset, deltas=d))
        arms[arm] = ms
        print(f"[{arm}] trained and scored on {len(ms)} frames", flush=True)

    result = dict(
        verdict=verdict(arms["control"], arms["aligned"]),
        by_source={s["label"]: verdict([m for m in arms["control"] if m["source"] == s["label"]],
                                       [m for m in arms["aligned"] if m["source"] == s["label"]])
                   for s in sources},
        exposure=exposure, k_updates=k_updates,
        alignment_only=summarize_regions(alignment_only),
        alignment_only_by_source={s["label"]: summarize_regions([m for m in alignment_only if m["source"] == s["label"]])
                                  for s in sources},
        arms={a: summarize_regions(ms) for a, ms in arms.items()},
        measurements=dict(alignment_only=alignment_only, **arms), promoted=False)
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    v = result["verdict"]
    print(f"\nH7 verdict: {v['call']}  (shadow delta control {v['S_control']:+.4f}, aligned {v['S_aligned']:+.4f}, "
          f"ratio {v['ratio']:.2f}; improved control {100 * v['F_control']:.0f}%, aligned {100 * v['F_aligned']:.0f}%)")
    for label, e in exposure.items():
        print(f"  {label}: k_source {e['k_source']:+.3f} stops, all frames median {e['all_median']:+.3f} "
              f"IQR [{e['all_iqr'][0]:+.3f}, {e['all_iqr'][1]:+.3f}]")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="the R2 training manifest (updates come from here)")
    p.add_argument("--eval", nargs="+", required=True, help="manifests whose 'val' rows are evaluated")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--source-protocol", required=True, help="protocol.json of the R2 transfer run")
    p.add_argument("--out", required=True)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--root", action="append", default=[], help="NAME=LOCALDIR to remap manifest paths")
    args = p.parse_args(argv)
    if len(args.eval) != len(args.labels):
        p.error("--eval and --labels must have the same length")
    run(args)


if __name__ == "__main__":
    main()
