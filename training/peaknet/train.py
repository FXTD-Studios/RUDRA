"""Train PeakNet on build_dataset.py shards and score it against the shipped clip fill (17 Oct 2026).

The loss IS the gate's measure: for each region, E(a) (mean error in stops
inside it as a function of the dome height a, from build_dataset.py) is
linearly interpolated at the predicted a, and regions are weighted by their
share of their frame's clipped pixels, so a frame counts once, as the clip
benches count it. E is convex-ish and piecewise linear in a, so its
interpolation is a usable gradient.

Reported per split, overall and by push bucket (+0, +1, +2 EV, as the
cp8_clip benches) and by source: mean frame error of the shipped default,
of PeakNet, of the per-region oracle, the paired difference with a 95 %
frame bootstrap interval, and how many frames PeakNet wins.

    python training/peaknet/train.py --data G:/datasets/corpora/rudra_peaknet_v1_20261017 --out checkpoints/peaknet_v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUCKETS = ((0.0, 0.75, "+0 EV"), (0.75, 1.75, "+1 EV"), (1.75, 9.0, "+2 EV"))


def load_split(data: Path, split: str) -> dict | None:
    files = sorted((data / "shards").glob(f"{split}_*.npz"))
    if not files:
        return None
    parts = [np.load(f, allow_pickle=False) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    meta = [json.loads(m) for m in out.pop("meta")]
    out["key"] = np.array([f"{m['frame']}#{m['exposure']}#{m['region']}" for m in meta])
    frame_key = [f"{m['frame']}#{m['exposure']}" for m in meta]
    keys, frame = np.unique(np.array(frame_key), return_inverse=True)
    tot = np.bincount(frame, weights=out["pixels"])
    out["weight"] = (out["pixels"] / tot[frame]).astype(np.float32)
    out["frame"] = frame.astype(np.int64)
    out["ev"] = np.array([m["render_ev"] for m in meta], np.float32)
    out["kind"] = np.array([m["kind"] for m in meta])
    out["n_frames"] = len(keys)
    return out


class PeakNet(nn.Module):
    """Context patch (6 x 64 x 64) + region features -> dome height a >= 0 (stops)."""

    def __init__(self, n_feats: int, a_max: float, width: int = 32, dropout: float = 0.0):
        super().__init__()
        c = width
        self.a_max = float(a_max)
        self.conv = nn.Sequential(
            nn.Conv2d(6, c, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(2 * c, 3 * c, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(3 * c, 4 * c, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.feat = nn.Sequential(nn.Linear(n_feats, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(8 * c + 64, 128), nn.SiLU(), nn.Linear(128, 1))
        self.dropout = float(dropout)      # applied in forward, so v1 checkpoints (no dropout) load strict
        self.register_buffer("f_mean", torch.zeros(n_feats))
        self.register_buffer("f_std", torch.ones(n_feats))

    def forward(self, patch: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        h = self.conv(patch)
        h = torch.cat((h.mean(dim=(2, 3)), h.amax(dim=(2, 3))), dim=1)
        f = self.feat((feats - self.f_mean) / self.f_std)
        z = self.head[1](self.head[0](torch.cat((h, f), dim=1)))
        z = F.dropout(z, self.dropout, self.training)
        return F.softplus(self.head[2](z).squeeze(1)).clamp(max=self.a_max)


def interp_curve(ecurve: torch.Tensor, a: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    step = grid[1] - grid[0]
    pos = (a / step).clamp(0, len(grid) - 1 - 1e-6)
    lo = pos.floor().long()
    t = pos - lo.to(pos.dtype)
    e0 = ecurve.gather(1, lo[:, None]).squeeze(1)
    e1 = ecurve.gather(1, (lo + 1).clamp(max=len(grid) - 1)[:, None]).squeeze(1)
    return e0 + (e1 - e0) * t


@torch.no_grad()
def predict(model, d: dict, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(d["feats"]), batch):
        out.append(model(torch.from_numpy(d["patch"][i:i + batch].astype(np.float32)).to(device),
                         torch.from_numpy(d["feats"][i:i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def frame_errors(d: dict, region_err: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    m = np.ones(len(region_err), bool) if mask is None else mask
    num = np.bincount(d["frame"][m], weights=(d["weight"] * region_err)[m], minlength=d["n_frames"])
    den = np.bincount(d["frame"][m], weights=d["weight"][m], minlength=d["n_frames"])
    keep = den > 0
    return num[keep] / den[keep]


def compare(d: dict, a_pred: np.ndarray, grid: np.ndarray, seed: int = 0) -> dict:
    e_pred = np.array([np.interp(a, grid, c) for a, c in zip(a_pred, d["ecurve"])], np.float32)
    rng = np.random.default_rng(seed)
    report = {}
    groups = {"all": np.ones(len(a_pred), bool)}
    for lo, hi, name in BUCKETS:
        groups[name] = (d["ev"] >= lo) & (d["ev"] < hi)
    for kind in np.unique(d["kind"]):
        groups[str(kind)] = d["kind"] == kind
    for name, m in groups.items():
        if not m.any():
            continue
        fd, fp, fo = (frame_errors(d, e, m) for e in (d["e_default"], e_pred, d["e_oracle"]))
        diff = fd - fp                                      # > 0: PeakNet better
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
        report[name] = {"frames": int(len(diff)), "default": round(float(fd.mean()), 4),
                        "peaknet": round(float(fp.mean()), 4), "oracle": round(float(fo.mean()), 4),
                        "gain_stops": round(float(diff.mean()), 4),
                        "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                        "frames_won": int((diff > 0).sum())}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--feat-noise", type=float, default=0.0,
                    help="std of Gaussian noise on standardised features in training (v2 regulariser)")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump-only", type=Path, default=None,
                    help="load this peaknet.pt, skip training, write predictions and the report")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = json.loads((args.data / "_peaknet_config.json").read_text())
    grid = np.asarray(cfg["a_grid"], np.float32)
    tr, va = load_split(args.data, "train"), load_split(args.data, "val")
    te = load_split(args.data, "test")
    print(f"train {len(tr['feats'])} regions / {tr['n_frames']} frames; "
          f"val {0 if va is None else len(va['feats'])}; test {0 if te is None else len(te['feats'])}", flush=True)
    if args.dump_only is not None:
        ck = torch.load(args.dump_only, map_location=device, weights_only=True)
        model = PeakNet(int(ck["n_feats"]), float(ck["a_grid"][-1]), int(ck["width"])).to(device)
        model.load_state_dict(ck["state_dict"])
        args.steps = 0
    else:
        model = PeakNet(tr["feats"].shape[1], float(grid[-1]), args.width, args.dropout).to(device)
    if args.dump_only is None:
        model.f_mean.copy_(torch.from_numpy(tr["feats"].mean(0)))
        model.f_std.copy_(torch.from_numpy(tr["feats"].std(0) + 1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(args.steps, 1))
    g = torch.from_numpy(grid).to(device)
    patch = torch.from_numpy(tr["patch"])          # float16 in memory; v2 is 3.6 GB as float32
    feats, ecurve = torch.from_numpy(tr["feats"]), torch.from_numpy(tr["ecurve"])
    weight = torch.from_numpy(tr["weight"])
    n = len(feats)
    best, best_state, t0, log = None, None, time.time(), []
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, n, (args.batch,))
        p = patch[idx].to(device).float()
        if torch.rand(()) < 0.5:
            p = p.flip(-1)                                   # the left-right mirror changes nothing physical
        fb = feats[idx].to(device)
        if args.feat_noise > 0:
            fb = fb + torch.randn_like(fb) * model.f_std * args.feat_noise
        a = model(p, fb)
        w = weight[idx].to(device)
        loss = (interp_curve(ecurve[idx].to(device), a, g) * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            row = {"step": step, "loss": round(float(loss.detach()), 4), "s": round(time.time() - t0)}
            if va is not None:
                rep = compare(va, predict(model, va, device), grid)
                row["val_gain"] = rep["all"]["gain_stops"]
                if best is None or row["val_gain"] > best:
                    best, best_state = row["val_gain"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log.append(row)
            print(json.dumps(row), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dump_only is None:
        torch.save({"state_dict": model.state_dict(), "n_feats": int(tr["feats"].shape[1]), "width": args.width, "dropout": args.dropout,
                    "a_grid": grid.tolist(), "features": cfg["features"]}, args.out / "peaknet.pt")
    result = {"config": vars(args) | {"data": str(args.data), "out": str(args.out)}, "log": log}
    preds = {}
    for name, d in (("train", tr), ("val", va), ("test", te)):
        if d is not None:
            a = predict(model, d, device)
            result[name] = compare(d, a, grid)
            if name != "train":
                preds.update({k: round(float(v), 4) for k, v in zip(d["key"], a)})
    (args.out / "predictions.json").write_text(json.dumps(preds))
    (args.out / "report.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("val", "test") if k in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
