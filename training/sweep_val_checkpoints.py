"""Re-score every step_*.pt of a run on the FULL val split, clean and hard,
with gains split by sdr_kind (real grade vs rendered pair).

    python training/sweep_val_checkpoints.py --run checkpoints/sdr2hdr_image_v7
    python training/sweep_val_checkpoints.py --run checkpoints/sdr2hdr_image_v7 --manifest <other.jsonl>
    python training/sweep_val_checkpoints.py --run ... --every 2   # every other checkpoint

Why (7 Oct 2026): v7's train.jsonl shows composite_gain peaking at step 1,500
and decaying for the remaining 48k steps while train loss keeps falling. That
eval is 8 batches x 4 = 32 records of a 538-row val split, ~30% of them real
SDR rows where the analytic inverse is stops wrong. Before deciding whether
the decay is overfitting or a 32-record artefact, score the whole split and
split the gain by kind. Output: one row per checkpoint, printed and written
as <run>/val_sweep.jsonl (resumable: scored steps are skipped).

Needs the GPU box (G: manifests). ~2 min per checkpoint at 538 rows.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.sdr2hdr_dataset import SDRHDRDataset  # noqa: E402
from training.train_sdr2hdr import (deterministic_eval_order, evaluate_image,  # noqa: E402
                                    load_image_checkpoint)

COLUMNS = ("clean_gain_db", "clean_real_gain_db", "clean_rendered_gain_db",
           "hard_gain_db", "hard_real_gain_db", "hard_rendered_gain_db",
           "composite_gain", "clean_psnr_log", "clean_baseline_psnr_log")


def step_of(path: Path) -> int:
    m = re.search(r"step_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="checkpoint dir holding config.json and step_*.pt")
    ap.add_argument("--manifest", help="override the manifest in config.json (same val split rules)")
    ap.add_argument("--every", type=int, default=1, help="score every Nth checkpoint")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-val-items", type=int, help="cap the split (default: all of it)")
    ap.add_argument("--include-best", action="store_true", help="also score best.pt as its own row")
    args = ap.parse_args()

    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    manifest = args.manifest or config["manifest"]
    crop = int(config.get("crop_size", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val = SDRHDRDataset(manifest, split="val", augment=False, augmentation_strength=0.0,
                        max_items=args.max_val_items, crop_size=crop)
    hard = SDRHDRDataset(manifest, split="val", augment=False, augmentation_strength=0.0,
                         max_items=args.max_val_items, crop_size=crop, deterministic_degradation=True)
    order = deterministic_eval_order(len(val))
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                       pin_memory=device.type == "cuda")
    clean_loader = DataLoader(Subset(val, order), **loader_args)
    hard_loader = DataLoader(Subset(hard, order[:len(hard)]), **loader_args)
    n_real = sum(1 for r in val.records if r.get("sdr_kind") == "real")
    print(f"val split: {len(val)} records ({n_real} real, {len(val) - n_real} rendered), crop {crop}, {device}")

    out = args.run / "val_sweep.jsonl"
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["checkpoint"]] = row
    ckpts = sorted((p for p in args.run.glob("step_*.pt")), key=step_of)[::max(args.every, 1)]
    if args.include_best and (args.run / "best.pt").exists():
        ckpts.append(args.run / "best.pt")
    header = f"{'ckpt':>14} " + " ".join(f"{c[:18]:>18}" for c in COLUMNS)
    print(header)
    for row in done.values():
        print(_fmt(row))
    with out.open("a", encoding="utf-8") as fh:
        for ckpt in ckpts:
            if ckpt.name in done:
                continue
            model, payload = load_image_checkpoint(ckpt, device)
            sc = float(config.get("shadow_chroma_weight", 0.15))
            ss = float(config.get("shadow_smoothness_weight", 0.02))
            with torch.no_grad():
                clean = evaluate_image(model, clean_loader, device, 0, sc, ss)
                hard_m = evaluate_image(model, hard_loader, device, 0, sc, ss)
            row = {"checkpoint": ckpt.name, "step": payload.get("step", step_of(ckpt)),
                   "val_records": len(val), "real_records": n_real}
            row.update({f"clean_{k}": v for k, v in clean.items()})
            row.update({f"hard_{k}": v for k, v in hard_m.items()})
            row["composite_gain"] = row["hard_gain_db"] + min(0.0, row["clean_gain_db"])
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(_fmt(row))
            del model
    print(f"-> {out}")
    return 0


def _fmt(row: dict) -> str:
    cells = []
    for c in COLUMNS:
        v = row.get(c)
        cells.append(f"{v:>18.3f}" if isinstance(v, (int, float)) else f"{'-':>18}")
    return f"{row['checkpoint']:>14} " + " ".join(cells)


if __name__ == "__main__":
    raise SystemExit(main())
