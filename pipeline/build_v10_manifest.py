"""The v10 training manifest: validation that looks like the test (13 Oct 2026).

The real-SDR validation split was Cosmos Laundromat, an animation, while the
real test title (Meridian) is live action; the two disagreed (STATUS line E).
This writes a copy of a mixed manifest where:

  * the real val rows are whole shots held out of the live-action train titles
    (Nocturne, Sparks by default), ``--val-fraction`` of their scenes chosen
    by a seeded draw, so no shot is on both sides;
  * the old real val title (Cosmos Laundromat) moves to train, or with
    ``--drop-old-val`` out of the manifest;
  * every rendered row and every test row is left as it was.

    python pipeline/build_v10_manifest.py --manifest <mix.jsonl> --out <dir>

Writes <dir>/sdr_hdr_manifest.jsonl and <dir>/v10_split.json (what moved).
The corpus exposure (_ingest_config.json beside the source manifest) is copied,
so corpus_ev_of() reads the same 0 EV.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

LIVE_ACTION = ("Nocturne", "Sparks")


def resplit(rows: list[dict], live_action=LIVE_ACTION, val_fraction: float = 0.12, seed: int = 20261013,
            drop_old_val: bool = False) -> tuple[list[dict], dict]:
    real = [r for r in rows if r.get("sdr_kind") == "real"]
    titles = {str(r.get("title", "")) for r in real}
    missing = [t for t in live_action if t not in titles]
    if missing:
        raise SystemExit(f"no real rows from {missing} (titles here: {sorted(titles)})")
    scenes = sorted({str(r["scene_id"]) for r in real
                     if r.get("split") == "train" and str(r.get("title")) in live_action})
    rng = random.Random(seed)
    n_val = max(1, round(len(scenes) * val_fraction))
    held = set(rng.sample(scenes, n_val))
    out, moved = [], Counter()
    for r in rows:
        r = dict(r)
        if r.get("sdr_kind") == "real":
            if r.get("split") == "val":                       # the old real val title
                if drop_old_val:
                    moved["old_val_dropped"] += 1
                    continue
                r["split"], r["split_v10_from"] = "train", "val"
                moved["old_val_to_train"] += 1
            elif r.get("split") == "train" and str(r["scene_id"]) in held:
                r["split"], r["split_v10_from"] = "val", "train"
                moved["live_action_to_val"] += 1
        out.append(r)
    by = Counter((r.get("split"), r.get("sdr_kind", "rendered"), str(r.get("title", "")) if r.get("sdr_kind") == "real" else "")
                 for r in out)
    report = {
        "live_action_titles": list(live_action), "val_fraction": val_fraction, "seed": seed,
        "held_out_scenes": sorted(held), "moved": dict(moved),
        "rows": {"/".join(k for k in key if k): n for key, n in sorted(by.items())},
    }
    return out, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--live-action", nargs="+", default=list(LIVE_ACTION))
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--drop-old-val", action="store_true")
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    out, report = resplit(rows, tuple(args.live_action), args.val_fraction, args.seed, args.drop_old_val)
    # Relative paths were relative to the source manifest's folder (or the cwd);
    # pin the ones that resolve there, so the copy reads the same files.
    base = args.manifest.parent
    for r in out:
        for k in ("sdr_path", "hdr_path", "metadata_path"):
            v = r.get(k)
            if v and not Path(v).is_absolute() and (base / v).exists():
                r[k] = str((base / v).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "sdr_hdr_manifest.jsonl"
    if target.resolve() == args.manifest.resolve():
        raise SystemExit("--out would overwrite the source manifest")
    with target.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    cfg = args.manifest.parent / "_ingest_config.json"
    if cfg.is_file():
        shutil.copyfile(cfg, args.out / "_ingest_config.json")
    report["source_manifest"] = str(args.manifest)
    (args.out / "v10_split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("moved", "rows")}, indent=2))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
