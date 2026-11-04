"""Real-footage training pairs for the 0 EV fix (option 1: keep the gate, fix the model; 4 Nov 2026).

The learned model fails 0 EV on real camera footage with many PARTIALLY clipped
highlights (fresh FXTD v2: 0.45 code RMS against 0.338) while passing every
development gate (w4). This builds training rows of that kind of data, with the
fresh set's own processing (tools/prepare_fresh_fxtd_grade_v2.py: longest side
--max-side in linear light, BT.2020 -> AP1, native exposure plus --offsets
stops, 8-bit codes through the ACES 2.0 SDR view), in the format
tools/train_partial_display.py reads (report.json rows: split, asset_id,
sdr_path canonical 16-bit PNG + sdr_sha256, target_path ACEScg .npy +
target_sha256), for its --extra-pilot.

Refused, whatever the input says: every scene of the fresh grading set and of
every holdout registration given in --exclude (any JSON naming scene_id, a
report.json or a holdout_kit registration), and anything run05's
evaluation-only rule refuses. The FXTD camera corpus is used up by the fresh
set (it took every scene with an unclipped reference), so in practice this
needs NEW footage, split by project: some projects for this, the others for
the holdout (docs/competitive/HOLDOUT_KIT.md).

    python tools/build_real_partial_corpus.py --sources new_shoot\\rows.jsonl --plates new_shoot\\hdr ^
        --exclude G:\\rudra_training\\fresh_fxtd_grade_v2_rebuild_20261102\\report.json configs\\experiments\\holdout_v1_*.json ^
        --out G:\\datasets\\corpora\\real_partial_v1 --dry-run

rows.jsonl (run05 sources format): one frame per line with asset_id, scene_id,
project, hdr_path (a decoded plate under --plates with its _ingest_config.json)
and the run05 plate fields (plate_primaries ...).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLIP = 254.5 / 255


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def excluded_scenes(paths: list[Path]) -> set[str]:
    """Every scene_id named anywhere in the given JSON files (rows, registrations, reports)."""
    out: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "scene_id" and isinstance(v, (str, int)):
                    out.add(str(v))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for p in paths:
        walk(json.loads(Path(p).read_text(encoding="utf-8")))
    return out


def pick_frames(rows: list[dict], per_scene: int, refused: set[str]) -> tuple[list[dict], dict]:
    """Up to `per_scene` frames per allowed scene, evenly spaced through each scene, deterministic."""
    by_scene = defaultdict(list)
    skipped = defaultdict(int)
    for r in rows:
        if str(r["scene_id"]) in refused:
            skipped["evaluation scene"] += 1
            continue
        by_scene[str(r["scene_id"])].append(r)
    picked = []
    for scene in sorted(by_scene):
        fr = sorted(by_scene[scene], key=lambda r: (r.get("frame_index", 0), r["asset_id"]))
        if len(fr) <= per_scene:
            picked += fr
        else:
            idx = np.linspace(0, len(fr) - 1, per_scene).round().astype(int)
            picked += [fr[i] for i in sorted(set(idx.tolist()))]
    return picked, dict(skipped)


def partial_pixels(src: np.ndarray) -> int:
    """The trainer's own partial support (build_pool): weight > 0, max > 0.5, not fully clipped."""
    import torch
    from training.aces_model import reconstruction_weight
    w = reconstruction_weight(torch.from_numpy(src).permute(2, 0, 1)[None])[0, 0].numpy()
    return int(((w > 0) & (src.max(-1) > .5) & (src.min(-1) < CLIP)).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--plates", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="+", required=True, help="the fresh set's report.json, holdout registrations ...")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames-per-scene", type=int, default=6)
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--max-side", type=int, default=768, help="the trainer's crops are 192 + borders")
    ap.add_argument("--min-partial", type=int, default=400, help="partial highlight pixels a pair needs to be kept")
    ap.add_argument("--dry-run", action="store_true", help="count what would be kept; write nothing")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in a.sources.read_text(encoding="utf-8").splitlines() if l.strip()]
    refused = excluded_scenes(a.exclude)
    picked, skipped = pick_frames(rows, a.frames_per_scene, refused)
    print(f"{len(rows)} frames in --sources; {len(refused)} evaluation scenes refused "
          f"({skipped.get('evaluation scene', 0)} frames); {len(picked)} frames from "
          f"{len({r['scene_id'] for r in picked})} scenes to process")
    if not picked:
        print("NOTHING LEFT: every scene is an evaluation scene. New footage is needed (see the docstring).")
        return 2

    import cv2
    import torch
    from pipeline.run05_corpus import NETWORK_PER_SCENE, canonical, load_plate
    from rudra.delivery.colorspace import convert
    from training.aces_reference import reference_view
    from training.sdr2hdr_dataset import storage_for
    torch.set_num_threads(2)
    view = reference_view()
    view.set_threads(4)
    if not a.dry_run:
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / "builder_snapshot.py").write_bytes(Path(__file__).read_bytes())
    out_rows, kept_scenes = [], set()
    for k, s in enumerate(picked):
        path = a.plates / Path(s["hdr_path"].replace("\\", "/")).name
        if storage_for(path) is None:
            raise ValueError(f"no ingest sentinel (_ingest_config.json) above {path}: a log2_extended plate would be read as linear")
        plate = load_plate({**s, "hdr_path": str(path)}) / NETWORK_PER_SCENE
        h, w = plate.shape[:2]
        f = a.max_side / max(h, w)
        if f < 1:
            plate = cv2.resize(plate, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        base = np.ascontiguousarray(convert(plate, "rec2020", "ap1").astype(np.float32))
        for off in a.offsets:
            tgt = (base * np.float32(2.0 ** off)).astype(np.float32)
            src = (np.rint(np.clip(view.forward(tgt), 0, 1) * 255) / 255).astype(np.float32)
            n_part = partial_pixels(src)
            keep = n_part >= a.min_partial and float(np.median(src.max(-1))) <= 0.9
            if not keep:
                continue
            kept_scenes.add(str(s["scene_id"]))
            aid = f"{s['asset_id']}__ev{off:+g}"
            row = {"split": "train", "asset_id": aid, "scene_id": str(s["scene_id"]), "project": s.get("project"),
                   "ev": off, "partial_pixels": n_part, "plate_sha256": sha(path)}
            if not a.dry_run:
                sp, tp = a.out / f"{len(out_rows):05d}_sdr.png", a.out / f"{len(out_rows):05d}_target.npy"
                canon = np.clip(np.rint(canonical(src) * 65535), 0, 65535).astype(np.uint16)
                if not cv2.imwrite(str(sp), np.ascontiguousarray(canon[..., ::-1])):
                    raise IOError(f"cannot write {sp}")
                np.save(tp, tgt)
                row.update(sdr_path=str(sp), sdr_sha256=sha(sp), target_path=str(tp), target_sha256=sha(tp))
            out_rows.append(row)
        print(k + 1, s["asset_id"], f"kept {sum(1 for r in out_rows if r['asset_id'].startswith(s['asset_id']))}", flush=True)
    summary = {"rows": out_rows, "kept_pairs": len(out_rows), "scenes": len(kept_scenes),
               "refused_evaluation_scenes": sorted(refused), "excluded_from": [str(p) for p in a.exclude],
               "exclude_sha256": {str(p): sha(p) for p in a.exclude}, "sources_sha256": sha(a.sources),
               "builder_sha256": sha(__file__), "offsets": a.offsets, "max_side": a.max_side, "min_partial": a.min_partial,
               "view": view.identity if isinstance(view.identity, dict) else str(view.identity),
               "scope": "training pairs for the 0 EV fix; no evaluation or holdout scene; release_authorized false"}
    print(f"\n{'would keep' if a.dry_run else 'kept'} {len(out_rows)} pairs from {len(kept_scenes)} scenes")
    if not a.dry_run:
        (a.out / "report.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        print(f"-> {a.out / 'report.json'}   (train with: --extra-pilot {a.out / 'report.json'})")
    return 0 if out_rows else 2


if __name__ == "__main__":
    sys.exit(main())
