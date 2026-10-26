#!/usr/bin/env python3
"""Collect run05 sources (scene-linear plates) from existing corpus manifests (22 Oct 2026).

    python pipeline/run05_sources.py --from <manifest.jsonl> --source-id fxtd_camera \
        [--hdr-field hdr_path] [--project-filter ...] [--assign-split-by-scene 0.1] --out sources_part.jsonl
    (repeat per corpus, then concatenate; run05_corpus.build checks scenes never cross splits)

Rules (fixed): test-split rows are dropped (Mataha and other reserved test material stay
untouched); evaluation-only scenes (Tears of Steel, Mataha, SI-HDR) are refused by
run05_corpus.check_source; rows without explicit commercial_ok True are dropped; an EXR field needs
--exr-scale and --exr-primaries rec2020 so units are stated, not guessed.
"""
from __future__ import annotations

import argparse
import sys
import hashlib
import json
from pathlib import Path


def split_for(scene: str, val_fraction: float) -> str:
    h = int.from_bytes(hashlib.sha256(scene.encode()).digest()[:4], "little") / 2 ** 32
    return "val" if h < val_fraction else "train"


def collect(rows, source_id, hdr_field="hdr_path", project=None, assign=None, exr_scale=None, exr_primaries=None):
    out, dropped = [], {"test": 0, "not_commercial": 0, "excluded": 0, "project": 0, "no_path": 0}
    for r in rows:
        if project and r.get("project") not in project:
            dropped["project"] += 1; continue
        if r.get("excluded"):
            dropped["excluded"] += 1; continue
        if r.get("commercial_ok") is not True:
            dropped["not_commercial"] += 1; continue
        split = split_for(r["scene_id"], assign) if assign is not None else r.get("split")
        if split == "test" or (assign is not None and r.get("split") == "test"):
            dropped["test"] += 1; continue
        path = r.get(hdr_field)
        if not path:
            dropped["no_path"] += 1; continue
        row = {"source_id": source_id, "scene_id": r["scene_id"], "split": split, "asset_id": r["asset_id"],
               "hdr_path": path, "hdr_sha256": r.get(f"{hdr_field.replace('_path', '')}_sha256"),
               "ceiling_nits": r.get("ceiling_nits"), "project": r.get("project"),
               "licence_source": r.get("licence_source"), "licence_licence": r.get("licence_licence"),
               "commercial_ok": r.get("commercial_ok"), "dataset_scope": r.get("dataset_scope")}
        if str(path).lower().endswith(".exr"):
            row.update(exr_scale=exr_scale, exr_primaries=exr_primaries)
        out.append(row)
    return out, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, required=True)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--hdr-field", default="hdr_path")
    ap.add_argument("--project-filter", nargs="*")
    ap.add_argument("--assign-split-by-scene", type=float, help="val fraction; for manifests without splits")
    ap.add_argument("--exr-scale", type=float)
    ap.add_argument("--exr-primaries", choices=["rec2020"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.src.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, dropped = collect(rows, a.source_id, a.hdr_field, a.project_filter, a.assign_split_by_scene,
                           a.exr_scale, a.exr_primaries)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"rows": len(out), "dropped": dropped,
                      "splits": {s: sum(r["split"] == s for r in out) for s in ("train", "val")}}, indent=2))
    return 0




# --------------------------------------------------------------------------
# The RUDRA disk layout as of 22 Oct 2026 (stdlib only; runs anywhere).
# --------------------------------------------------------------------------
import re
from pathlib import PureWindowsPath

FORBIDDEN = ("tears", "tos_", "mataha", "sihdr", "si-hdr", "si_hdr")


def panorama_key(scene_id: str) -> str:
    s = scene_id.lower()
    for pre in ("tif:", "exr:", "hdr:", "hl_ph_", "ph_capture_", "ph_"):
        if s.startswith(pre):
            s = s[len(pre):]
    return "ph:" + re.sub(r"_\d+k$", "", s)


def collect_rudra(run02_manifest: Path, corpus_a: Path, moves_manifest: Path, val_fraction: float = 0.1,
                  frames_per_clip: str = "ends", root_map: dict | None = None) -> tuple[list, dict]:
    """Split authority: the run02 frozen snapshot (its val/test scenes are the frozen kits).
    A scene run02 holds keeps run02's split; test scenes are dropped; Poly Haven panoramas are
    keyed by name across corpora so one panorama never sits on both sides."""
    root_map = root_map or {}

    def fix(p: str) -> str:
        for a, b in root_map.items():
            if p.lower().startswith(a.lower()):
                return b + p[len(a):]
        return p

    r02 = [json.loads(l) for l in run02_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    split_of = {r["scene_id"]: r["split"] for r in r02}
    ph_split = {}
    for r in r02:
        if "polyhaven" in str(r.get("source_id")) or str(r.get("scene_id", "")).startswith(("hl_ph_", "ph_")):
            ph_split[panorama_key(r["scene_id"])] = r["split"]
    out, notes = [], {"dropped_test": 0, "dropped_forbidden": 0, "dropped_noncommercial": 0}

    def forbidden(r):
        t = f"{r.get('scene_id', '')} {r.get('asset_id', '')} {r.get('source_id', '')} {r.get('project', '')}".lower()
        return any(f in t for f in FORBIDDEN)

    def add(row):
        if forbidden(row):
            notes["dropped_forbidden"] += 1
        elif row["split"] == "test":
            notes["dropped_test"] += 1
        elif row.get("commercial_ok") is not True:
            notes["dropped_noncommercial"] += 1
        else:
            out.append(row)

    # 1. FXTD camera plates (Corpus A): scene-linear BT.2020, sentinel-decoded PNG.
    for r in (json.loads(l) for l in corpus_a.read_text(encoding="utf-8").splitlines() if l.strip()):
        add({"source_id": f"fxtd_{r['source_id']}", "scene_id": r["scene_id"],
             "split": split_of.get(r["scene_id"], r["split"]), "asset_id": r["asset_id"],
             "hdr_path": fix(r["hdr_path"]), "hdr_sha256": r.get("hdr_sha256"), "plate_primaries": "rec2020",
             "ceiling_nits": r.get("ceiling_nits"), "project": r.get("project"),
             "licence_source": r.get("licence_source"), "licence_licence": r.get("licence_licence"),
             "commercial_ok": r.get("commercial_ok"), "dataset_scope": "fxtd_camera_plate"})
    # 2. CC0 / authored renders: the retained scene-linear Rec.709 EXRs behind run02's pipeline rows.
    seen = set()
    for r in r02:
        src = str(r.get("source_path") or "")
        # Only our own retained Rec.709 renders. Camera EXRs (RED linear, camera gamut) are
        # NOT Rec.709 and are already covered, correctly converted, by Corpus A.
        if not src.lower().endswith(".exr") or "\\linear_exr\\" not in src.lower() or src in seen:
            continue
        seen.add(src)
        if r.get("commercial_ok") is not True:
            notes["dropped_noncommercial"] += 1
            continue
        add({"source_id": f"{r['source_id']}_linear", "scene_id": r["scene_id"], "split": r["split"],
             "asset_id": PureWindowsPath(src).stem, "hdr_path": fix(src), "plate_primaries": "rec709",
             "exr_scale": 203.0 / 10000.0, "project": r.get("project"),
             "licence_source": r.get("licence_source"), "licence_licence": r.get("licence_licence"),
             "commercial_ok": r.get("commercial_ok"), "dataset_scope": "cc0_render_linear"})
    # 3. Poly Haven camera moves (CC0): scene-linear Rec.709 (render_hdri_moves does no gamut change).
    for c in (json.loads(l) for l in moves_manifest.read_text(encoding="utf-8").splitlines() if l.strip()):
        key = panorama_key(c["scene_id"])
        split = ph_split.get(key) or split_for(key, val_fraction)
        frames = c["hdr_frames"]
        pick = [frames[0], frames[-1]] if frames_per_clip == "ends" else frames
        for f in dict.fromkeys(pick):
            add({"source_id": "polyhaven_moves", "scene_id": key, "split": split, "asset_id": PureWindowsPath(f).stem,
                 "hdr_path": fix(f), "plate_primaries": "rec709", "project": "polyhaven_moves",
                 "licence_source": "Poly Haven", "licence_licence": "CC0", "commercial_ok": True,
                 "dataset_scope": "polyhaven_camera_move"})
    scenes = {}
    for r in out:
        if scenes.setdefault(r["scene_id"], r["split"]) != r["split"]:
            raise ValueError(f"scene {r['scene_id']} would cross splits")
    notes.update(rows=len(out), by_source={s: sum(r["source_id"] == s for r in out) for s in sorted({r["source_id"] for r in out})},
                 splits={s: sum(r["split"] == s for r in out) for s in ("train", "val")})
    return out, notes


def main_rudra(argv) -> int:
    ap = argparse.ArgumentParser(description="RUDRA layout collector (22 Oct 2026)")
    ap.add_argument("--run02", type=Path, default=Path(r"G:\rudra_training\fxtd_native1080_hdr_run02\data\manifest.jsonl"))
    ap.add_argument("--corpus-a", type=Path, default=Path(r"G:\datasets\corpora\rudra_fxtd_camera_20261016\sdr_hdr_manifest.jsonl"))
    ap.add_argument("--moves", type=Path, default=Path(r"G:\rudra_v02\video_manifest_moves.jsonl"))
    ap.add_argument("--map", nargs="*", default=[r"D:\A.I\Devlopments\RUDRA_v02=G:\rudra_v02"],
                    help="old=new path prefixes")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows, notes = collect_rudra(a.run02, a.corpus_a, a.moves, a.val_fraction,
                                root_map=dict(m.split("=", 1) for m in a.map))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(notes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_rudra(sys.argv[2:]) if sys.argv[1:2] == ["rudra"] else main())
