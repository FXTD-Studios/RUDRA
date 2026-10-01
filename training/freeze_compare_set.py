"""Freeze the external comparison set (roadmap task 2.1). Deterministic, run once.

The set every outside SDR-to-HDR tool is scored on, and that no RUDRA training
run may touch:

  stills   60 frames from the corpus_v4b TEST split, one per scene, from the
           60 scenes ranked first by sha256("rudra-compare-v1|" + scene_id);
           within a scene the record with the lowest sha256 of its asset_id.
           Video scenes are excluded here (they are covered as clips).
  clips    every TEST clip of the corpus_v4b video manifest (7 on 24 Sep 2026,
           all from carousel_fireworks and fireplace; the plan said 10, there
           are 7).
  netflix  40 frames from the Netflix real-SDR corpus TEST title (Meridian),
           spread round-robin across its shots, ranked by the same hash.

Selection depends only on ids, never on pixels, so it cannot be tuned to a
result. Each referenced file is hashed, so a later change to any input is
detected by --check.

    python training/freeze_compare_set.py \
        --v4b-manifest G:\\datasets\\corpora\\corpus_v4b\\sdr_hdr_manifest.jsonl \
        --v4b-video-manifest G:\\datasets\\corpora\\corpus_v4b\\video_manifest_9f.jsonl \
        --netflix-manifest G:\\datasets\\corpora\\rudra_netflix_realsdr_20261001\\manifest.jsonl \
        --out configs\\compare_set_v1.json
    python training/freeze_compare_set.py --check configs\\compare_set_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

SALT = "rudra-compare-v1|"
N_STILLS = 60
N_NETFLIX = 40


def _h(text: str) -> str:
    return hashlib.sha256((SALT + text).encode("utf-8")).hexdigest()


def _file_sha(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pick_stills(rows: list[dict], video_scenes: set[str]) -> list[dict]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("split") == "test" and not r.get("is_video") and r["scene_id"] not in video_scenes:
            by_scene[r["scene_id"]].append(r)
    scenes = sorted(by_scene, key=_h)[:N_STILLS]
    if len(scenes) < N_STILLS:
        raise SystemExit(f"only {len(scenes)} eligible test scenes, need {N_STILLS}")
    out = []
    for s in scenes:
        r = min(by_scene[s], key=lambda x: _h(str(x["asset_id"])))
        out.append({"kind": "still", "source": "corpus_v4b", "asset_id": r["asset_id"],
                    "scene_id": s, "sdr_path": r["sdr_path"], "hdr_path": r["hdr_path"]})
    return out


def pick_clips(rows: list[dict]) -> list[dict]:
    clips = [r for r in rows if r.get("split") == "test"]
    clips.sort(key=lambda r: str(r["clip_id"]))
    return [{"kind": "clip", "source": "corpus_v4b_video", "clip_id": r["clip_id"],
             "scene_id": r.get("scene_id"), "sdr_frames": r["sdr_frames"], "hdr_frames": r["hdr_frames"]}
            for r in clips]


def pick_netflix(rows: list[dict]) -> list[dict]:
    by_shot: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("split") == "test":
            by_shot[r["scene_id"]].append(r)
    for shot in by_shot:
        by_shot[shot].sort(key=lambda r: _h(r["asset_id"]))
    shots = sorted(by_shot, key=_h)
    out, i = [], 0
    while len(out) < N_NETFLIX and any(by_shot.values()):
        shot = shots[i % len(shots)]
        if by_shot[shot]:
            r = by_shot[shot].pop(0)
            out.append({"kind": "netflix_real_sdr", "source": "rudra_netflix_realsdr",
                        "asset_id": r["asset_id"], "scene_id": shot, "title": r.get("title"),
                        "sdr_path": r["sdr_path"], "hdr_path": r["hdr_path"]})
        i += 1
    if len(out) < N_NETFLIX:
        raise SystemExit(f"only {len(out)} Netflix test frames, need {N_NETFLIX}")
    return out


def _hash_items(items: list[dict]) -> None:
    for it in items:
        if it["kind"] == "clip":
            it["sdr_sha256"] = [_file_sha(p) for p in it["sdr_frames"]]
            it["hdr_sha256"] = [_file_sha(p) for p in it["hdr_frames"]]
        else:
            it["sdr_sha256"] = _file_sha(it["sdr_path"])
            it["hdr_sha256"] = _file_sha(it["hdr_path"])


def check(path: Path) -> int:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    bad = 0
    for it in frozen["items"]:
        pairs = (zip(it["sdr_frames"] + it["hdr_frames"], it["sdr_sha256"] + it["hdr_sha256"])
                 if it["kind"] == "clip" else
                 ((it["sdr_path"], it["sdr_sha256"]), (it["hdr_path"], it["hdr_sha256"])))
        for p, want in pairs:
            if _file_sha(p) != want:
                print(f"CHANGED or MISSING: {p}")
                bad += 1
    print(f"{len(frozen['items'])} items, {bad} file(s) differ from the freeze")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v4b-manifest", type=Path)
    ap.add_argument("--v4b-video-manifest", type=Path)
    ap.add_argument("--netflix-manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("configs/compare_set_v1.json"))
    ap.add_argument("--check", type=Path, help="verify an existing freeze against the files")
    args = ap.parse_args(argv)
    if args.check:
        return check(args.check)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a freeze is written once (use --check)")
    if not (args.v4b_manifest and args.v4b_video_manifest and args.netflix_manifest):
        raise SystemExit("--v4b-manifest, --v4b-video-manifest and --netflix-manifest are required")
    video_rows = _read(args.v4b_video_manifest)
    clips = pick_clips(video_rows)
    video_scenes = {str(c["scene_id"]) for c in clips if c.get("scene_id")}
    items = pick_stills(_read(args.v4b_manifest), video_scenes) + clips + pick_netflix(_read(args.netflix_manifest))
    _hash_items(items)
    missing = sum(1 for it in items for k in ("sdr_sha256", "hdr_sha256")
                  if (it[k] is None if not isinstance(it[k], list) else None in it[k]))
    if missing:
        raise SystemExit(f"{missing} referenced file(s) missing; nothing written")
    frozen = {
        "name": "compare_set_v1", "created": time.strftime("%Y-%m-%d"), "salt": SALT,
        "rule": "ids only, sha256 ranking; see training/freeze_compare_set.py",
        "inputs": {k: str(v) for k, v in (("v4b_manifest", args.v4b_manifest),
                                          ("v4b_video_manifest", args.v4b_video_manifest),
                                          ("netflix_manifest", args.netflix_manifest))},
        "inputs_sha256": {k: _file_sha(str(v)) for k, v in (("v4b_manifest", args.v4b_manifest),
                                                            ("v4b_video_manifest", args.v4b_video_manifest),
                                                            ("netflix_manifest", args.netflix_manifest))},
        "counts": {k: sum(1 for i in items if i["kind"] == k) for k in ("still", "clip", "netflix_real_sdr")},
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    print(json.dumps(frozen["counts"]), "->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
