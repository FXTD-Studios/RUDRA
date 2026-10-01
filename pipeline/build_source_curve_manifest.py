"""One training manifest where every row says which curve made its SDR (task 1.2).

Merges a rendered corpus (corpus_v4c: SDR drawn per pair from
pipeline/sdr_render.py, curve id in ``sdr_curve``) with real-SDR corpora
(rudra_netflix_realsdr: a colourist's Dolby Vision trim pass, curve unknown)
and writes ``source_curve`` on every row:

  rendered rows   source_curve = sdr_curve        (aces, hable, reinhard, agx,
                                                    camera_log, clip)
                  sdr_kind     = "rendered"
  real rows       source_curve = "unknown"
                  sdr_kind     = "real"           (corpus_ev_of skips these)

A rendered corpus without ``sdr_curve`` (corpus_v4b and earlier: one ACES
render for everything) needs ``--assume-curve aces``; nothing is guessed
silently.

Refuses to write when:
  * a row's curve is outside the vocabulary,
  * a scene appears in two splits after the merge,
  * any item of the frozen comparison set (configs/compare_set_v1.json) would
    land in train or val -- by asset id, by scene, or by file path,
  * two rows share an asset id.

    python pipeline/build_source_curve_manifest.py ^
        --rendered G:\\datasets\\corpora\\corpus_v4c\\sdr_hdr_manifest.jsonl ^
        --real G:\\datasets\\corpora\\rudra_netflix_realsdr_20261001\\manifest.jsonl ^
        --compare-set configs\\compare_set_v1.json ^
        --out G:\\datasets\\corpora\\rudra_mix_v4c_netflix_20261001\\sdr_hdr_manifest.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

# The curves pipeline/sdr_render.py can draw, plus the label for "not ours".
RENDER_CURVES = ("aces", "hable", "reinhard", "agx", "camera_log", "clip")
UNKNOWN = "unknown"
SOURCE_CURVES = RENDER_CURVES + (UNKNOWN,)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(path: str) -> str:
    return str(path).replace("\\", "/").lower()


def label_rendered(rows: list[dict], assume: str | None, origin: str) -> list[dict]:
    out = []
    for r in rows:
        curve = r.get("sdr_curve") or assume
        if curve is None:
            raise SystemExit(f"{origin}: row {r.get('asset_id')} has no sdr_curve; pass --assume-curve "
                             f"only if the whole corpus was rendered with one curve")
        if curve not in RENDER_CURVES:
            raise SystemExit(f"{origin}: row {r.get('asset_id')} has curve {curve!r}, not in {RENDER_CURVES}")
        out.append({**r, "source_curve": curve, "sdr_kind": "rendered", "source_corpus": origin})
    return out


def label_real(rows: list[dict], origin: str) -> list[dict]:
    out = []
    for r in rows:
        if r.get("sdr_kind", "real") != "real":
            raise SystemExit(f"{origin}: row {r.get('asset_id')} says sdr_kind={r.get('sdr_kind')!r}")
        out.append({**r, "source_curve": UNKNOWN, "sdr_kind": "real", "source_corpus": origin})
    return out


def frozen_keys(compare_set: Path | None) -> tuple[set[str], set[str], set[str]]:
    if compare_set is None:
        return set(), set(), set()
    items = json.loads(compare_set.read_text(encoding="utf-8"))["items"]
    ids, scenes, paths = set(), set(), set()
    for it in items:
        if it.get("asset_id"):
            ids.add(str(it["asset_id"]))
        if it.get("scene_id"):
            scenes.add(str(it["scene_id"]))
        for k in ("sdr_path", "hdr_path"):
            if it.get(k):
                paths.add(_norm(it[k]))
        for k in ("sdr_frames", "hdr_frames"):
            paths.update(_norm(p) for p in it.get(k, []))
    return ids, scenes, paths


def frozen_conflicts(rows: list[dict], compare_set: Path | None) -> list[dict]:
    """Train/val rows that touch the frozen comparison set (id, scene or file)."""
    ids, scenes, paths = frozen_keys(compare_set)
    return [r for r in rows if r["split"] in ("train", "val") and (
        str(r["asset_id"]) in ids or str(r["scene_id"]) in scenes
        or _norm(r["sdr_path"]) in paths or _norm(r["hdr_path"]) in paths)]


def check(rows: list[dict], compare_set: Path | None) -> dict:
    dup = [a for a, n in Counter(str(r["asset_id"]) for r in rows).items() if n > 1]
    if dup:
        raise SystemExit(f"{len(dup)} duplicate asset ids, e.g. {dup[:3]}")
    splits = defaultdict(set)
    for r in rows:
        splits[str(r["scene_id"])].add(r["split"])
    straddle = sorted(s for s, v in splits.items() if len(v) > 1)
    if straddle:
        raise SystemExit(f"{len(straddle)} scenes in two splits, e.g. {straddle[:3]}")
    _, scenes, _ = frozen_keys(compare_set)
    leaks = [r["asset_id"] for r in frozen_conflicts(rows, compare_set)]
    if leaks:
        raise SystemExit(f"{len(leaks)} comparison-set items would train or validate, e.g. {leaks[:3]} "
                         f"(--drop-frozen removes them from the merged manifest)")
    bad = [r["asset_id"] for r in rows if r["source_curve"] not in SOURCE_CURVES]
    if bad:
        raise SystemExit(f"{len(bad)} rows with an unknown source_curve label")
    table = Counter((r["split"], r["source_curve"]) for r in rows)
    return {
        "records": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_split_curve": {f"{s}/{c}": n for (s, c), n in sorted(table.items())},
        "by_kind": dict(Counter(r["sdr_kind"] for r in rows)),
        "scenes": len(splits),
        "compare_set_checked": bool(compare_set),
        "frozen_scenes_in_test": sorted(s for s in scenes if s in splits),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rendered", type=Path, action="append", default=[], help="rendered-SDR manifest (repeatable)")
    ap.add_argument("--real", type=Path, action="append", default=[], help="real-SDR manifest (repeatable)")
    ap.add_argument("--assume-curve", choices=RENDER_CURVES, default=None,
                    help="curve for rendered rows that lack sdr_curve (v4b and earlier: aces)")
    ap.add_argument("--compare-set", type=Path, default=None)
    ap.add_argument("--drop-frozen", action="store_true",
                    help="drop train/val rows that touch the comparison set instead of refusing "
                         "(corpus_v4c trains on carousel_fireworks, a v4b test scene)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if not args.rendered and not args.real:
        raise SystemExit("give at least one --rendered or --real manifest")
    if args.compare_set is None:
        print("WARNING: no --compare-set; the frozen comparison set is not being kept out of training")
    rows: list[dict] = []
    for p in args.rendered:
        rows += label_rendered(read_jsonl(p), args.assume_curve, p.parent.name)
    for p in args.real:
        rows += label_real(read_jsonl(p), p.parent.name)
    dropped: list[dict] = []
    if args.drop_frozen:
        conflicts = {id(r) for r in frozen_conflicts(rows, args.compare_set)}
        dropped = [r for r in rows if id(r) in conflicts]
        rows = [r for r in rows if id(r) not in conflicts]
    report = check(rows, args.compare_set)
    report["dropped_for_compare_set"] = {
        "records": len(dropped),
        "by_scene": dict(Counter(f"{r['source_corpus']}:{r['split']}:{r['scene_id']}" for r in dropped)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; write a new manifest instead of overwriting one a run may cite")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report["manifest"] = str(args.out)
    report["manifest_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    report["inputs"] = {"rendered": [str(p) for p in args.rendered], "real": [str(p) for p in args.real],
                        "assume_curve": args.assume_curve,
                        "compare_set": str(args.compare_set) if args.compare_set else None}
    args.out.with_name(args.out.stem + "_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
