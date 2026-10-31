"""The commercial training manifest: only sources a sold model may learn from (17 Oct 2026).

Every corpus built before this date mixes in HdM (academic only), so every
earlier mix and checkpoint is non-commercial. This rebuilds a manifest from
existing rows, keeping a row only when ALL of these hold:

  * pipeline.licences.classify() -- re-run here, not trusted from the row --
    puts it in a source allowed by the tier and says commercial_ok;
  * no path of the row lies under a quarantine folder (``_noncommercial``);
  * its scene sits in one split only.

Tiers:
  strict           Poly Haven (CC0) and FXTD's own footage. Clear today.
  netflix_pending  strict + Netflix Open Content (CC BY 4.0 basis, awaiting
                   counsel: docs/LEGAL_QUESTIONS.md Q1). Do not ship weights
                   trained on this tier until counsel clears it.

    python pipeline/build_commercial_manifest.py --tier strict \
        --manifest G:/datasets/corpora/rudra_mix_v11b_fxtd_20261017/sdr_hdr_manifest.jsonl \
        --out G:/datasets/corpora/rudra_commercial_v1_20261017/strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.licences import classify  # noqa: E402

TIERS = {
    "strict": {"polyhaven", "polyhaven_moves", "fxtd"},
    "netflix_pending": {"polyhaven", "polyhaven_moves", "fxtd", "netflix_sparks", "netflix_sol_levante",
                        "netflix_nocturne", "netflix_meridian", "netflix_cosmos", "netflix_chimera"},
}
PENDING = {"netflix_sparks", "netflix_sol_levante", "netflix_nocturne", "netflix_meridian",
           "netflix_cosmos", "netflix_chimera"}
PATH_KEYS = ("sdr_path", "hdr_path", "metadata_path", "source_path")


def _norm(p) -> str:
    return str(p or "").replace("\\", "/").lower()


def select(rows: list[dict], tier: str) -> tuple[list[dict], Counter]:
    allowed = TIERS[tier]
    kept, dropped = [], Counter()
    for r in rows:
        if any("/_noncommercial/" in _norm(r.get(k)) for k in PATH_KEYS):
            dropped["quarantined_path"] += 1
            continue
        c = classify(r)
        if c["source"] not in allowed or not c["commercial_ok"]:
            dropped[c["source"]] += 1
            continue
        r = dict(r, licence_source=c["source"], licence_licence=c["licence"], commercial_ok=True,
                 rights_status="pending_counsel" if c["source"] in PENDING else "clear",
                 commercial_tier=tier)
        kept.append(r)
    splits = defaultdict(set)
    for r in kept:
        splits[str(r["scene_id"])].add(r.get("split"))
    straddle = sorted(s for s, v in splits.items() if len(v) > 1)
    if straddle:
        raise SystemExit(f"{len(straddle)} scenes straddle splits, e.g. {straddle[:3]}")
    ids = [r["asset_id"] for r in kept]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate asset ids")
    return kept, dropped


def report_for(kept: list[dict], dropped: Counter, tier: str) -> dict:
    train = [r for r in kept if r.get("split") == "train"]
    scenes = Counter(r["scene_id"] for r in train)
    top = scenes.most_common(3)
    return {
        "tier": tier, "rows": len(kept),
        "by_split": dict(Counter(r.get("split") for r in kept)),
        "by_source_split": {f"{s}/{sp}": n for (s, sp), n in
                            sorted(Counter((r["licence_source"], r.get("split")) for r in kept).items())},
        "by_kind": dict(Counter(f"{r.get('sdr_kind', 'rendered')}{'/' + r['render_family'] if r.get('render_family') else ''}"
                                for r in kept)),
        "train_scenes": len(scenes),
        "train_top_scene_share": [(s, round(n / max(len(train), 1), 4)) for s, n in top],
        "rights_status": dict(Counter(r["rights_status"] for r in kept)),
        "dropped_by_source": dict(dropped),
        "tonemap_ev_rendered": sorted({str(r.get("tonemap_ev")) for r in kept
                                       if r.get("sdr_kind") != "real" and r.get("tonemap_ev") is not None}),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", nargs="+", required=True, type=Path)
    ap.add_argument("--tier", choices=sorted(TIERS), required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for m in args.manifest
            for line in m.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, dropped = select(rows, args.tier)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "sdr_hdr_manifest.jsonl"
    with target.open("w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    report = report_for(kept, dropped, args.tier)
    report.update({"inputs": [str(m) for m in args.manifest],
                   "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (args.out / "commercial_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
