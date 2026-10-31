"""Quarantine non-commercial datasets and write the dataset registry (17 Oct 2026).

Moves (renames, same volume, reversible) every academic / non-commercial
source and every corpus made only of them into ``<root>/_noncommercial/``,
then writes ``<root>/datasets_registry.json`` and ``<root>/DATASETS.md``.
Nothing is deleted. Mixed corpora (corpus_v4b, corpus_v4c, the rudra_mix_*
manifests, temporal_clips) are NOT moved: they hold the clean Poly Haven and
Netflix rows the commercial manifest reads. They are marked non-commercial in
the registry instead, and build_commercial_manifest.py filters them row by row.

    python pipeline/organize_datasets.py --root G:/datasets            # dry run
    python pipeline/organize_datasets.py --root G:/datasets --apply
    python pipeline/organize_datasets.py --root G:/datasets --restore  # move everything back
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

QUARANTINE = "_noncommercial"

# (relative path, licence, reason)
NON_COMMERCIAL = [
    ("sources/stuttgart_hdr_2014", "HdM-HDR-2014, academic only", "commercial use needs an HdM licence"),
    ("sources/hdm_hfr_2017", "HdM-HFR-2017, academic only", "commercial use needs an HdM licence"),
    ("sources/hdrtv1k", "HDRTV1K, academic only", "no commercial grant"),
    ("sources/dml_hdr", "DML-HDR, non-commercial", "no commercial grant"),
    ("sources/kaist_sr_itm", "KAIST SR-ITM, academic, no licence stated", "no commercial grant"),
    ("sources/hdrtv4k", "HDRTV4K, academic (never downloaded)", "no commercial grant"),
    ("corpora/rudra_upgrade_20260930", "98 % HDRTV1K", "built from academic data"),
    ("corpora/rudra_realsdr_pilot_20261001", "84 % HDRTV1K / unknown", "built from academic data"),
    ("corpora/rudra_kaist_eval_20261001", "100 % KAIST", "built from academic data"),
]

REGISTRY_STATIC = {
    "commercial_clear": {
        "sources/polyhaven": "Poly Haven HDRIs, CC0",
        "corpora/rudra_fxtd_camera_20261016": "FXTD own camera footage (rendered SDR)",
        "corpora/rudra_fxtd_pipeline_20261017": "FXTD own footage, ACES SDR/HDR pipeline pairs",
        "corpora/rudra_commercial_v1_20261017/strict": "TRAINING MANIFEST: Poly Haven + FXTD only",
    },
    "commercial_pending_counsel": {
        "sources/netflix_sparks": "Netflix Open Content, CC BY 4.0 basis vs old NC-ND README (LEGAL_QUESTIONS Q1)",
        "sources/netflix_chimera": "same",
        "sources/netflix_open": "same",
        "sources/netflix_sol_levante": "same",
        "corpora/rudra_netflix_realsdr_20261001": "Netflix real SDR/HDR pairs, same",
        "corpora/rudra_netflix_eval_20261001": "same",
        "corpora/rudra_commercial_v1_20261017/netflix_pending": "TRAINING MANIFEST: strict + Netflix; no shipping until counsel clears",
        "sources/pandora": "custom 'commercial research' wording (Q5); incomplete download",
        "sources/liu_hdrv": "CC BY-SA 4.0, ShareAlike vs weights (Q6); not extracted",
    },
    "mixed_noncommercial_keep_in_place": {
        "corpora/corpus_v4b": "56 % HdM; also holds clean rows",
        "corpora/corpus_v4c": "56 % HdM; holds the clean Poly Haven / Sparks / Chimera rendered pairs",
        "corpora/rudra_mix_v4c_netflix_20261001": "manifest only, 49 % HdM",
        "corpora/rudra_mix_v4c_netflix_20261008": "manifest only, HdM",
        "corpora/rudra_mix_v4c_netflix_20261008b": "manifest only, 41 % HdM (v8, v9)",
        "corpora/rudra_mix_v10_20261013": "manifest only, 42 % HdM (v10)",
        "corpora/rudra_mix_v11_fxtd_20261017": "manifest only, HdM inherited from v10",
        "corpora/rudra_mix_v11b_fxtd_20261017": "manifest only, 39 % HdM",
        "corpora/temporal_clips_20261011": "3 of 5 clips HdM",
    },
}


def plan(root: Path, restore: bool = False) -> list[tuple[Path, Path, str]]:
    moves = []
    for rel, licence, _ in NON_COMMERCIAL:
        live, quarantined = root / rel, root / QUARANTINE / rel
        src, dst = (quarantined, live) if restore else (live, quarantined)
        if src.exists():
            if dst.exists():
                raise SystemExit(f"refusing: {dst} already exists")
            moves.append((src, dst, licence))
    return moves


def write_registry(root: Path) -> dict:
    reg = {"updated": time.strftime("%Y-%m-%d"), "root": str(root),
           "non_commercial_quarantined": {f"{QUARANTINE}/{rel}": f"{lic}; {why}"
                                          for rel, lic, why in NON_COMMERCIAL
                                          if (root / QUARANTINE / rel).exists()},
           **REGISTRY_STATIC,
           "rule": "Only manifests under corpora/rudra_commercial_* may train a model that is sold; "
                   "build them with pipeline/build_commercial_manifest.py. Not legal advice."}
    (root / "datasets_registry.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    lines = ["# G:\\datasets — what may train a commercial model", "",
             f"Updated {reg['updated']} by pipeline/organize_datasets.py. Not legal advice; see the repo's",
             "docs/LEGAL_QUESTIONS.md and docs/ai/RUDRA_DATASET_LICENSE_AUDIT.md.", ""]
    for section, title in (("commercial_clear", "Clear for commercial training"),
                           ("commercial_pending_counsel", "Pending counsel: do not ship weights trained on these"),
                           ("mixed_noncommercial_keep_in_place", "Mixed (non-commercial as a whole, left in place)"),
                           ("non_commercial_quarantined", "Non-commercial, quarantined (moved, not deleted)")):
        lines += [f"## {title}", ""] + [f"- `{k}`: {v}" for k, v in reg[section].items()] + [""]
    lines += ["To undo the quarantine: `python pipeline/organize_datasets.py --root G:/datasets --restore`", ""]
    (root / "DATASETS.md").write_text("\n".join(lines), encoding="utf-8")
    return reg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=Path)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--restore", action="store_true")
    args = ap.parse_args(argv)
    moves = plan(args.root, restore=args.restore)
    for src, dst, lic in moves:
        print(f"{'MOVE' if (args.apply or args.restore) else 'would move'}  {src}  ->  {dst}   ({lic})")
    if not (args.apply or args.restore):
        print(f"dry run: {len(moves)} moves; pass --apply")
        return 0
    for src, dst, _ in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)          # same volume: a rename, nothing copied or deleted
    reg = write_registry(args.root)
    print(f"done: {len(moves)} moved; registry lists {len(reg['non_commercial_quarantined'])} quarantined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
