"""Build the FROZEN run04 G1 stress manifest (22 Oct 2026).

Takes the validation rows of a frozen manifest, degrades each SDR frame with
GeneratorStyleAugment(p=1, fixed seed, default ops) and writes 16-bit PNGs plus
a manifest whose rows point at them (new sdr_path / sdr_sha256; HDR untouched,
`genstyle_ops` records what was applied). Then freeze it like any validation kit:

    python training/genstyle_stress_kit.py --manifest <run02 data>/manifest.jsonl --out G:/.../genstyle_stress_v1
    python training/production_eval_kit.py prepare --manifest G:/.../genstyle_stress_v1/manifest.jsonl --out G:/.../genstyle_stress_kit_v1

Build it BEFORE any run04 checkpoint is scored (run04_prereg.json G1); never
rebuild it with another seed after seeing scores. Synthetic degradation of real
validation frames -- a proxy for generator output, not generator output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 20261022


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(manifest: Path, out: Path, seed: int = SEED) -> dict:
    from training.generator_style import DEFAULT_OPS, GeneratorStyleAugment
    from training.sdr2hdr_dataset import load_rgb
    if out.exists():
        raise ValueError("Use a new directory; stress sets are immutable")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("split") == "val"]
    if not rows:
        raise ValueError("no validation rows")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise ValueError(f"leftover {partial}; rename it aside first")
    (partial / "sdr").mkdir(parents=True)
    aug = GeneratorStyleAugment(1.0, seed=seed)
    written, counts = [], {k: 0 for k in DEFAULT_OPS}
    for r in sorted(rows, key=lambda r: r["asset_id"]):
        if _digest(Path(r["sdr_path"])) != r["sdr_sha256"]:
            raise ValueError(f"SDR hash mismatch: {r['asset_id']}")
        sdr = load_rgb(r["sdr_path"], hdr=False)
        deg, ops = aug._one(sdr)
        name = "".join(c if c.isalnum() or c in "-_." else "_" for c in r["asset_id"]) + ".png"
        dst = partial / "sdr" / name
        code = np.clip(np.rint(deg * 65535.0), 0, 65535).astype(np.uint16)
        if not cv2.imwrite(str(dst), np.ascontiguousarray(code[..., ::-1])):
            raise RuntimeError(f"write failed: {dst}")
        final = out / "sdr" / name
        row = dict(r, sdr_path=str(final), sdr_sha256=_digest(dst), genstyle_ops=ops,
                   genstyle_source_sdr_sha256=r["sdr_sha256"])
        written.append(row)
        for k in ops:
            counts[k] += 1
    (partial / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in written), encoding="utf-8")
    receipt = dict(source_manifest=str(manifest), source_manifest_sha256=_digest(manifest), seed=seed,
                   ops=DEFAULT_OPS, frames=len(written), scenes=len({r["scene_id"] for r in written}),
                   op_counts=counts, code_sha256={p: _digest(ROOT / p) for p in
                                                  ("training/generator_style.py", "training/genstyle_stress_kit.py")},
                   purpose="run04 G1 frozen generator-style stress set (proxy, not generator output)",
                   release_authorized=False)
    (partial / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    partial.rename(out)
    return receipt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    print(json.dumps(build(a.manifest, a.out, a.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
