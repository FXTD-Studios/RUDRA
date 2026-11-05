"""Golden values for the reconstruction region (composite spec revision 2).

    python tools/emit_region_golden.py   # writes native/tests/golden/composite/region_gate.json

Numpy only (no torch, no checkpoint), from rudra/delivery/region.py, the oracle
native/tests/test_region.cpp checks core reconstruction_gate against. Deterministic.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.region import ReconstructionRegion, reconstruction_gate  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "composite" / "region_gate.json"

CASES = [
    ("off", ReconstructionRegion(), "all", 1.0, 1.0),
    ("tones_default", ReconstructionRegion(enabled=True), "all", 1.0, 1.0),
    ("tones_split_strength", ReconstructionRegion(enabled=True, highlight_from=0.85, highlight_softness=0.2,
                                                  highlight_strength=1.5, shadow_below=0.05, shadow_softness=0.0,
                                                  shadow_strength=0.25), "all", 0.8, 0.7),
    ("highlights_hard", ReconstructionRegion(enabled=True, highlight_from=0.9, highlight_softness=0.0),
     "highlights", 1.0, 1.0),
    ("band", ReconstructionRegion(enabled=True, band=True, band_low=0.6, band_high=0.95, band_softness=0.1,
                                  highlight_strength=1.2, shadow_strength=0.5), "all", 1.0, 1.0),
    ("band_shadows_mode", ReconstructionRegion(enabled=True, band=True, band_low=0.0, band_high=0.2,
                                               band_softness=0.04), "shadows", 1.3, 0.9),
]


def main() -> int:
    y = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    cases = []
    for name, r, mode, strength, sw in CASES:
        g = reconstruction_gate(y, sw, mode, strength, r)
        cases.append({"name": name, "region": asdict(r), "mode": mode, "strength": strength,
                      "shadow_weight": sw, "gate": [float(v) for v in g]})
    OUT.write_text(json.dumps({"y": [float(v) for v in y], "cases": cases}, indent=0) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
