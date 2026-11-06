"""Where a master's extreme highlights and its failing pixels come from (3 Nov 2026).

For one EXR master and the source frame it was made from, prints:
  - the brightest pixels (AP1 luminance, in master units, 1.0 = diffuse white);
  - every pixel whose ACES 2.0 SDR view is more than --tol codes from the source;
and for each: the source's 8-bit codes, the master, the plain ACES inverse of
those codes, and the round-trip error. Decodes as the engine does
(tools/measure_flicker.py), reads AP0 masters as AP1.

    python tools/inspect_highlights.py --clip D:/flicker_clips/09V.mp4 --frame 45 ^
        --master G:/rudra_training/flicker_clips_20261103/09V/09V_000045.exr --out 09V_45.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.measure_flicker import LUMA_AP1, decode_frames, decode_movie, read_master  # noqa: E402


def inspect(codes: np.ndarray, master: np.ndarray, view, top: int = 12, tol: float = 1.0) -> dict:
    plain = np.asarray(view.inverse(np.clip(codes, 0, 1).astype(np.float32)), np.float64)
    back = np.asarray(view.forward(np.nan_to_num(master, nan=0.0, posinf=1e6).astype(np.float32)), np.float64)
    err = np.abs(back - codes).max(-1) * 255.0
    lum = master @ LUMA_AP1
    plum = plain @ LUMA_AP1

    def row(y, x):
        return {"x": int(x), "y": int(y), "source_codes": [int(round(v * 255)) for v in codes[y, x]],
                "master": [round(float(v), 3) for v in master[y, x]], "master_luma": round(float(lum[y, x]), 3),
                "plain": [round(float(v), 3) for v in plain[y, x]], "plain_luma": round(float(plum[y, x]), 3),
                "master_over_plain_stops": round(float(np.log2(max(lum[y, x], 1e-9) / max(plum[y, x], 1e-9))), 2),
                "roundtrip_error_codes": round(float(err[y, x]), 3)}

    order = np.argsort(lum, axis=None)[::-1][:top]
    brightest = [row(*np.unravel_index(i, lum.shape)) for i in order]
    fy, fx = np.nonzero(err > tol)
    failing = [row(y, x) for y, x in list(zip(fy, fx))[:40]]
    return {"master_peak_luma": float(lum.max()), "plain_peak_luma": float(plum.max()),
            "master_nonfinite": int((~np.isfinite(master)).sum()),
            "failing_pixels": int((err > tol).sum()), "tolerance_codes": tol,
            "brightest": brightest, "failing": failing,
            "failing_are_among_brightest_1pct": (None if not len(fy) else
                                                 float(np.mean(lum[fy, fx] >= np.quantile(lum, 0.99))))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    from rudra.aces_view import ACES2View
    codes = (decode_frames(a.clip, a.frame, 1) if a.clip.is_dir() else decode_movie(a.clip, a.frame, 1))[0]
    master, _ = read_master(a.master)
    r = inspect(codes, master, ACES2View(threads=4), tol=a.tol)
    print(f"master peak luma {r['master_peak_luma']:.2f} (x203 = {r['master_peak_luma'] * 203:,.0f} nits)   "
          f"plain inverse peak {r['plain_peak_luma']:.2f}   non-finite {r['master_nonfinite']}")
    print(f"failing pixels (> {a.tol} code): {r['failing_pixels']}   "
          f"share of them in the brightest 1 %: {r['failing_are_among_brightest_1pct']}")
    print("\nbrightest:   source codes      master luma   plain luma   master/plain stops   round-trip codes")
    for b in r["brightest"]:
        print(f"  ({b['x']:4d},{b['y']:4d})  {str(b['source_codes']):16s} {b['master_luma']:11.2f} {b['plain_luma']:11.2f} "
              f"{b['master_over_plain_stops']:12.2f} {b['roundtrip_error_codes']:16.3f}")
    if r["failing"]:
        print("\nfailing:     source codes      master luma   plain luma   master/plain stops   round-trip codes")
        for b in r["failing"][:15]:
            print(f"  ({b['x']:4d},{b['y']:4d})  {str(b['source_codes']):16s} {b['master_luma']:11.2f} {b['plain_luma']:11.2f} "
                  f"{b['master_over_plain_stops']:12.2f} {b['roundtrip_error_codes']:16.3f}")
    if a.out:
        a.out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
