"""Does a master keep the SDR's colour? The colour gate of the shot check (1.3.1, 25 Nov 2026).

Per SDR code range (the largest channel's code), the master's Oklab saturation (chroma over lightness,
which no exposure moves) and hue against the SDR read through its Source curve:

    shadows 10-50, mid-tones 50-180, bright 180-250   (8-bit codes; 250+ is the reconstruction's own)

    saturation  median(C/L) of the master over the SDR's, minus 1     gate: within +-10 %
    hue         median |hue difference|, pixels with SDR C/L >= 0.05  gate: <= 3 degrees

On 04.mp4 (24 Nov 2026) 1.3.0's analytic BT.1886 master measured -38 % in the shadows and -16 % in the
mid-tones (the colour carry read the SDR as sRGB; native core/master.cpp, fixed in 1.3.1): a FAIL here.
The learned ACES 2.0 route undoes ACES 2.0's look on a source not made through it (-35 % shadows,
+86 % bright on 04), so on such shots its colour fails by design of the model, not of the code.

    python tools/colour_gate.py --frames <PNG stills> --master <EXR folder> [--source bt1886|srgb] [--out c.json]

numpy and OpenCV only (and rudra.delivery.colorspace for the primaries).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.colorspace import rgb_to_rgb_matrix  # noqa: E402

RANGES = {"shadows": (10, 50), "mid": (50, 180), "bright": (180, 250)}
SAT_TOLERANCE = 0.10
HUE_TOLERANCE_DEG = 3.0
MIN_PIXELS = 500
MIN_SDR_SAT = 0.05   # below this the SDR's hue is noise

# Björn Ottosson's Oklab, from linear Rec.709 (D65).
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab(rec709_linear: np.ndarray) -> np.ndarray:
    lms = np.maximum(rec709_linear.astype(np.float64) @ _M1.T, 0.0)
    return np.cbrt(lms) @ _M2.T


def sdr_light(codes: np.ndarray, source: str) -> np.ndarray:
    """The SDR's display light (white 1.0) through its Source curve."""
    c = np.clip(codes.astype(np.float64), 0.0, 1.0)
    if source == "bt1886":
        return c ** 2.4
    if source == "srgb":
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    if source == "gamma22":
        return c ** 2.2
    raise ValueError(f"source is bt1886, srgb or gamma22, not {source!r}")


def measure(codes: np.ndarray, master: np.ndarray, primaries: str = "ap1", source: str = "bt1886") -> dict:
    """Saturation change and hue difference per code range for one frame.

    ``codes``: H x W x 3 SDR codes (0..1, Rec.709); ``master``: H x W x 3 scene-linear in ``primaries``.
    Returns per range the pixel counts and the per-pixel values the summary takes medians of."""
    if codes.shape != master.shape:
        raise ValueError(f"shape mismatch: {codes.shape} vs {master.shape}")
    m709 = master.astype(np.float64) @ rgb_to_rgb_matrix(primaries, "rec709").T if primaries != "rec709" else master
    lab_s = oklab(sdr_light(codes, source)).reshape(-1, 3)
    lab_m = oklab(m709).reshape(-1, 3)
    top = (codes.max(-1) * 255.0).reshape(-1)
    out = {}
    for name, (lo, hi) in RANGES.items():
        sel = (top >= lo) & (top < hi) & (lab_s[:, 0] > 1e-4) & (lab_m[:, 0] > 1e-4)
        cs = np.hypot(lab_s[sel, 1], lab_s[sel, 2]) / lab_s[sel, 0]
        cm = np.hypot(lab_m[sel, 1], lab_m[sel, 2]) / lab_m[sel, 0]
        hued = cs >= MIN_SDR_SAT
        hs = np.arctan2(lab_s[sel, 2], lab_s[sel, 1])[hued]
        hm = np.arctan2(lab_m[sel, 2], lab_m[sel, 1])[hued]
        dh = np.degrees(np.abs(np.angle(np.exp(1j * (hm - hs)))))
        out[name] = {"sat_sdr": cs, "sat_master": cm, "hue_deg": dh}
    return out


def summarise(frames: list[dict]) -> dict:
    """Medians over every measured frame, and the gate."""
    r: dict = {"ranges": {}, "tolerance": {"saturation": SAT_TOLERANCE, "hue_deg": HUE_TOLERANCE_DEG}}
    ok = []
    for name in RANGES:
        ss = np.concatenate([f[name]["sat_sdr"] for f in frames]) if frames else np.zeros(0)
        sm = np.concatenate([f[name]["sat_master"] for f in frames]) if frames else np.zeros(0)
        dh = np.concatenate([f[name]["hue_deg"] for f in frames]) if frames else np.zeros(0)
        if ss.size < MIN_PIXELS:
            r["ranges"][name] = {"pixels": int(ss.size), "pass": None}
            continue
        sat = float(np.median(sm) / max(np.median(ss), 1e-9) - 1.0)
        hue = float(np.median(dh)) if dh.size else 0.0
        p = abs(sat) <= SAT_TOLERANCE and hue <= HUE_TOLERANCE_DEG
        r["ranges"][name] = {"pixels": int(ss.size), "saturation_change": round(sat, 4),
                             "hue_median_deg": round(hue, 2),
                             "hue_p90_deg": round(float(np.percentile(dh, 90)), 2) if dh.size else 0.0,
                             "pass": bool(p)}
        ok.append(p)
    r["pass"] = None if not ok else bool(all(ok))
    return r


def main(argv: list[str] | None = None) -> int:
    import cv2

    tmp_gates = REPO / "tmp" / "09V_test" / "v4_gates"
    if str(tmp_gates) not in sys.path:
        sys.path.insert(0, str(tmp_gates))
    from shot_gates import read_exr  # the shot check's own EXR reader (AP0 masters come back as AP1)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--source", default="bt1886", choices=["bt1886", "srgb", "gamma22"])
    ap.add_argument("--primaries", default="", help="the master's primaries (default: from the EXR)")
    ap.add_argument("--samples", type=int, default=9)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    pngs = sorted(Path(a.frames).glob("*.png"))
    exrs = sorted(Path(a.master).glob("*.exr"))
    n = min(len(pngs), len(exrs))
    if n < 1:
        print(f"need stills and masters (got {len(pngs)}, {len(exrs)})", file=sys.stderr)
        return 2
    frames = []
    for k in sorted({int(k) for k in np.linspace(0, n - 1, min(a.samples, n))}):
        im = cv2.imread(str(pngs[k]), cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]
        codes = im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
        rgb, _, prim = read_exr(exrs[k], with_primaries=True)
        frames.append(measure(codes, rgb, a.primaries or prim, a.source))
    r = summarise(frames)
    r.update({"frames_measured": len(frames), "source": a.source})
    print(json.dumps(r, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
