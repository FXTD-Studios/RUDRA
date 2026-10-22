"""Read-only: an HDR10 review kit for the opt-in BT.1886 shadows Source entry.

For chosen manifest frames, three panels side by side, each the whole frame:
the graded HDR reference | BT.1886 (default) | BT.1886 shadows, computed
through the app's knot path (tools/shadow_correction_eval.py app_baseline).
Baseline only: no clip fill, anchor or exposure, so only the curve differs.

Writes 16-bit PQ BT.2020 PNG stills, one HDR10 HEVC clip (4 s per frame,
PQ / BT.2020 tagged) and SDR preview PNGs with shadows pushed +4 EV so a
normal screen can see the black levels. Never writes next to the media.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.shadow_correction_eval import app_baseline, app_knots, load_pair  # noqa: E402

M1, M2, C1, C2, C3 = 2610 / 16384, 2523 / 4096 * 128, 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq(nits):
    y = np.clip(nits / 10000.0, 0.0, 1.0) ** M1
    return ((C1 + C2 * y) / (1.0 + C3 * y)) ** M2


def label(img, text, white):
    mask = np.zeros(img.shape[:2], np.uint8)       # putText draws 8-bit only
    cv2.putText(mask, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    a = (mask.astype(np.float64) / 255.0)[..., None]
    return np.round(img * (1 - a) + white * a).astype(img.dtype)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--fit", type=Path, required=True, help="fit file with lut_nits['40'] (the shipped table)")
    ap.add_argument("--assets", nargs="+", required=True)
    ap.add_argument("--map", nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty; never overwritten")
    remap = (lambda p: a.map[1] + p[len(a.map[0]):].replace("\\", "/") if a.map and p.startswith(a.map[0]) else p)
    rows = {r["asset_id"]: r for r in (json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip())}
    lut = json.loads(a.fit.read_text())["lut_nits"]["40"]
    k_plain, k_shadow = app_knots(), app_knots(lut, 40)
    a.out.mkdir(parents=True, exist_ok=True)
    frames, sheet = [], []
    for n, asset in enumerate(a.assets):
        sdr, ref = load_pair(rows[asset], remap, max_side=0)
        plain, shadow = app_baseline(sdr, k_plain), app_baseline(sdr, k_shadow)
        h, w = sdr.shape[:2]
        size = (a.panel_width, round(h * a.panel_width / w / 2) * 2)
        panels = [cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (ref, plain, shadow)]
        names = ("reference HDR", "BT.1886", "BT.1886 shadows")
        pq16 = [label(np.round(pq(p) * 65535).astype(np.uint16), t, int(pq(np.array(100.0)) * 65535)) for p, t in zip(panels, names)]
        trip = np.concatenate(pq16, axis=1)
        cv2.imwrite(str(a.out / f"{n:02d}_{asset}_pq.png"), cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))
        frames.append(trip)
        # SDR preview: +4 EV, BT.2020 treated as display primaries, simple clip, sRGB.
        prev = [np.clip(p / 203.0 * 16.0, 0, 1) ** (1 / 2.2) for p in panels]
        prev8 = [label(np.round(x * 255).astype(np.uint8), t + " (+4 EV)", 255) for x, t in zip(prev, names)]
        cv2.imwrite(str(a.out / f"{n:02d}_{asset}_preview_plus4ev.png"), cv2.cvtColor(np.concatenate(prev8, 1), cv2.COLOR_RGB2BGR))
        dark = (sdr.max(-1) < 40 / 255.0)
        luma = lambda x: (x * [0.2627, 0.6780, 0.0593]).sum(-1)
        sheet.append({"index": n, "asset_id": asset, "hdr_mean_nits": rows[asset].get("hdr_mean_nits"),
                      "below_code40_fraction": float(dark.mean()),
                      "median_nits_below_code40": {k: float(np.median(luma(v)[dark])) if dark.any() else None
                                                   for k, v in (("reference", ref), ("bt1886", plain), ("bt1886_shadows", shadow))}})
        print(n, asset, sheet[-1]["median_nits_below_code40"], flush=True)
    hgt = max(f.shape[0] for f in frames)
    import tempfile
    raw = Path(tempfile.gettempdir()) / f"shadow_review_{a.out.name}.rgb48"   # scratch, outside the kit
    with raw.open("wb") as fh:
        for f in frames:
            pad = np.zeros((hgt, f.shape[1], 3), np.uint16)
            pad[:f.shape[0]] = f
            for _ in range(4):        # 4 s at 1 fps
                fh.write(pad.astype("<u2").tobytes())
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le",
           "-s", f"{frames[0].shape[1]}x{hgt}", "-r", "1", "-i", str(raw),
           "-vf", "scale=out_color_matrix=bt2020nc:out_range=tv,format=yuv420p10le",
           "-c:v", "libx265", "-crf", "12", "-preset", "medium", "-r", "24",
           "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited:"
                           "hdr10=1:max-cll=1000,400:master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1)",
           "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-tag:v", "hvc1",
           str(a.out / "shadow_review_hdr10.mp4")]
    subprocess.run(cmd, check=True)
    raw.unlink(missing_ok=True)
    (a.out / "sheet.json").write_text(json.dumps(sheet, indent=1))


if __name__ == "__main__":
    main()
