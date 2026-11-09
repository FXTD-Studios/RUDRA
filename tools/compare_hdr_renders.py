#!/usr/bin/env python3
"""HDR renders of one SDR shot, from any tool, measured the same way (8 Nov 2026; in tools/ 9 Nov).

For one source movie and any number of HDR renders of it (EXR folders, ACEScg or ACES2065-1,
names that sort in frame order), the same numbers for each, so they can be compared:

  scale         the method's SDR white: the median, over mid-tone pixels (every channel between
                codes 50 and 180), of the render's AP1 luminance over the source's (BT.1886 2.4)
  round trip    the render through the ACES 2.0 SDR 100-nit view against the source codes, on
                pixels the source did not clip (max code < 250): share within 1 and 2 codes,
                median and 99th percentile; as delivered, then exposure-matched (each frame scaled
                so its mid-tones sit where the plain ACES 2.0 inverse puts them: a brighter or
                darker SDR white is forgiven, a re-rendered picture is not), then after the best
                per-channel curve (a code-by-code median map; the view mixes channels, so this
                cannot absorb everything)
  highlights    on clipped pixels (source max code >= 250): the render's luminance in SDR whites
                (over its scale) and in nits at 203 per white: median, p95; the frame's p99.9
  flicker       tools/measure_flicker.py's measure (36 x 64 cells, the render's change in stops
                minus the plain inverse's, per transition: mean and max), and on clipped pixels
                whose source is static to 2 codes: the median and p95 frame-to-frame |change| in EV
  bright band   on the picture frames, the region the source parks just under white (max code
                >= 235), eroded 12 px from its edges so silhouettes do not count: the render's
                luminance in SDR whites by source code bin (median), its hue (G/R, B/R of the
                mean) against the source's, and the structure inside it (the high-pass of log
                luminance, Gaussian sigma 6, in stops: its std, and its correlation with the
                source's own); a render whose interior structure does not correlate with the
                source's made it up (9 Nov 2026: a first reading of one tool's output got this
                wrong; the silhouettes' edges carried the correlation)
  pictures      chosen frames at -2.5 EV through the ACES 2.0 SDR view, every method normalised
                to the first method's scale, side by side: <out>/frame_<n>.png and index.html

    python tools/compare_hdr_renders.py --source shot.mp4 --out reports/compare_shot ^
        --method "other tool=renders/other/shot" --method "RUDRA analytic=renders/rudra/shot"

A render a few pixels smaller than the source (one tool delivers 1916 wide) is compared with the source
resized to it (area). Nothing is written into the renders. Needs ffmpeg, OpenEXR and OpenColorIO.
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.measure_flicker import LUMA_AP1, cell_luminance, read_master, temporal_change  # noqa: E402

CLIP_CODE = 250
NITS_PER_WHITE = 203.0
BRACKET_EV = -2.5
SAMPLE_STEP = 4          # every 4th pixel each way feeds the percentiles (memory)


def probe(movie: Path) -> dict:
    """Width, height, pixel format and range of the movie's first video stream."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,pix_fmt,color_range", "-of", "json", str(movie)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"][0]


def source_frames(movie: Path, count: int):
    """Yields frames 0..count-1 as display codes in [0, 1] (float32), decoded as the engine does
    (tools/measure_flicker.py decode_movie), one at a time."""
    s = probe(movie)
    w, h = int(s["width"]), int(s["height"])
    rng = "pc" if s.get("color_range") == "pc" else "tv"
    eight = not any(k in s.get("pix_fmt", "") for k in ("10", "12", "16"))
    vf = (f"scale=in_color_matrix=bt709:in_range={rng}:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb48le")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-i", str(movie), "-map", "0:v:0", "-vf", vf,
                          "-fps_mode", "passthrough", "-frames:v", str(count), "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    size = w * h * 3 * 2
    try:
        for _ in range(count):
            buf = p.stdout.read(size)
            if len(buf) < size:
                return
            codes = np.frombuffer(buf, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535.0
            yield np.rint(codes * 255.0) / 255.0 if eight else codes
    finally:
        p.stdout.close()
        p.wait()


def fit_to(codes: np.ndarray, shape) -> np.ndarray:
    """The source at the render's size (area), when they differ."""
    if codes.shape[:2] == shape[:2]:
        return codes
    import cv2
    return cv2.resize(codes, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def luma(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, np.float64) @ LUMA_AP1


def measure(name: str, folder: Path, movie: Path, view, frames: list[int], first_scale: float | None, out: Path):
    """Every number for one method; also writes its bracket pictures. -> (row, scale)."""
    exrs = sorted(folder.glob("*.exr"))
    if not exrs:
        return {"name": name, "error": f"no EXR in {folder}"}, None
    lim = 254.5 / 255.0
    rt_d, rt_m, rt_curve_src, rt_curve_dst, hi_rel, frame999, scale_samples = [], [], [], [], [], [], []
    flick, static_ev, shots, band_rows = [], [], {}, []
    prev = None
    for i, codes_full in enumerate(source_frames(movie, len(exrs))):
        rgb, _ = read_master(exrs[i])
        rgb = rgb.astype(np.float32)
        codes = fit_to(codes_full, rgb.shape)
        top = codes.max(-1)
        clipped = top >= CLIP_CODE / 255.0
        shown = np.clip(np.asarray(view.forward(rgb), np.float32), 0, 1)
        d = np.abs(shown - codes) * 255.0
        ss = (slice(None, None, SAMPLE_STEP), slice(None, None, SAMPLE_STEP))
        keep = ~clipped[ss]
        rt_d.append(d[ss].max(-1)[keep])
        base = np.asarray(view.inverse(codes.astype(np.float32)), np.float64)
        mid0 = np.all((codes > 50 / 255) & (codes < 180 / 255), -1)
        if mid0.any():   # exposure-matched: this frame's mid-tones where the plain inverse puts them
            k = float(np.median(luma(base[mid0]) / np.maximum(luma(rgb[mid0]), 1e-9)))
            shown_m = np.clip(np.asarray(view.forward((rgb * np.float32(k)).astype(np.float32)), np.float32), 0, 1)
            rt_m.append((np.abs(shown_m - codes) * 255.0)[ss].max(-1)[keep])
        rt_curve_src.append(np.rint(codes[ss][keep] * 255).astype(np.int16))
        rt_curve_dst.append(shown[ss][keep] * 255)
        # scale: render luminance over the source's light (BT.1886 2.4, SDR white = 1) on mid-tones
        mid = np.all((codes > 50 / 255) & (codes < 180 / 255), -1)
        if mid.any():
            ratio = luma(rgb[mid]) / np.maximum(luma(np.power(codes[mid], 2.4)), 1e-6)
            scale_samples.append(np.median(ratio))
        y = luma(rgb)
        hi_rel.append(y[ss][clipped[ss]])
        frame999.append(float(np.percentile(y, 99.9)))
        # flicker: measure_flicker's cells, and static clipped pixels
        cur = (cell_luminance(rgb), cell_luminance(base), codes, y)
        if prev is not None:
            t = temporal_change(prev[0], cur[0], prev[1], cur[1])
            if t["mean_abs_excess_ev"] is not None:
                flick.append(t["mean_abs_excess_ev"])
            static = clipped & (np.abs(codes - prev[2]).max(-1) * 255 <= 2) & (y > 0) & (prev[3] > 0)
            if static.any():
                static_ev.append(np.abs(np.log2(y[static]) - np.log2(prev[3][static]))[::SAMPLE_STEP])
        prev = cur
        if i + 1 in frames:
            shots[i + 1] = rgb
            band_rows.append(bright_band(rgb, codes))
    if prev is None:
        return {"name": name, "error": "no source frames decoded"}, None
    scale = float(np.median(scale_samples)) if scale_samples else float("nan")
    d = np.concatenate(rt_d)
    dm = np.concatenate(rt_m) if rt_m else np.zeros(0)
    # best per-channel curve: for each channel, the median shown code per source code
    src, dst = np.concatenate(rt_curve_src), np.concatenate(rt_curve_dst)
    resid = np.zeros_like(dst)
    for c in range(3):
        med = np.zeros(256, np.float32)
        for k in np.unique(src[:, c]):
            sel = src[:, c] == k
            med[k] = np.median(dst[sel, c])
        resid[:, c] = np.abs(dst[:, c] - med[src[:, c]])
    dc = resid.max(-1)
    hi = np.concatenate(hi_rel) / scale if hi_rel else np.zeros(0)
    st = np.concatenate(static_ev) if static_ev else np.zeros(0)
    pct = lambda a, q: float(np.percentile(a, q)) if a.size else None  # noqa: E731
    row = {
        "name": name, "folder": str(folder), "frames": len(exrs),
        "scale_sdr_white": scale,
        "round_trip": {"within_1_code": float((d <= 1.0).mean()), "within_2_codes": float((d <= 2.0).mean()),
                       "median_codes": pct(d, 50), "p99_codes": pct(d, 99)},
        "round_trip_exposure_matched": {"within_1_code": float((dm <= 1.0).mean()) if dm.size else None,
                                        "within_2_codes": float((dm <= 2.0).mean()) if dm.size else None,
                                        "median_codes": pct(dm, 50), "p99_codes": pct(dm, 99)},
        "round_trip_after_best_curve": {"within_1_code": float((dc <= 1.0).mean()), "within_2_codes": float((dc <= 2.0).mean()),
                                        "median_codes": pct(dc, 50), "p99_codes": pct(dc, 99)},
        "highlights": {"clipped_median_white": pct(hi, 50), "clipped_p95_white": pct(hi, 95),
                       "clipped_median_nits": None if not hi.size else pct(hi, 50) * NITS_PER_WHITE,
                       "clipped_p95_nits": None if not hi.size else pct(hi, 95) * NITS_PER_WHITE,
                       "frame_p999_white": float(np.median(frame999)) / scale},
        "flicker": {"cells_mean_ev": float(np.mean(flick)) if flick else None, "cells_max_ev": max(flick) if flick else None,
                    "clipped_static_median_ev": pct(st, 50), "clipped_static_p95_ev": pct(st, 95)},
        "bright_band": band_summary(band_rows, scale),
    }
    norm = (first_scale or scale) / scale
    for n, rgb in shots.items():
        pic = np.clip(np.asarray(view.forward((rgb * np.float32(norm * 2.0 ** BRACKET_EV)).astype(np.float32)), np.float32), 0, 1)
        to_png(out / f"frame_{n:03d}_{slug(name)}.png", pic)
    return row, scale


BAND_CODE = 235
BAND_BINS = ((235, 245), (245, 250), (250, 253), (253, 256))
BAND_ERODE_PX = 12
BAND_SIGMA_PX = 6


def bright_band(rgb: np.ndarray, codes: np.ndarray) -> dict | None:
    """One frame's bright band (see the module docstring): None when the frame has none."""
    import cv2
    top = codes.max(-1)
    band = (top >= BAND_CODE / 255.0).astype(np.uint8)
    k = 2 * BAND_ERODE_PX + 1
    inner = cv2.erode(band, np.ones((k, k), np.uint8)) > 0
    if inner.sum() < 500:
        return None
    y = luma(rgb)
    y_src = np.power(codes, 2.4) @ np.array([0.2126, 0.7152, 0.0722])   # the source's light, BT.1886
    def highpass(a):
        la = np.log2(np.maximum(a, 1e-6)).astype(np.float32)
        return la - cv2.GaussianBlur(la, (0, 0), BAND_SIGMA_PX)
    hr, hs = highpass(y)[inner], highpass(y_src)[inner]
    corr = float(np.corrcoef(hr, hs)[0, 1]) if hr.std() > 0 and hs.std() > 0 else None
    mean_rgb = rgb[inner].reshape(-1, 3).mean(0)
    src_lin = np.power(codes[inner].reshape(-1, 3), 2.4).mean(0)
    by_bin = {}
    cb = np.rint(top * 255)
    for lo, hi in BAND_BINS:
        sel = inner & (cb >= lo) & (cb < hi)
        by_bin[f"{lo}-{hi - 1}"] = float(np.median(y[sel])) if sel.sum() >= 100 else None
    return {"pixels": int(inner.sum()), "luma_by_code_median": by_bin,
            "hue_gr": float(mean_rgb[1] / max(mean_rgb[0], 1e-9)), "hue_br": float(mean_rgb[2] / max(mean_rgb[0], 1e-9)),
            "source_hue_gr": float(src_lin[1] / max(src_lin[0], 1e-9)), "source_hue_br": float(src_lin[2] / max(src_lin[0], 1e-9)),
            "structure_std_stops": float(hr.std()), "source_structure_std_stops": float(hs.std()), "structure_corr_source": corr}


def band_summary(rows: list, scale: float) -> dict | None:
    """The picture frames' bright bands together: luminance in SDR whites, the rest averaged."""
    rows = [r for r in rows if r]
    if not rows:
        return None
    mean = lambda key: float(np.mean([r[key] for r in rows if r[key] is not None])) if any(r[key] is not None for r in rows) else None  # noqa: E731
    bins = {}
    for name in rows[0]["luma_by_code_median"]:
        v = [r["luma_by_code_median"][name] for r in rows if r["luma_by_code_median"][name] is not None]
        bins[name] = float(np.median(v)) / scale if v else None
    return {"frames": len(rows), "pixels": int(np.mean([r["pixels"] for r in rows])), "luma_by_code_white": bins,
            "hue_gr": mean("hue_gr"), "hue_br": mean("hue_br"), "source_hue_gr": mean("source_hue_gr"), "source_hue_br": mean("source_hue_br"),
            "structure_std_stops": mean("structure_std_stops"), "source_structure_std_stops": mean("source_structure_std_stops"),
            "structure_corr_source": mean("structure_corr_source")}


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def to_png(path: Path, codes01: np.ndarray) -> None:
    """An 8-bit PNG of display codes in [0, 1]."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(np.rint(codes01 * 255), 0, 255).astype(np.uint8)[..., ::-1])


def report_md(shot: str, rows: list[dict]) -> str:
    """The comparison as a Markdown table."""
    ok = [r for r in rows if "error" not in r]
    f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
    p = lambda v: "n/a" if v is None else f"{100 * v:.1f} %"  # noqa: E731
    head = "| | " + " | ".join(r["name"] for r in ok) + " |\n|---|" + "---|" * len(ok) + "\n"
    lines = [
        ("SDR white (render units)", [f(r["scale_sdr_white"]) for r in ok]),
        ("Round trip: within 1 code", [p(r["round_trip"]["within_1_code"]) for r in ok]),
        ("Round trip: median / p99 (codes)", [f"{f(r['round_trip']['median_codes'], 1)} / {f(r['round_trip']['p99_codes'], 1)}" for r in ok]),
        ("Exposure-matched: within 1 code", [p(r["round_trip_exposure_matched"]["within_1_code"]) for r in ok]),
        ("Exposure-matched: median / p99", [f"{f(r['round_trip_exposure_matched']['median_codes'], 1)} / {f(r['round_trip_exposure_matched']['p99_codes'], 1)}" for r in ok]),
        ("After best curve: within 1 code", [p(r["round_trip_after_best_curve"]["within_1_code"]) for r in ok]),
        ("After best curve: median / p99", [f"{f(r['round_trip_after_best_curve']['median_codes'], 1)} / {f(r['round_trip_after_best_curve']['p99_codes'], 1)}" for r in ok]),
        ("Clipped: median (nits)", [f(r["highlights"]["clipped_median_nits"], 0) for r in ok]),
        ("Clipped: p95 (nits)", [f(r["highlights"]["clipped_p95_nits"], 0) for r in ok]),
        ("Frame p99.9 (SDR whites)", [f(r["highlights"]["frame_p999_white"]) for r in ok]),
        ("Flicker, cells: mean / max (EV)", [f"{f(r['flicker']['cells_mean_ev'], 4)} / {f(r['flicker']['cells_max_ev'], 4)}" for r in ok]),
        ("Flicker, static clipped: median / p95 (EV)", [f"{f(r['flicker']['clipped_static_median_ev'], 3)} / {f(r['flicker']['clipped_static_p95_ev'], 3)}" for r in ok]),
    ]
    bb = [r.get("bright_band") for r in ok]
    if any(bb):
        g = lambda b, key, d=2: "n/a" if not b or b.get(key) is None else f"{b[key]:.{d}f}"  # noqa: E731
        for name in BAND_BINS:
            key = f"{name[0]}-{name[1] - 1}"
            lines.append((f"Bright band, codes {key}: median (SDR whites)", ["n/a" if not b or b["luma_by_code_white"].get(key) is None else f"{b['luma_by_code_white'][key]:.2f}" for b in bb]))
        lines.append(("Bright band: hue G/R, B/R (source's)", [f"{g(b, 'hue_gr')}, {g(b, 'hue_br')} ({g(b, 'source_hue_gr')}, {g(b, 'source_hue_br')})" for b in bb]))
        lines.append(("Bright band: structure std, stops (source's)", [f"{g(b, 'structure_std_stops', 3)} ({g(b, 'source_structure_std_stops', 3)})" for b in bb]))
        lines.append(("Bright band: structure correlation with the source", [g(b, "structure_corr_source") for b in bb]))
    body = "".join(f"| {k} | " + " | ".join(v) + " |\n" for k, v in lines)
    errs = "".join(f"\n- {r['name']}: {r['error']}" for r in rows if "error" in r)
    return f"# {shot}: head to head\n\n{head}{body}{errs}\n"


def main(argv: list[str] | None = None) -> int:
    """Measure every method, write report.json, report.md and index.html."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="the SDR movie every render was made from")
    ap.add_argument("--method", action="append", required=True, help="name=EXR folder (repeat; the first sets the pictures' scale)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=[1, 48, 97], help="frames (1-based) for the pictures")
    ap.add_argument("--name", default="", help="the shot's name in the report")
    a = ap.parse_args(argv)
    from rudra.aces_view import ACES2View
    view = ACES2View(threads=8)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, first = [], None
    for spec in a.method:
        name, folder = spec.split("=", 1)
        print(f"measuring {name} ...", flush=True)
        row, scale = measure(name, Path(folder), a.source, view, a.frames, first, a.out)
        if first is None and scale:
            first = scale
        rows.append(row)
        print(json.dumps(row, indent=1), flush=True)
    shot = a.name or a.source.stem
    (a.out / "report.json").write_text(json.dumps({"schema": "rudra_compare_hdr_renders_v2", "shot": shot,
                                                   "source": str(a.source), "bracket_ev": BRACKET_EV, "methods": rows}, indent=1),
                                       encoding="utf-8")
    md = report_md(shot, rows)
    (a.out / "report.md").write_text(md, encoding="utf-8")
    names = [r["name"] for r in rows if "error" not in r]
    cells = "".join(
        f"<tr><th>frame {n}</th>" + "".join(
            f'<td><a href="frame_{n:03d}_{slug(m)}.png"><img src="frame_{n:03d}_{slug(m)}.png"></a></td>' for m in names) + "</tr>"
        for n in a.frames)
    (a.out / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>{html.escape(shot)}: head to head</title>
<style>body{{font:14px system-ui;margin:16px;background:#111;color:#ddd}}img{{width:420px;display:block}}td,th{{padding:4px}}
pre{{white-space:pre-wrap}}</style><h1>{html.escape(shot)}</h1>
<p>Every render at {BRACKET_EV} EV through the ACES 2.0 SDR 100-nit view, normalised to the first method's SDR white.</p>
<table><tr><th></th>{''.join(f'<th>{html.escape(m)}</th>' for m in names)}</tr>{cells}</table>
<pre>{html.escape(md)}</pre>""", encoding="utf-8")
    print(md)
    print(f"-> {a.out / 'index.html'}")
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
