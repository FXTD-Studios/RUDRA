"""Read-only: score a measured BT.1886 shadow correction against plain BT.1886.

Fit (training titles only): per 8-bit SDR code below the knee K, the median
target nits, each training title weighted equally, monotonic, blended back to
BT.1886 in log between K/2 and K. From K up, the curve is BT.1886 exactly.

Score (held out: Sparks val, Meridian test): full-frame PU21-PSNR per frame,
at the bench's resolution (area, long side 960). The crushed-region and
below-knee shadow errors are mean |log2 luminance| errors, as
training/bench_val.py. Positive gain favours the correction. Numpy/OpenCV
only. Never writes next to the media; never overwrites evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.dark_tone_targets import bt1886_baseline_nits, read, storage_for  # noqa: E402
from pipeline.hdr_io import decode_hdr_u16  # noqa: E402

FLOOR, PEAK = 0.005, 10000.0
LUMA_2020 = np.array([0.2627, 0.6780, 0.0593])
_PU21 = (0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142)
CRUSHED = 1.0 / 255.0
MIN_REGION = 256
KNEES = (20, 40, 80)
BLACK = [0.0]               # set by --black
PRIMARY_KNEE = 40          # declared before scoring (docs/ai/DARK_TONE_TARGETS.md: "below about code 40")


def pu21(nits):
    p0, p1, p2, p3, p4, p5, p6 = _PU21
    yp = np.clip(nits, FLOOR, PEAK) ** p3
    return p6 * (((p0 + p1 * yp) / (1.0 + p2 * yp)) ** p4 - p5)


PU_PEAK = float(pu21(np.array(PEAK)))


def pu_psnr(test, ref):
    mse = float(np.mean((pu21(test) - pu21(ref)) ** 2))
    return math.inf if mse <= 0 else 10.0 * math.log10(PU_PEAK ** 2 / mse)


def log_luma(nits):
    return np.log2(np.maximum((np.maximum(nits, 0.0) * LUMA_2020).sum(-1), FLOOR))


def region_error(test, ref, mask):
    if int(mask.sum()) < MIN_REGION:
        return None
    return float(np.abs(log_luma(test) - log_luma(ref))[mask].mean())


def load_pair(row, remap, max_side=960, stride=1):
    sdr_p, hdr_p = Path(remap(row["sdr_path"])), Path(remap(row["hdr_path"]))
    sdr = read(sdr_p)
    sdr = sdr.astype(np.float64) / float(np.iinfo(sdr.dtype).max)
    st = storage_for(hdr_p)
    hdr = decode_hdr_u16(read(hdr_p), st).astype(np.float64) * st.diffuse_white_nits
    if max_side:
        h, w = sdr.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            size = (max(1, round(w * s)), max(1, round(h * s)))
            sdr = cv2.resize(sdr, size, interpolation=cv2.INTER_AREA)
            hdr = cv2.resize(hdr, size, interpolation=cv2.INTER_AREA)
    return sdr[::stride, ::stride], np.maximum(hdr[::stride, ::stride], 0.0)


HIST_LO, HIST_HI, HIST_BINS = math.log2(FLOOR), 14.0, 2400      # 0.0091-stop bins


def fit_medians(rows, remap, max_code=max(KNEES), per_title=60, stride=4):
    """Per title, per code below max_code: median log2 target nits (histogram, one read pass)."""
    titles = sorted({r["title"] for r in rows})
    out = {}
    edges = np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for title in titles:
        group = [r for r in rows if r["title"] == title]
        idx = sorted(set(np.linspace(0, len(group) - 1, min(per_title, len(group))).round().astype(int)))
        hist = np.zeros((max_code, HIST_BINS), np.int64)
        for i in idx:
            sdr, hdr = load_pair(group[i], remap, max_side=0, stride=stride)
            code = np.rint(sdr * 255).astype(int).ravel()
            target = np.log2(np.maximum(hdr, FLOOR)).ravel()
            keep = code < max_code
            b = np.clip(((target[keep] - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS).astype(int), 0, HIST_BINS - 1)
            np.add.at(hist, (code[keep], b), 1)
        med = np.full(256, np.nan)
        for c in range(max_code):
            n = hist[c].sum()
            if n >= 200:
                med[c] = centers[np.searchsorted(np.cumsum(hist[c]), n / 2)]
        out[title] = med
    return out


def lut_from_medians(medians, knee, censor=False, min_step=0.0):
    """(256,) corrected nits per code; BT.1886 from the knee up.

    censor: drop a title's median within 0.05 stop of the storage floor (its
    HDR clipped there, so the median is a bound, not a value). min_step: the
    curve rises at least this many stops per code below the knee, so no two
    SDR codes share one HDR value (no posterising plateaus)."""
    floor = math.log2(FLOOR)
    stack = np.array([np.where(np.isfinite(m) & (~censor | (m > floor + 0.05)), m, np.nan) if censor else m
                      for m in medians.values()], dtype=float)
    with np.errstate(all="ignore"):
        pooled = np.nanmean(stack, axis=0)                 # equal title weight
    bt = np.log2(np.maximum(bt1886_baseline_nits(np.arange(256) / 255.0), 1e-12))
    lut = bt.copy()
    for c in range(knee):
        if np.isfinite(pooled[c]):
            w = 1.0 if c < knee / 2 else (knee - c) / (knee / 2)   # 1 -> 0 across [K/2, K)
            lut[c] = w * pooled[c] + (1 - w) * bt[c]
    lut[:knee] = np.maximum.accumulate(lut[:knee])
    lut[:knee] = np.minimum(lut[:knee], bt[knee])           # never above BT.1886 at the knee
    if min_step > 0:
        # Walk down from the knee: each code at least min_step below the next.
        for c in range(knee - 1, 0, -1):
            lut[c] = min(lut[c], lut[c + 1] - min_step)
    return np.exp2(lut)


def apply_lut(sdr01, lut_nits, knee):
    pos = np.clip(sdr01, 0.0, 1.0) * 255.0
    exact = bt1886_baseline_nits(np.clip(sdr01, 0.0, 1.0))
    below = np.exp2(np.interp(pos, np.arange(256), np.log2(np.maximum(lut_nits, 1e-12))))
    return np.where(pos >= knee, exact, below)


# --- The app's own path (native/core source_curve.cpp + baseline.cpp) -------
# A picker entry is 256 per-code log2 knots over the ACES inverse; the baseline
# is inverse_aces(srgb_to_linear(x)) * 2^(knots interpolated at x*255) * 203
# nits (corpus EV 0). Code 0 is 0 nits whatever the curve, and knot 0 copies
# knot 1, as build_params does. --app scores BOTH arms through this path.
ACES = (2.51, 0.03, 2.43, 0.59, 0.14)


def inverse_aces(y):
    a, b, c, d, e = ACES
    y = np.clip(y, 0.0, 0.995)
    qa, qb, qc = y * c - a, y * d - b, y * e
    disc = np.maximum(qb * qb - 4.0 * qa * qc, 0.0)
    den = np.minimum(2.0 * qa, -1e-7)
    return np.maximum(np.maximum((-qb - np.sqrt(disc)) / den, (-qb + np.sqrt(disc)) / den), 0.0)


def knots_for(scene_at_code):
    """build_params: log2(want / have) per integer code; knot 0 = knot 1."""
    display = srgb_eotf_np(np.arange(256) / 255.0)
    want, have = scene_at_code, inverse_aces(display)
    ok = (want > 0) & (have > 0)
    k = np.where(ok, np.log2(np.where(ok, want, 1.0) / np.where(ok, have, 1.0)), 0.0)
    k[0] = k[1]
    return k


def srgb_eotf_np(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def app_baseline(sdr01, knots):
    x = np.clip(sdr01, 0.0, 1.0)
    corr = np.interp(x * 255.0, np.arange(256), knots)
    return inverse_aces(srgb_eotf_np(x)) * np.exp2(corr) * 203.0


def bt1886_black_nits(code01, black):
    """ITU-R BT.1886 with a black level: L = a * max(V + b, 0)^2.4, white 1.0 =
    203 nits, Lb = black (relative to white). V is the video code, recovered as
    the plain entry does: bt709_oetf(srgb_to_linear(code)). black = 0 is plain BT.1886."""
    g = 2.4
    r = black ** (1.0 / g)
    a, b = (1.0 - r) ** g, r / (1.0 - r)
    v = bt709_oetf_np(np.clip(srgb_eotf_np(np.clip(code01, 0.0, 1.0)), 0.0, 0.995))
    return a * np.maximum(v + b, 0.0) ** g * 203.0


def fit_black(medians, codes=range(1, 40)):
    """One black level for all training titles: title-weighted least squares in
    log2 nits over codes 1-39. Medians within 0.05 stop of the storage floor are
    censored (the HDR clipped at FLOOR there) and left out."""
    floor = math.log2(FLOOR)
    obs = [(t, c, float(m[c])) for t, m in medians.items() for c in codes
           if m[c] is not None and np.isfinite(m[c]) and m[c] > floor + 0.05]
    per_title = {t: sum(1 for o in obs if o[0] == t) for t in medians}
    w = np.array([1.0 / per_title[t] for t, _, _ in obs])
    cs = np.array([c for _, c, _ in obs])
    ys = np.array([v for _, _, v in obs])
    grid = np.exp2(np.linspace(-20.0, -4.0, 1601))
    err = [float(np.sum(w * (np.log2(bt1886_black_nits(cs / 255.0, lb)) - ys) ** 2)) for lb in grid]
    i = int(np.argmin(err))
    per = {}
    for t in medians:
        sel = [(c, v) for tt, c, v in obs if tt == t]
        c2, y2 = np.array([c for c, _ in sel]), np.array([v for _, v in sel])
        e2 = [float(np.sum((np.log2(bt1886_black_nits(c2 / 255.0, lb)) - y2) ** 2)) for lb in grid]
        per[t] = float(grid[int(np.argmin(e2))])
    return {"black": float(grid[i]), "black_nits_at_203": float(grid[i] * 203.0),
            "rmse_stops": math.sqrt(err[i] / w.sum()), "observations": per_title, "per_title_black": per}


def bt709_oetf_np(x):
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.maximum(x, 0.0) ** 0.45 - 0.099)


def app_knots(lut_nits=None, knee=None):
    """BT.1886's knots, or the shadow entry's: the LUT below the knee, BT.1886 from it."""
    scene = bt1886_baseline_nits(np.arange(256) / 255.0) / 203.0
    if lut_nits is not None:
        scene = scene.copy()
        scene[1:knee] = np.asarray(lut_nits)[1:knee] / 203.0
        scene[0] = 0.0
    return knots_for(scene)


def score(rows, remap, luts, label, app=False):
    out = []
    started = time.time()
    for n, row in enumerate(rows):
        sdr, ref = load_pair(row, remap)
        base = app_baseline(sdr, app_knots()) if app else bt1886_baseline_nits(np.clip(sdr, 0.0, 1.0))
        top = sdr.max(-1)
        rec = {"asset_id": row["asset_id"], "scene_id": row["scene_id"],
               "hdr_mean_nits": row.get("hdr_mean_nits"), "bt1886_pu_db": pu_psnr(base, ref), "knees": {}}
        crushed = top <= CRUSHED
        eb_c = region_error(base, ref, crushed)
        for k, lut in luts.items():
            if k == "black":
                cor = app_baseline(sdr, app_knots(lut, 256)) if app else bt1886_black_nits(sdr, BLACK[0])
                shadow = top < PRIMARY_KNEE / 255.0
            else:
                cor = app_baseline(sdr, app_knots(lut, k)) if app else apply_lut(sdr, lut, k)
                shadow = top < k / 255.0
            eb_s, ec_s = region_error(base, ref, shadow), region_error(cor, ref, shadow)
            ec_c = region_error(cor, ref, crushed)
            rec["knees"][str(k)] = {"pu_gain_db": pu_psnr(cor, ref) - rec["bt1886_pu_db"],
                                    "shadow_fraction": float(shadow.mean()),
                                    "shadow_gain_stops": None if eb_s is None else eb_s - ec_s,
                                    "crushed_gain_stops": None if eb_c is None else eb_c - ec_c}
        out.append(rec)
        if n % 25 == 0:
            print(f"[{label}] {n + 1}/{len(rows)} {time.time() - started:.0f}s", flush=True)
    return out


def summarize(frames, knee):
    k = str(knee)
    g = [f["knees"][k]["pu_gain_db"] for f in frames]
    s = [f["knees"][k]["shadow_gain_stops"] for f in frames if f["knees"][k]["shadow_gain_stops"] is not None]
    c = [f["knees"][k]["crushed_gain_stops"] for f in frames if f["knees"][k]["crushed_gain_stops"] is not None]
    scenes = sorted({f["scene_id"] for f in frames})
    per_scene = {sc: float(np.mean([f["knees"][k]["pu_gain_db"] for f in frames if f["scene_id"] == sc])) for sc in scenes}
    sm = np.array(list(per_scene.values()))
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(sm, len(sm)))) for _ in range(2000)] if len(sm) > 1 else []
    dark = [f["knees"][k]["pu_gain_db"] for f in frames if (f["hdr_mean_nits"] or 1e9) < 10]
    return {"frames": len(g), "pu_gain_mean_db": float(np.mean(g)), "pu_gain_median_db": float(np.median(g)),
            "frames_improved": int(sum(v > 0 for v in g)), "frames_worse_by_0p1db": int(sum(v < -0.1 for v in g)),
            "worst_db": float(min(g)), "best_db": float(max(g)),
            "scene_mean_db": float(sm.mean()), "scene_ci95_db": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
            "scenes": len(sm), "per_scene_db": per_scene,
            "dark_frames": len(dark), "dark_pu_gain_mean_db": float(np.mean(dark)) if dark else None,
            "shadow_frames": len(s), "shadow_gain_mean_stops": float(np.mean(s)) if s else None,
            "crushed_frames": len(c), "crushed_gain_mean_stops": float(np.mean(c)) if c else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("fit", "fit-black", "relut", "score", "combine"))
    ap.add_argument("--fit-file", default="fit.json", help="fit file in --work the LUTs come from")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--map", nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--work", type=Path, required=True, help="stage directory (fit.json, score_*.json)")
    ap.add_argument("--set", choices=("sparks_bench48", "sparks_wide", "meridian_test"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--sparks-wide", type=int, default=200)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--app", action="store_true", help="score both arms through the app's knot path")
    ap.add_argument("--black", type=float, help="score BT.1886 with this black level (relative to white) instead of the LUTs")
    a = ap.parse_args(argv)
    tag = ("app_" if a.app else "") + (f"black{a.black:.3e}_" if a.black is not None else "") + \
        ("" if a.fit_file == "fit.json" else a.fit_file.removesuffix(".json") + "_")
    remap = (lambda p: a.map[1] + p[len(a.map[0]):].replace("\\", "/") if a.map and p.startswith(a.map[0]) else p)
    rows = [json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    a.work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if a.stage == "fit":
        target = a.work / "fit.json"
        if target.exists():
            raise SystemExit("fit.json exists")
        medians = fit_medians(train, remap)
        luts = {str(k): lut_from_medians(medians, k).tolist() for k in KNEES}
        with target.open("x") as h:
            json.dump({"fit_titles": sorted(medians), "median_log2_nits": {t: [None if not np.isfinite(v) else float(v) for v in m]
                       for t, m in medians.items()}, "lut_nits": luts, "seconds": time.time() - started}, h)
        print("fit", f"{time.time() - started:.0f}s")
        return
    fit = json.loads((a.work / ("fit.json" if a.stage == "relut" else a.fit_file)).read_text())
    if a.stage == "relut":
        target = a.work / a.fit_file
        if target.exists():
            raise SystemExit(f"{target} exists")
        med = {t: np.array([np.nan if v is None else v for v in m]) for t, m in fit["median_log2_nits"].items()}
        censor = "censor" in a.fit_file
        out = dict(fit, lut_rule=("censored floor medians, " if censor else "all medians, ") + "min step 0.02 stop/code",
                   lut_nits={str(k): lut_from_medians(med, k, censor=censor, min_step=0.02).tolist() for k in KNEES})
        with target.open("x") as h:
            json.dump(out, h)
        print("relut", target)
        return
    if a.stage == "fit-black":
        target = a.work / "black.json"
        if target.exists():
            raise SystemExit("black.json exists")
        med = {t: np.array([np.nan if v is None else v for v in m]) for t, m in fit["median_log2_nits"].items()}
        res = fit_black(med)
        with target.open("x") as h:
            json.dump(res, h, indent=1)
        print(json.dumps(res))
        return
    luts = {k: np.array(fit["lut_nits"][str(k)]) for k in KNEES}
    knees = KNEES
    if a.black is not None:
        BLACK[0] = a.black
        luts = {"black": bt1886_black_nits(np.arange(256) / 255.0, a.black)}
        knees = ("black",)
    real_val = [r for r in val if r.get("sdr_kind") == "real"]
    sets = {"sparks_bench48": [real_val[i] for i in sorted(np.random.default_rng(20261013).permutation(len(real_val))[:48])],
            "sparks_wide": [val[i] for i in sorted(set(np.linspace(0, len(val) - 1, a.sparks_wide).round().astype(int)))],
            "meridian_test": [r for r in rows if r["split"] == "test"]}
    if a.stage == "score":
        subset = sets[a.set][a.start:a.end]
        target = a.work / f"score_{tag}{a.set}_{a.start:04d}.json"
        if target.exists():
            raise SystemExit(f"{target} exists")
        frames = score(subset, remap, luts, a.set, app=a.app)
        with target.open("x") as h:
            json.dump(frames, h)
        print("scored", a.set, a.start, len(frames), f"{time.time() - started:.0f}s")
        return
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; evidence is never overwritten")
    result = {"tool": "tools/shadow_correction_eval.py", "app_path": a.app, "black": a.black, "fit_file": a.fit_file, "primary_knee": PRIMARY_KNEE, "knees": KNEES,
              "fit": fit, "bt1886_nits": bt1886_baseline_nits(np.arange(256) / 255.0).tolist(), "sets": {}}
    for name, subset in sets.items():
        frames = []
        for part in sorted(a.work.glob(f"score_{tag}{name}_*.json")):
            frames += json.loads(part.read_text())
        if [f["asset_id"] for f in frames] != [r["asset_id"] for r in subset]:
            raise SystemExit(f"{name}: scored frames do not cover the set exactly")
        result["sets"][name] = {"summary": {str(k): summarize(frames, k) for k in knees}, "frames": frames}
        s = result["sets"][name]["summary"][str(knees[0] if a.black is not None else PRIMARY_KNEE)]
        print(name, json.dumps({k: s[k] for k in ("frames", "pu_gain_mean_db", "pu_gain_median_db", "frames_improved",
                                                  "frames_worse_by_0p1db", "scene_ci95_db", "dark_pu_gain_mean_db",
                                                  "shadow_gain_mean_stops", "crushed_gain_mean_stops")}))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as h:
        json.dump(result, h)


if __name__ == "__main__":
    main()
