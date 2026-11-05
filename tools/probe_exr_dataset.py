r"""Read-only probe of an EXR footage folder, for dataset intake.

    .venv\Scripts\python.exe tools\probe_exr_dataset.py "Z:\path\to\EXR" [--frames 12]

Writes tmp\dataset_probe_<folder>.json next to this repo and prints a summary.
Never writes to the footage folder.
"""
from __future__ import annotations

import argparse, json, math, os, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEQ_RE = re.compile(r"^(.*?)(\d+)(\.exr)$", re.I)


def _plain(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return str(v)


def read_header(p: Path) -> dict:
    import OpenEXR
    if hasattr(OpenEXR, "File"):
        try:
            return {k: _plain(v) for k, v in OpenEXR.File(str(p), header_only=True).header().items()}
        except TypeError:
            pass
    return {k: _plain(v) for k, v in OpenEXR.InputFile(str(p)).header().items()}


def read_rgb(p: Path) -> tuple[np.ndarray, list[str]]:
    import OpenEXR
    if hasattr(OpenEXR, "File"):
        f = OpenEXR.File(str(p), separate_channels=True)
        ch = {k: v.pixels for k, v in f.channels().items()}
    else:
        import Imath
        f = OpenEXR.InputFile(str(p))
        dw = f.header()["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        ft = Imath.PixelType(Imath.PixelType.FLOAT)
        ch = {k: np.frombuffer(f.channel(k, ft), np.float32).reshape(h, w) for k in f.header()["channels"]}
    names = list(ch)
    if "RGB" in ch or "RGBA" in ch:  # interleaved
        a = ch.get("RGB", ch.get("RGBA"))
        return a[..., :3].astype(np.float32), names
    pick = []
    for c in "RGB":
        cand = [n for n in ch if n == c or n.endswith("." + c)]
        if not cand:
            raise RuntimeError(f"no {c} channel in {names}")
        pick.append(sorted(cand, key=len)[0])
    return np.stack([ch[n].astype(np.float32) for n in pick], -1), names


def frame_stats(rgb: np.ndarray) -> dict:
    finite = np.isfinite(rgb)
    bad = int((~finite).sum())
    x = np.where(finite, rgb, 0.0)
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    chmax = x.reshape(-1, 3).max(0)
    # plateau = pixels sitting within 0.5% of the channel max: sensor/grade clipping
    plateau = [float((x[..., c] >= chmax[c] * 0.995).mean()) if chmax[c] > 0 else 0.0 for c in range(3)]
    q = np.percentile(y, [0.1, 1, 50, 99, 99.9])
    return {
        "nonfinite": bad,
        "neg_frac": float((x < 0).any(-1).mean()),
        "min": float(x.min()),
        "chan_max": [float(v) for v in chmax],
        "Y_p0.1": float(q[0]), "Y_p1": float(q[1]), "Y_p50": float(q[2]),
        "Y_p99": float(q[3]), "Y_p99.9": float(q[4]), "Y_max": float(y.max()),
        "frac_Y_gt_1": float((y > 1.0).mean()),
        "plateau_frac_rgb": plateau,
        "dynamic_range_stops_p0.1_p99.9": float(math.log2(max(q[4], 1e-9) / max(q[0], 1e-9))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--frames", type=int, default=12, help="frames sampled per sequence")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.is_dir():
        print(f"not a folder: {root}"); return 2

    t0 = time.time()
    seqs: dict[str, list[Path]] = defaultdict(list)
    other = defaultdict(int)
    total_bytes = 0
    for dp, _, fs in os.walk(root):
        for fn in fs:
            p = Path(dp) / fn
            if fn.lower().endswith(".exr"):
                m = SEQ_RE.match(fn)
                key = str(Path(dp).relative_to(root) / (m.group(1) + "#" if m else fn))
                seqs[key].append(p)
                try: total_bytes += p.stat().st_size
                except OSError: pass
            else:
                other[Path(fn).suffix.lower() or "<none>"] += 1

    out = {"root": str(root), "n_sequences": len(seqs),
           "n_exr": sum(map(len, seqs.values())), "total_gb": round(total_bytes / 1e9, 2),
           "other_files": dict(other), "sequences": {}}
    print(f"{out['n_exr']} EXR in {len(seqs)} sequences, {out['total_gb']} GB")

    for key, files in sorted(seqs.items()):
        files.sort()
        nums = [int(m.group(2)) for f in files if (m := SEQ_RE.match(f.name))]
        gaps = (max(nums) - min(nums) + 1 - len(nums)) if nums else 0
        s = {"frames": len(files), "first": files[0].name, "last": files[-1].name,
             "frame_range": [min(nums), max(nums)] if nums else None, "missing_frames": gaps}
        try:
            h = read_header(files[0])
            s["header"] = {k: v for k, v in h.items() if k not in ("channels",)}
            s["channels"] = _plain(h.get("channels"))
        except Exception as e:  # noqa: BLE001
            s["header_error"] = repr(e)
        idx = sorted(set(np.linspace(0, len(files) - 1, min(a.frames, len(files))).round().astype(int)))
        samples = []
        for i in idx:
            try:
                rgb, names = read_rgb(files[i])
                st = frame_stats(rgb); st["file"] = files[i].name; st["shape"] = list(rgb.shape)
                samples.append(st)
            except Exception as e:  # noqa: BLE001
                samples.append({"file": files[i].name, "error": repr(e)})
        s["samples"] = samples
        ok = [x for x in samples if "error" not in x]
        if ok:
            s["summary"] = {
                "Y_max": max(x["Y_max"] for x in ok),
                "Y_p99.9_median": float(np.median([x["Y_p99.9"] for x in ok])),
                "Y_p50_median": float(np.median([x["Y_p50"] for x in ok])),
                "frac_Y_gt_1_median": float(np.median([x["frac_Y_gt_1"] for x in ok])),
                "plateau_max": max(max(x["plateau_frac_rgb"]) for x in ok),
                "neg_frac_max": max(x["neg_frac"] for x in ok),
                "nonfinite_total": sum(x["nonfinite"] for x in ok),
                "dr_stops_median": float(np.median([x["dynamic_range_stops_p0.1_p99.9"] for x in ok])),
            }
        out["sequences"][key] = s
        sm = s.get("summary", {})
        print(f"  {key}: {len(files)} fr, Ymax {sm.get('Y_max', float('nan')):.2f}, "
              f"p99.9 {sm.get('Y_p99.9_median', float('nan')):.2f}, "
              f"plateau {sm.get('plateau_max', float('nan')):.4f}, "
              f"{s.get('header', {}).get('compression', '?')}")

    out["elapsed_s"] = round(time.time() - t0, 1)
    dst = REPO / "tmp" / f"dataset_probe_{re.sub(r'[^A-Za-z0-9_-]+', '_', root.as_posix())[-80:]}.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nreport: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
