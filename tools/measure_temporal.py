"""Temporal consistency on real footage (STATUS row 5, 11 Oct 2026).

Runs every clip through `rudra video` three ways -- each frame its own
(--temporal-smoothing 0 --static-hold off), the shot hold alone
(--static-hold off) and the shot hold with the static hold (the defaults) --
and tabulates the flicker measures each report carries: `flicker_ev` (the EV
the HDR's average moved beyond the source's) and `local_flicker_ev` (the same
per cell of a 36 x 64 grid), mean and worst frame, and the change against
"each frame its own". The masters are thrown away; the reports are kept.

    # the native CLI with a model package (what the app ships):
    python tools/measure_temporal.py CLIPS... --native path/to/rudra-native --package dist/models/sdr2hdr_shadow_v1
    # analytic (the app's default: no model), with the native CLI:
    python tools/measure_temporal.py CLIPS... --native path/to/rudra-native --analytic
    # or the Python reference with a checkpoint:
    python tools/measure_temporal.py CLIPS... --checkpoint checkpoints/sdr2hdr_shadow_v1.pt

CLIPS are files or folders (every .mp4 .mov .mkv .mxf in them). Writes
<out>/temporal.json and <out>/temporal.md (default out: reports/temporal_<date>).

--reference DIR (12 Oct 2026, STATUS row 5) also scores each master against the
clip's HDR (DIR/<clip>_ref/*.png, 16-bit PQ BT.2020, as
tools/make_temporal_clips.py --reference writes them) with ColorVideoVDP twice:
as a clip (its temporal model sees flicker) and frame by frame (it cannot). The
row 5 condition is clip JOD with the holds >= clip JOD of each frame its own.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CONFIGS = (
    ("each frame", ["--temporal-smoothing", "0", "--static-hold", "off"]),
    ("shot hold", ["--static-hold", "off"]),
    ("shot + static hold", []),
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".mxf"}


def clips_of(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO_SUFFIXES)
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    if not out:
        raise SystemExit("no clips")
    return out


def command(args, clip: Path, output: Path, flags: list[str]) -> list[str]:
    common = ["--output", str(output), "--format", "hdr10", "--audio", "none", "--preset", "ultrafast",
              "--dynamic-metadata", "off", *flags, *args.extra]
    if args.native:
        grade = ["--grade", str(args.grade)] if args.grade else []
        return [str(args.native), "video", str(args.package or "-"), str(clip), *common, *grade]
    return [sys.executable, "-m", "rudra.video", str(clip), "--checkpoint", str(args.checkpoint), *common]


def measure(args, clip: Path, work: Path) -> list[dict]:
    rows = []
    for i, (name, flags) in enumerate(CONFIGS):
        output = work / f"{clip.stem}_{i}.mp4"
        started = time.monotonic()
        run = subprocess.run(command(args, clip, output, flags), capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(f"{clip.name} ({name}) failed:\n{run.stderr[-3000:]}")
        report = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
        t = report["temporal"]
        jods = cvvdp_scores(output, args.reference / f"{clip.stem}_ref", report) if args.reference else {}
        rows.append({**jods, "clip": clip.name, "config": name, "frames": len(report["frames"]),
                     "shots": len(report["scene_cuts"]), "retention": t["retention"],
                     "flicker_ev_mean": t["flicker_ev_mean"], "flicker_ev_max": t["flicker_ev_max"],
                     "local_flicker_ev_mean": t["local_flicker_ev_mean"], "local_flicker_ev_max": t["local_flicker_ev_max"],
                     "seconds": round(time.monotonic() - started, 2)})
        output.unlink(missing_ok=True)
    return rows


def pq_to_nits(code):
    import numpy as np
    m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
    p = np.power(np.clip(code, 0.0, 1.0), 1.0 / m2)
    return 10000.0 * np.power(np.maximum(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1)


def cvvdp_scores(master: Path, ref_dir: Path, report: dict) -> dict:
    """ColorVideoVDP of the HDR10 master against the clip's HDR: as a clip and per frame."""
    import cv2
    import numpy as np
    import torch
    repo = str(Path(__file__).resolve().parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from rudra.hdrvdp import hdr_vdp3_clip_jod, hdr_vdp3_jod
    refs = sorted(ref_dir.glob("*.png"))
    if not refs:
        raise SystemExit(f"no reference frames in {ref_dir} (tools/make_temporal_clips.py --reference)")
    ref = np.stack([cv2.imread(str(f), cv2.IMREAD_UNCHANGED)[..., ::-1] for f in refs]).astype(np.float64) / 65535.0
    n, h, w = ref.shape[:3]
    # The master's PQ codes as full-range RGB in BT.2020 (the master's own primaries and transfer).
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(master), "-vf",
                          "zscale=min=2020_ncl:rin=limited:tin=smpte2084:pin=2020:m=gbr:r=full:t=smpte2084:p=2020,format=gbrp16le,format=rgb48le", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    test = np.frombuffer(raw, dtype="<u2").reshape(-1, h, w, 3).astype(np.float64) / 65535.0
    n = min(n, test.shape[0])
    white = 203.0
    to_t = lambda a: torch.from_numpy((pq_to_nits(a[:n]) / white).astype(np.float32)).permute(0, 3, 1, 2)
    t, r = to_t(test), to_t(ref)
    from fractions import Fraction
    fps = float(Fraction(str(report.get("timing", {}).get("fps") or "24")))
    clip, backend = hdr_vdp3_clip_jod(t, r, frames_per_second=fps, diffuse_white_nits=white)
    frames = [hdr_vdp3_jod(t[i:i + 1], r[i:i + 1], diffuse_white_nits=white)[0] for i in range(n)]
    return {"cvvdp_clip_jod": float(clip), "cvvdp_frame_jod": float(np.mean(frames)), "cvvdp_backend": backend,
            "cvvdp_fps": fps}


def change(value, against):
    if value is None or not against:
        return ""
    pct = round(100.0 * (value - against) / against)
    return "0 %" if pct == 0 else f"{pct:+d} %"


def markdown(rows: list[dict], how: str) -> str:
    jod = any("cvvdp_clip_jod" in r for r in rows)
    lines = [f"# Temporal consistency on footage ({datetime.date.today().isoformat()})", "",
             f"Run with {how}. Flicker in EV beyond the source's own change; local is per 36 x 64 cell. "
             "Change is against each frame its own.", "",
             "| clip | frames | shots | config | flicker mean | max | local mean | max | local change | s |"
             + (" CVVDP clip | per frame |" if jod else ""),
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|" + ("---:|---:|" if jod else "")]
    fmt = lambda v: "n/a" if v is None else f"{v:.5f}"
    own = {}
    for r in rows:
        if r["config"] == "each frame":
            own[r["clip"]] = r
        base = own.get(r["clip"], {})
        lines.append(f"| {r['clip']} | {r['frames']} | {r['shots']} | {r['config']} | {fmt(r['flicker_ev_mean'])} | "
                     f"{fmt(r['flicker_ev_max'])} | {fmt(r['local_flicker_ev_mean'])} | {fmt(r['local_flicker_ev_max'])} | "
                     f"{change(r['local_flicker_ev_mean'], base.get('local_flicker_ev_mean')) if r is not base else ''} | "
                     f"{r['seconds']} |"
                     + (f" {r.get('cvvdp_clip_jod', float('nan')):.3f} | {r.get('cvvdp_frame_jod', float('nan')):.3f} |"
                        if jod else ""))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="+", type=Path)
    p.add_argument("--native", type=Path, help="rudra-native; with --package")
    p.add_argument("--package", type=Path)
    p.add_argument("--checkpoint", type=Path, help="the Python reference instead of --native")
    p.add_argument("--analytic", action="store_true", help="--native only: the analytic reconstruction (no model)")
    p.add_argument("--grade", type=Path, help="--native only: an export settings file (the app's <file>.settings.json)")
    p.add_argument("--out", type=Path, default=Path("reports") / f"temporal_{datetime.date.today():%Y-%m-%d}")
    p.add_argument("--reference", type=Path, help="score each master with CVVDP against <DIR>/<clip>_ref/*.png")
    p.add_argument("--extra", default="", help='more flags for every run, quoted: --extra "--peak-nits 2000"')
    args = p.parse_args(argv)
    args.extra = args.extra.split()
    if bool(args.native) == bool(args.checkpoint) or (args.native and not (args.package or args.analytic or args.grade)):
        p.error("give --native with --package (or --analytic / --grade), or --checkpoint")
    if (args.analytic or args.grade) and not args.native:
        p.error("--analytic and --grade need --native")
    rows = []
    with tempfile.TemporaryDirectory(prefix="rudra-temporal-") as work:
        if args.analytic and not args.grade:
            args.grade = Path(work) / "analytic.settings.json"
            args.grade.write_text('{"analytic": true}', encoding="utf-8")
        for clip in clips_of(args.clips):
            print(f"{clip.name} ...", flush=True)
            rows += measure(args, clip, Path(work))
    args.out.mkdir(parents=True, exist_ok=True)
    how = (("rudra-native, analytic" if args.analytic else f"rudra-native ({args.package.name})") if args.native
           else f"rudra video ({args.checkpoint.name})")
    (args.out / "temporal.json").write_text(json.dumps(rows, indent=2), encoding="utf-8", newline="\n")
    (args.out / "temporal.md").write_text(markdown(rows, how), encoding="utf-8", newline="\n")
    print(markdown(rows, how))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
