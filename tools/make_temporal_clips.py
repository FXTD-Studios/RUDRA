"""SDR test clips for the temporal measurements, from HDR PQ frame sequences.

Real camera footage with consecutive frames (HdM HFR 2017 at 192 fps taken
every 8th frame, Netflix Chimera at 23.976) is mastered in PQ; this renders an
SDR version the way a delivery might look -- linearised, Hable tone-mapped to
BT.709, 8-bit H.264 4:2:0 -- so tools/measure_temporal.py has real grain,
real motion and real highlights to work on. The SDR is a tone map, not a
colourist's grade; that is said wherever the numbers are reported.

    python tools/make_temporal_clips.py G:/datasets/sources --out G:/datasets/corpora/temporal_clips_20261011
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

# name: (folder under sources, glob, primaries, first index, step, frames, fps)
CLIPS = {
    "hdm_bar_01": ("hdm_hfr_2017/Bar-Scene_HDR_Rec2100-PQ-1K_192fps_192f-handles_Shot-0001", "*/*.tif", "2020", 0, 8, 48, "24"),
    "hdm_fire_01": ("hdm_hfr_2017/Fire-Scene_HDR_Rec2100-PQ-1K_192fps_192f-handles_Shot-0001", "*.tif", "2020", 0, 8, 48, "24"),
    "hdm_fire_02": ("hdm_hfr_2017/Fire-Scene_HDR_Rec2100-PQ-1K_192fps_192f-handles_Shot-0002", "*.tif", "2020", 0, 8, 48, "24"),
    "chimera_a": ("netflix_chimera", "*.tif", "smpte432", 0, 1, 48, "24000/1001"),
    "chimera_b": ("netflix_chimera", "*.tif", "smpte432", 2000, 1, 48, "24000/1001"),
}


def build(sources: Path, out: Path, name: str, spec, width: int, height: int) -> Path:
    folder, pattern, primaries, first, step, count, fps = spec
    files = sorted((sources / folder).glob(pattern))[first::step][:count]
    if len(files) < count:
        raise SystemExit(f"{name}: {len(files)} frames, wanted {count}")
    target = out / f"{name}.mp4"
    # Every stage names its whole colourspace: zscale finds no path otherwise.
    vf = (f"zscale=tin=smpte2084:pin={primaries}:min=gbr:rin=full:t=linear:p={primaries}:m=gbr:npl=203,"
          "format=gbrpf32le,zscale=p=709,tonemap=tonemap=hable:desat=0,"
          "zscale=t=709:m=709:r=tv,format=yuv420p")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{width}x{height}",
                           "-r", fps, "-i", "pipe:0", "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
                           str(target)], stdin=subprocess.PIPE)
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        ff.stdin.write(np.ascontiguousarray(img[..., ::-1]).astype("<u2").tobytes())   # BGR -> RGB
    ff.stdin.close()
    if ff.wait():
        raise SystemExit(f"{name}: ffmpeg failed")
    return target


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", default="960x540")
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, spec in CLIPS.items():
        if args.only and name not in args.only:
            continue
        print(name, build(args.sources, args.out, name, spec, w, h), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
