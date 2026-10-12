"""SDR test clips for the temporal measurements, from HDR PQ frame sequences.

Real camera footage with consecutive frames (HdM HFR 2017 at 192 fps taken
every 8th frame, Netflix Chimera at 23.976) is mastered in PQ; this renders an
SDR version the way a delivery might look -- linearised, Hable tone-mapped to
BT.709, 8-bit H.264 4:2:0 -- so tools/measure_temporal.py has real grain,
real motion and real highlights to work on. The SDR is a tone map, not a
colourist's grade; that is said wherever the numbers are reported.

    python tools/make_temporal_clips.py G:/datasets/sources --out G:/datasets/corpora/temporal_clips_20261011

--reference also writes each clip's HDR as the reference the clip-mode CVVDP of
tools/measure_temporal.py scores against: <name>_ref/<nnn>.png, 16-bit PQ in
BT.2020 at the clip's size, resized exactly as the SDR's source was (P3 sources
are carried to BT.2020 in linear light).
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


PQ_M1, PQ_M2, PQ_C1, PQ_C2, PQ_C3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
# SMPTE RP 431-2 / EG 432-1 (P3-D65) to BT.2020, linear light.
P3D65_TO_2020 = np.array([[0.7538330344, 0.1985973691, 0.0475695966],
                          [0.0457438490, 0.9417772198, 0.0124789312],
                          [-0.0012103404, 0.0176017173, 0.9836086231]])


def pq_decode(e: np.ndarray) -> np.ndarray:
    p = np.power(np.clip(e, 0.0, 1.0), 1.0 / PQ_M2)
    return np.power(np.maximum(p - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * p), 1.0 / PQ_M1)


def pq_encode(y: np.ndarray) -> np.ndarray:
    p = np.power(np.clip(y, 0.0, 1.0), PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p), PQ_M2)


def reference(sources: Path, out: Path, name: str, spec, width: int, height: int) -> Path:
    folder, pattern, primaries, first, step, count, fps = spec
    files = sorted((sources / folder).glob(pattern))[first::step][:count]
    target = out / f"{name}_ref"
    target.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        dst = target / f"{i:03d}.png"
        if dst.exists():
            continue
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)   # as build() does
        if primaries != "2020":
            rgb = pq_decode(img[..., ::-1].astype(np.float64) / 65535.0)
            rgb = pq_encode(np.clip(np.einsum("ij,hwj->hwi", P3D65_TO_2020, rgb), 0.0, None))
            img = np.ascontiguousarray((np.rint(rgb * 65535.0)).astype(np.uint16)[..., ::-1])
        cv2.imwrite(str(dst), img)
    return target


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", default="960x540")
    p.add_argument("--only", nargs="*")
    p.add_argument("--reference", action="store_true", help="also write each clip's HDR reference (<name>_ref/)")
    p.add_argument("--no-clips", action="store_true", help="with --reference: the references only")
    args = p.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, spec in CLIPS.items():
        if args.only and name not in args.only:
            continue
        if not args.no_clips:
            print(name, build(args.sources, args.out, name, spec, w, h), flush=True)
        if args.reference:
            print(name, reference(args.sources, args.out, name, spec, w, h), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
