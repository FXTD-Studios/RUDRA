r"""Read-only survey of a project folder for dataset intake (non-EXR media).

    .venv\Scripts\python.exe tools\survey_folder.py "Z:\path" [--movs 20]

Per directory: file counts by extension. PNG: bit depth / colour type from the
IHDR chunk (header bytes only). MOV/MXF/MP4: ffprobe stream info (codec, pix_fmt,
transfer, primaries) when ffprobe is on PATH. Writes tmp\survey_<folder>.json.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, struct, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PNG_CT = {0: "gray", 2: "rgb", 3: "palette", 4: "gray+a", 6: "rgba"}
VIDEO = {".mov", ".mxf", ".mp4", ".mkv", ".avi"}


def png_info(p: Path):
    try:
        with open(p, "rb") as f:
            b = f.read(29)
        if b[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        w, h, depth, ct = struct.unpack(">IIBB", b[16:26])
        return {"w": w, "h": h, "bits": depth, "type": PNG_CT.get(ct, ct)}
    except OSError:
        return None


def ffprobe(p: Path, exe: str):
    try:
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_frames,"
                            "color_transfer,color_primaries,color_space,color_range:format=duration",
                            "-of", "json", str(p)], capture_output=True, text=True, timeout=60)
        d = json.loads(r.stdout or "{}")
        s = (d.get("streams") or [{}])[0]
        s["duration"] = d.get("format", {}).get("duration")
        return s
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--movs", type=int, default=20, help="max video files to ffprobe")
    a = ap.parse_args()
    root = Path(a.root)
    exe = shutil.which("ffprobe")
    dirs: dict[str, Counter] = defaultdict(Counter)
    png_stats: dict[str, Counter] = defaultdict(Counter)
    png_size: dict[str, Counter] = defaultdict(Counter)
    videos = []
    for dp, _, fs in os.walk(root):
        rel = str(Path(dp).relative_to(root))
        for fn in fs:
            ext = Path(fn).suffix.lower() or "<none>"
            dirs[rel][ext] += 1
            p = Path(dp) / fn
            if ext == ".png" and dirs[rel][ext] <= 3:  # first 3 per dir
                i = png_info(p)
                if i:
                    png_stats[rel][f"{i['bits']}bit-{i['type']}"] += 1
                    png_size[rel][f"{i['w']}x{i['h']}"] += 1
            elif ext in VIDEO:
                videos.append(p)
    vids = []
    for p in videos[: a.movs]:
        v = {"file": str(p.relative_to(root))}
        try: v["mb"] = round(p.stat().st_size / 1e6, 1)
        except OSError: pass
        if exe: v.update(ffprobe(p, exe))
        vids.append(v)
    out = {"root": str(root), "ffprobe": exe, "n_videos": len(videos),
           "dirs": {d: {"counts": dict(c), "png": dict(png_stats.get(d, {})), "png_size": dict(png_size.get(d, {}))}
                    for d, c in sorted(dirs.items())},
           "videos": vids}
    dst = REPO / "tmp" / f"survey_{re.sub(r'[^A-Za-z0-9_-]+', '_', root.as_posix())[-80:]}.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=str))
    print(f"{len(dirs)} dirs, {len(videos)} videos, ffprobe={'yes' if exe else 'NO'}\nreport: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
