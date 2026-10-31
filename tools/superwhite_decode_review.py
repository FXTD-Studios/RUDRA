"""Bounded synthetic headroom review; no production processing or datasets touched.

Uses the explicitly supplied FFmpeg, writes into a new directory only, and
compares known YUV444 code ramps with an independent matrix calculation.
This is decoder evidence, not an HDR quality or performance benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def run(args: list[str], commands: list[list[str]]) -> bytes:
    commands.append(args)
    result = subprocess.run(args, capture_output=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg = str(args.ffmpeg.resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=False)
    commands: list[list[str]] = []
    report = {"review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "ffmpeg_version": run([ffmpeg, "-version"], commands).decode().splitlines()[0],
              "ffmpeg_sha256": hashlib.sha256(Path(ffmpeg).read_bytes()).hexdigest(),
              "cases": [], "checks": {}}
    for bits in (8, 10):
        k = 2 ** (bits - 8)
        width, height = 2 ** bits, 4
        dtype = "u1" if bits == 8 else "<u2"
        fmt = "yuv444p" if bits == 8 else "yuv444p10le"
        codes = np.arange(width)
        y = np.tile(codes, (height, 1))
        u = np.full_like(y, 128 * k)
        v = np.full_like(y, 128 * k)
        # Rows 0/1: neutral full-code ramps (including reserved edge codes).
        # Rows 2/3: in-range Y/Cb/Cr, but some RGB channels exceed nominal range.
        y[2:] = 180 * k
        u[2:] = np.clip(codes, 16 * k, 240 * k)
        v[2:] = 200 * k
        raw = args.out / f"ramp_{bits}bit.yuv"
        raw.write_bytes(np.stack([y, u, v]).astype(dtype).tobytes())
        for matrix, kr, kb, primaries in (("709", .2126, .0722, "bt709"),
                                          ("2020_ncl", .2627, .0593, "bt2020")):
            matrix_tag = "bt709" if matrix == "709" else "bt2020nc"
            base = [ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pixel_format", fmt,
                    "-video_size", f"{width}x{height}", "-framerate", "1", "-color_range", "tv",
                    "-colorspace", matrix_tag, "-color_trc", "bt709", "-color_primaries", primaries,
                    "-i", str(raw), "-frames:v", "1"]
            prefix = f"zscale=matrixin={matrix}:rangein=limited:matrix=gbr:range="
            specs = {
                "delivery_integer_full": (prefix + "full,format=gbrp16le,format=rgb48le", "rgb48le"),
                "candidate_integer_limited": (prefix + "limited,format=gbrp16le", "gbrp16le"),
                "candidate_float_full": (prefix + "full,format=gbrpf32le", "gbrpf32le"),
                "playback_integer": (None, "rgb48le"),
            }
            yy = (y.astype(float) - 16 * k) / (219 * k)
            cb = (u.astype(float) - 128 * k) / (224 * k)
            cr = (v.astype(float) - 128 * k) / (224 * k)
            r = yy + 2 * (1 - kr) * cr
            b = yy + 2 * (1 - kb) * cb
            g = (yy - kr * r - kb * b) / (1 - kr - kb)
            expected = np.stack([r, g, b], axis=-1)
            results = {}
            arrays = {}
            for name, (vf, pixfmt) in specs.items():
                command = base + (["-vf", vf] if vf else []) + ["-pix_fmt", pixfmt, "-f", "rawvideo", "pipe:1"]
                data = run(command, commands)
                floating = pixfmt == "gbrpf32le"
                values = np.frombuffer(data, dtype="<f4" if floating else "<u2").astype(float)
                if pixfmt == "rgb48le":
                    rgb = values.reshape(height, width, 3) / 65535
                else:
                    rgb = values.reshape(3, height, width)[[2, 0, 1]].transpose(1, 2, 0)
                    if not floating:
                        # Nominal limited 16-bit endpoints are 4096/60160.
                        rgb = (rgb - 4096) / 56064
                arrays[name] = rgb
                legal = slice(16 * k, 235 * k + 1)
                headroom = rgb[0, 235 * k + 1:255 * k, 0]
                picks = [16 * k, 128 * k, 235 * k, 236 * k, 254 * k, 255 * k - 1]
                results[name] = {
                    "raw_buffer_min_max": [float(values.min()), float(values.max())],
                    "neutral_samples": {str(c): rgb[0, c].tolist() for c in sorted(set(picks))},
                    "headroom_distinct_r": int(np.unique(headroom).size),
                    "headroom_samples": int(headroom.size),
                    "headroom_strictly_increasing": bool(np.all(np.diff(headroom) > 0)),
                    "neutral_legal_max_abs_error": float(np.max(np.abs(rgb[0, legal] - expected[0, legal]))),
                    "all_max_abs_error": float(np.max(np.abs(rgb - expected))),
                    "colored_rgb_max": float(rgb[2:].max()),
                    "colored_max_abs_error": float(np.max(np.abs(rgb[2:] - expected[2:]))),
                }
            # Pause/seek path caches a PNG, whose integer RGB cannot hold >1.
            png = args.out / f"paused_{bits}_{matrix}.png"
            run(base + ["-update", "1", str(png)], commands)
            data = run([ffmpeg, "-v", "error", "-i", str(png), "-frames:v", "1", "-pix_fmt",
                        "rgb48le", "-f", "rawvideo", "pipe:1"], commands)
            paused = np.frombuffer(data, dtype="<u2").reshape(height, width, 3).astype(float) / 65535
            results["paused_png"] = {
                "headroom_distinct_r": int(np.unique(paused[0, 235 * k + 1:255 * k, 0]).size),
                "headroom_samples": 20 * k - 1,
                "max": float(paused.max()),
                "max_difference_from_playback": float(np.max(np.abs(paused - arrays["playback_integer"]))),
            }
            legacy = arrays["delivery_integer_full"]
            candidate = arrays["candidate_float_full"]
            legal = slice(16 * k, 235 * k + 1)
            # Explicitly test the earlier bit-exact assertion after repacking float.
            requantized = np.floor(np.clip(candidate, 0, 1) * 65535 + .5) / 65535
            case = {"bits": bits, "matrix": matrix, "results": results,
                    "float_requantized_legal_mismatch_components": int(np.count_nonzero(
                        requantized[0, legal] != legacy[0, legal]))}
            report["cases"].append(case)
            key = f"{bits}_{matrix}"
            report["checks"][key + "_float_matches_independent_matrix"] = bool(np.max(np.abs(candidate - expected)) < 2e-6)
            report["checks"][key + "_float_preserves_neutral_headroom"] = results["candidate_float_full"]["headroom_strictly_increasing"]
            report["checks"][key + "_legacy_collapses_headroom"] = results["delivery_integer_full"]["headroom_distinct_r"] == 1
            # swscale leaves 941..943 distinct in 10-bit input, then clips the
            # remainder. Test loss, not the incorrect claim that *every* value
            # above nominal white becomes identical in every decoder path.
            report["checks"][key + "_png_loses_headroom"] = (
                results["paused_png"]["headroom_distinct_r"] < 20 * k - 1
                and results["paused_png"]["max"] <= 1)
            report["checks"][key + "_limited_integer_proposal_loses_headroom"] = (
                results["candidate_integer_limited"]["headroom_distinct_r"] == 1)
    # Existing native inverse clamps display light to .995. Continuing its
    # implemented endpoint slope is flat, not a highlight reconstruction.
    endpoint = (1.099 * .995 ** .45 - .099) ** 2.4
    report["bt1886_boundary"] = {"legacy_relative_white": endpoint,
                                "legacy_white_nits_at_203": endpoint * 203,
                                "proposed_power_extension_at_1019_nits": 203 * ((1019 - 64) / 876) ** 2.4,
                                "note": "Analytical policy illustration, not a rendered master or normative headroom EOTF."}
    report["commands"] = commands
    report["passed"] = all(report["checks"].values())
    (args.out / "review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "report": str(args.out / "review.json")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
