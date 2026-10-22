"""Bounded synthetic native decode benchmark; no datasets, training or downloads.

Build rudra_headroom_frames_bench explicitly, then pass --binary. Reports preserve
per-frame XXH64 evidence. Working-set sums are sampled, not unique physical RAM.
Each new --out directory is exclusive; fixture generation never overwrites files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import psutil


def monitored(argv: list[str], log: Path, timeout: float = 120, trace: bool = False) -> dict:
    started = time.perf_counter()
    peak_tree = peak_parent = child_count = 0
    children_seen: set[int] = set()
    process_peaks: dict[int, dict] = {}
    samples: list[dict] = []
    with log.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT)
        parent = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                if time.perf_counter() - started > timeout:
                    raise TimeoutError(f"Bounded benchmark exceeded {timeout}s")
                try:
                    children = parent.children(recursive=True)
                    children_seen.update(p.pid for p in children)
                    child_count = max(child_count, len(children))
                    tree_bytes = 0
                    ffmpeg_bytes = 0
                    for p in [parent, *children]:
                        try:
                            rss = p.memory_info().rss
                            tree_bytes += rss
                            name = p.name().lower()
                            threads = p.num_threads()
                            entry = process_peaks.setdefault(p.pid, {
                                "name": name, "peak_rss_bytes": 0, "peak_threads": 0,
                                "first_seen_seconds": time.perf_counter() - started,
                            })
                            entry["peak_rss_bytes"] = max(entry["peak_rss_bytes"], rss)
                            entry["peak_threads"] = max(entry["peak_threads"], threads)
                            entry["last_seen_seconds"] = time.perf_counter() - started
                            if name == "ffmpeg.exe" or name == "ffmpeg":
                                ffmpeg_bytes += rss
                            if p.pid == process.pid:
                                peak_parent = max(peak_parent, rss)
                        except psutil.Error:
                            pass
                    peak_tree = max(peak_tree, tree_bytes)
                    if trace:
                        samples.append({"seconds": time.perf_counter() - started,
                                        "tree_bytes": tree_bytes, "ffmpeg_bytes": ffmpeg_bytes})
                except psutil.Error:
                    pass
                time.sleep(.01)
        except BaseException:
            try:
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.Error:
                pass
            process.wait()
            raise
        code = process.wait()
    if code:
        raise RuntimeError(f"Benchmark command failed ({code}); see {log}")
    if trace:
        log.with_suffix(".memory.json").write_text(json.dumps(samples) + "\n", encoding="utf-8")
    return {
        "wall_seconds": time.perf_counter() - started,
        "sampled_tree_working_set_bytes": peak_tree,
        "sampled_parent_working_set_bytes": peak_parent,
        "observed_child_processes": len(children_seen),
        "maximum_simultaneous_children": child_count,
        "sample_interval_ms": 10,
        "process_peaks": list(process_peaks.values()),
    }


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--compare-binary", type=Path, help="Frozen baseline; alternate paired measurements")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--thread-sweep", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Include candidate-only continuous stream prototype")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be between 1 and 5")
    binary = args.binary.resolve(strict=True)
    baseline = args.compare_binary.resolve(strict=True) if args.compare_binary else None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("Pinned FFmpeg must be on PATH")
    args.out.mkdir(parents=True, exist_ok=False)
    args.fixtures.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, width, height in [("hd", 1920, 1080), ("uhd", 3840, 2160)]:
        movie = args.fixtures / f"{name}_24frames_ffv1.mkv"
        if not movie.exists():
            monitored([
                ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
                f"testsrc2=size={width}x{height}:rate=24000/1001,format=yuv444p10le,lutyuv=y=val*1.08,"
                "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                "-frames:v", "24", "-c:v", "ffv1", "-level", "3", "-threads", "4",
                "-pix_fmt", "yuv444p10le", "-color_range", "tv", "-color_trc", "bt709",
                "-color_primaries", "bt709", "-colorspace", "bt709", "-output_ts_offset", "5", str(movie),
            ], args.out / f"generate_{name}.log")
        sources.append((name, movie.resolve()))
    summary = {
        "schema": "headroom-benchmark-run-v1", "binary": str(binary), "binary_sha256": sha256(binary),
        "ffmpeg": str(ffmpeg), "ffmpeg_sha256": sha256(Path(ffmpeg)),
        "fixtures": [{"name": n, "path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                     for n, p in sources], "runs": [],
        "limits": "24 synthetic frames each; no rendering/display/encoding in timed decode; includes consumer hash",
    }
    if baseline:
        summary["baseline_binary"] = str(baseline)
        summary["baseline_binary_sha256"] = sha256(baseline)
    for repeat in range(args.repeats):
        for name, movie in sources:
            # Rotate order to make systematic warm-cache effects visible.
            modes = [("sequential", None), ("legacy", None), ("feeder", None)]
            if args.stream:
                modes.append(("stream", None))
            if args.thread_sweep:
                modes.extend([("sequential", 4), ("sequential", 8)])
            modes = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            reference = None
            for mode, threads in modes:
                binaries = [("baseline", baseline), ("candidate", binary)] if baseline else [("candidate", binary)]
                if mode == "stream":
                    binaries = [("candidate", binary)]  # Frozen baseline predates this API.
                if repeat % 2:
                    binaries.reverse()
                for variant, executable in binaries:
                    stem = f"{name}_{repeat}_{variant}_{mode}" + (f"_threads{threads}" if threads else "")
                    report = args.out / f"{stem}.json"
                    argv = [str(executable), str(movie), mode, str(report.resolve())]
                    if threads:
                        argv.append(str(threads))
                    metrics = monitored(argv, args.out / f"{stem}.log")
                    data = json.loads(report.read_text(encoding="utf-8"))
                    if not data["ok"]:
                        raise RuntimeError(data)
                    hashes = data["frame_xxh64"]
                    if mode == "sequential" and threads is None:
                        if reference is not None and hashes != reference:
                            raise RuntimeError(f"Reference float parity failure: {report}")
                        reference = hashes
                    row = {"fixture": name, "repeat": repeat, "mode": mode, "threads": threads, "variant": variant,
                           "report": str(report.resolve()), **metrics, **data}
                    summary["runs"].append(row)
                    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                    print(f"{stem}: {data['feed_ms']:.1f} ms; {metrics['sampled_tree_working_set_bytes']/2**20:.1f} MiB sampled tree", flush=True)
            for row in summary["runs"]:
                if row["fixture"] == name and row["repeat"] == repeat and row["mode"] != "legacy":
                    if row["frame_xxh64"] != reference:
                        raise RuntimeError(f"Float sample parity failure: {row['report']}")
    summary["float_hash_parity"] = True
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
