"""Exercise native export refusal paths without loading weights or using a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    exe = args.exe.resolve()
    cases = [
        ("same_stem", ["shot.png", "shot.tif"], "folder", "same EXR name"),
        ("case_alias", ["Shot.png", "shot.tif"], "folder", "same EXR name"),
        ("empty", [], "folder", "No stills found"),
        ("missing", [], "missing.png", "existing supported still"),
        ("unsupported_video", ["movie.mp4"], "movie.mp4", "existing supported still"),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="rudra-aces-preflight-") as temp:
        root = Path(temp)
        for name, files, source, message in cases:
            folder = root / name
            folder.mkdir()
            for filename in files:
                (folder / filename).write_bytes(b"preflight only; not decoded")
            output = root / (name + "_output")
            result = subprocess.run(
                [str(exe), "aces-v2", str(root / "absent-package"),
                 str(folder if source == "folder" else folder / source),
                 "--out", str(output)],
                capture_output=True, text=True, timeout=30,
            )
            passed = result.returncode == 2 and message in result.stderr and not output.exists()
            results.append(dict(case=name, passed=passed, exit_code=result.returncode,
                                stderr=result.stderr, output_created=output.exists()))
    report = dict(executable=str(exe), sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  cases=results, passed=all(row["passed"] for row in results),
                  scope="Input preflight only; no model inference or video qualification")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
