"""Refuse a public RUDRA package that carries model weights.

Since 1.0.3 the public packages carry no model at all: the learned ACES 2.0
model and AI fill are downloaded inside the app (Model packages), each with its
own licence shown first. The Beta 2 weights (sdr2hdr_image_v8, sdr2hdr_shadow_v1
and the other sdr2hdr_* checkpoints) are non-commercial and must never ship in
an Apache-2.0 package; 1.0.0 to 1.0.2 shipped sdr2hdr_image_v8 by mistake.

    python tools/check_package_weights.py dist/beta/RUDRA-1.0.3-windows-x64.zip
    python tools/check_package_weights.py /Volumes/RUDRA\\ 1.0.3/RUDRA.app

Takes a folder, a .zip or a .app; exits 1 and lists every offending file.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import PurePosixPath, Path

# Weight files by extension. .ts is TorchScript here (the package has no TypeScript).
WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".ts", ".mlmodel", ".mlpackage", ".bin"}
# Names that are non-commercial whatever their extension.
NONCOMMERCIAL_MARKERS = ("sdr2hdr_", "shadow_v1", "shadow_s2", "shadow_s3", "temporal_v1")
# Runtime libraries that legitimately end in a weight-like extension: none today;
# kept as an explicit list so an exception is a reviewed change.
ALLOWED = set()


def offending(names: list[str]) -> list[str]:
    bad = []
    for name in names:
        p = PurePosixPath(name.replace("\\", "/"))
        if not p.name or name in ALLOWED:
            continue
        low = [part.lower() for part in p.parts]
        if any(m in part for part in low for m in NONCOMMERCIAL_MARKERS):
            bad.append(name)
        elif p.suffix.lower() in WEIGHT_EXTENSIONS and "models" in low:
            bad.append(name)
        elif p.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}:
            bad.append(name)
        elif p.name == "LICENSE-weights":
            bad.append(name)   # only travels with bundled Beta 2 weights
    return bad


def list_names(target: Path) -> list[str]:
    if target.is_file() and target.suffix.lower() == ".zip":
        with zipfile.ZipFile(target) as z:
            return z.namelist()
    if target.is_dir():
        return [str(p.relative_to(target)) for p in target.rglob("*")]
    raise SystemExit(f"not a folder, .app or .zip: {target}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", type=Path)
    args = ap.parse_args(argv)
    failed = False
    for t in args.targets:
        bad = offending(list_names(t))
        if bad:
            failed = True
            print(f"FAILED: {t} carries model weights a public package must not ship:")
            for b in sorted(bad):
                print(f"  {b}")
        else:
            print(f"ok  {t}: no model weights")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
