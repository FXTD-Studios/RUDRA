#!/usr/bin/env python3
"""Roadmap 2.2 and 2.3 (release row R3): RUDRA against the market, on the frozen set.

`configs/compare_set_v1.json` (2.1) is 60 held-out stills and 8 nine-frame clips
from the v4b test split and 40 real SDR frames from Netflix's Meridian. This puts
every method's output for those frames in one bench tree, in one convention, and
scores them all the way release row R1 was scored: PU21-PSNR and CVVDP against
the graded HDR (rudra/delivery/bench.py), error in stops where the SDR clipped
(training/score_regions.py), and each method's paired difference from RUDRA's
analytic inverse with a bootstrap 95 % interval, per part of the set.

    # 1. the reference, RUDRA's analytic inverse and the optional model, plus the
    #    SDR every external tool is given (input709/: 16-bit and 8-bit BT.709 PNGs)
    python training/compare_market.py prepare --out bench/compare_v1 \\
        --checkpoint checkpoints/sdr2hdr_image_v8.pt --test-name v8

    # 2. methods that run here: the BT.2408 direct mapping, HDRTVDM (Guo et al.
    #    2023, its three published weights)
    python training/compare_market.py builtin --out bench/compare_v1 --hdrtvdm <clone>

    # 3. a tool run by hand (Resolve, Topaz, an NLE, a converter): its outputs for
    #    input709/png16/<scene>/<asset>.png, any folder layout, matched by stem
    python training/compare_market.py import --out bench/compare_v1 --name resolve \\
        --src <dir> --encoding pq2020 --note "Resolve 20, ..."

    # 4. score every method and write the table (reports/ is not tracked)
    python training/compare_market.py score --out bench/compare_v1 \\
        --report reports/compare_v1

The convention is the bench's: scene-linear Rec.2020 EXR, diffuse white 1.0 =
203 cd/m2 (score with --nits-scale 203). Every external input is the frame's own
SDR converted from the corpus's canonical form (sRGB code in BT.2020 primaries)
to plain BT.709 sRGB code, the SDR a tool expects; outputs come back as PQ or
linear and are put into Rec.2020 here, so no tool is scored on a gamut it was
never given. What each tool did with colour is recorded in its <name>.run.json
and checked by hand before a table is published (2.2).

The table is published whatever it says (the R3 gate).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DIFFUSE_WHITE_NITS = 203.0
KINDS = ("still", "clip", "netflix_real_sdr")
KIND_LABEL = {"still": "v4b stills (rendered SDR)", "clip": "v4b clips (rendered SDR, frames)",
              "netflix_real_sdr": "Meridian (real SDR)", "all": "the whole set"}


# --------------------------------------------------------------------------- set

def map_path(path: str, maps: list[tuple[str, str]]) -> Path:
    p = path.replace("\\", "/")
    for old, new in maps:
        o = old.replace("\\", "/")
        if p.lower().startswith(o.lower()):
            p = new.replace("\\", "/").rstrip("/") + "/" + p[len(o):].lstrip("/")
            break
    return Path(p)


def records(set_path: Path, maps: list[tuple[str, str]]) -> list[dict]:
    """One record per frame: stills and real frames as they are, a clip's nine frames each."""
    from training.export_bench_pairs import UNSAFE
    data = json.loads(set_path.read_text(encoding="utf-8"))
    out = []
    for item in data["items"]:
        kind = item["kind"]
        if kind == "clip":
            for sdr, hdr in zip(item["sdr_frames"], item["hdr_frames"]):
                out.append(dict(kind=kind, scene_id=item["clip_id"], split="compare",
                                asset_id=UNSAFE.sub("_", Path(sdr.replace("\\", "/")).stem),
                                sdr_path=str(map_path(sdr, maps)), hdr_path=str(map_path(hdr, maps))))
        else:
            out.append(dict(kind=kind, scene_id=item["scene_id"], split="compare",
                            asset_id=UNSAFE.sub("_", str(item["asset_id"])),
                            sdr_path=str(map_path(item["sdr_path"], maps)),
                            hdr_path=str(map_path(item["hdr_path"], maps))))
    missing = [r for r in out if not Path(r["sdr_path"]).exists() or not Path(r["hdr_path"]).exists()]
    if missing:
        raise SystemExit(f"error: {len(missing)} frames not found, first {missing[0]['sdr_path']} "
                         f"(pass --map OLD=NEW for moved corpora)")
    return out


def frame_key(record: dict) -> tuple[str, str]:
    from training.export_bench_pairs import slug
    return slug(record.get("scene_id", "scene")), record["asset_id"]


# ------------------------------------------------------------------ colour helpers

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def matrix(src: str, dst: str) -> np.ndarray:
    from rudra.delivery.colorspace import rgb_to_rgb_matrix
    return np.asarray(rgb_to_rgb_matrix(src, dst), dtype=np.float64)


def apply(m: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    return np.einsum("ij,hwj->hwi", m, rgb)


def bt709_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(x, 0.45) - 0.099)


def canonical_to_bt709(code2020: np.ndarray, video: bool = False) -> np.ndarray:
    """The corpus's SDR (sRGB code in BT.2020 primaries) as plain BT.709 code values.

    Linear light, 2020 -> 709, clipped to the 709 gamut, re-encoded: with the BT.709
    OETF for video (the real SDR was BT.709 video, read through the inverse OETF, so
    this gives back its codes; 13 Oct 2026), with sRGB for the rendered stills."""
    lin = np.clip(apply(matrix("rec2020", "rec709"), srgb_to_linear(code2020)), 0.0, 1.0)
    return bt709_oetf(lin) if video else linear_to_srgb(lin)


VIDEO_KINDS = {"netflix_real_sdr"}   # real SDR that was BT.709 video


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float64) / 65535.0
    return image.astype(np.float64)


def write_png(path: Path, rgb01: np.ndarray, bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = 255.0 if bits == 8 else 65535.0
    code = np.clip(np.rint(rgb01 * top), 0, top).astype(np.uint8 if bits == 8 else np.uint16)
    if not cv2.imwrite(str(path), np.ascontiguousarray(code[..., ::-1])):
        raise OSError(f"could not write {path}")


def write_white1(path: Path, nits2020: np.ndarray) -> None:
    from rudra.delivery.exr import write_exr
    path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(path, (np.maximum(nits2020, 0.0) / DIFFUSE_WHITE_NITS).astype(np.float32), half=True)


def pq_to_nits(signal: np.ndarray) -> np.ndarray:
    from rudra.hdr10 import pq_eotf
    return np.asarray(pq_eotf(np.clip(signal, 0.0, 1.0).astype(np.float32)), dtype=np.float64)


# Each import encoding: (description, function from the file's float values to Rec.2020 nits).
def _decode(encoding: str, values: np.ndarray, white_nits: float) -> np.ndarray:
    if encoding == "pq2020":                 # PQ signal, code / max, BT.2020
        return pq_to_nits(values)
    if encoding == "pq2020_hdrtv1k":         # HDRTV1K files: PQ white at 876 * 74.52 (65282.5)
        return pq_to_nits(values * (65535.0 / (876 * 74.52344347071411)))
    if encoding == "linear2020":             # scene-linear BT.2020, 1.0 = white_nits
        return values * white_nits
    if encoding == "linear709":              # scene-linear BT.709, 1.0 = white_nits
        return apply(matrix("rec709", "rec2020"), values * white_nits)
    raise SystemExit(f"unknown encoding {encoding}")


ENCODINGS = ("pq2020", "pq2020_hdrtv1k", "linear2020", "linear709")


def read_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        from rudra.delivery.exr import read_exr
        rgb, _ = read_exr(path)
        return np.asarray(rgb, dtype=np.float64)[..., :3]
    return read_rgb(path)


# ---------------------------------------------------------------------- commands

def cmd_prepare(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = records(Path(args.set), args.map)
    manifest = out / "_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    kinds = {}
    for r in recs:
        scene, asset = frame_key(r)
        kinds[f"{scene}/{asset}"] = r["kind"]
        sdr = read_rgb(Path(r["sdr_path"]))
        b709 = canonical_to_bt709(sdr, video=r["kind"] in VIDEO_KINDS)
        write_png(out / "input709" / "png16" / scene / f"{asset}.png", b709, 16)
        write_png(out / "input709" / "png8" / scene / f"{asset}.png", b709, 8)
    (out / "_kinds.json").write_text(json.dumps(kinds, indent=1), encoding="utf-8")
    (out / "input709" / "README.txt").write_text(
        "The SDR every external tool is given: each frame of configs/compare_set_v1.json as BT.709 sRGB\n"
        "code values, full range (png16: 16-bit, png8: 8-bit, same pixels rounded), converted from the\n"
        "corpus's canonical form (sRGB code in BT.2020 primaries) in linear light and clipped to BT.709.\n"
        "Interpret as Rec.709 primaries, full range. The Meridian frames (nf_meridian_*) are BT.709\n"
        "video codes (BT.1886 display); the rendered v4b frames are sRGB codes.\n"
        "Ask the tool for HDR10 (PQ, BT.2020) or scene-linear EXR, keep the file stem, then:\n"
        "  python training/compare_market.py import --out <this folder's parent> --name <tool> --src <dir> "
        "--encoding pq2020|linear2020|linear709 [--white-nits N]\n", encoding="utf-8")
    cmd = [sys.executable, str(REPO / "training" / "export_bench_pairs.py"), "--checkpoint", args.checkpoint,
           "--manifest", str(manifest), "--split", "compare", "--out", str(out), "--test-name", args.test_name,
           "--precision", args.precision, "--device", args.device]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"prepared {len(recs)} frames in {out}")


def cmd_pairs(args) -> None:
    """Another real-SDR set as a bench tree (13 Oct 2026): SDR video frames and their graded HDR
    (HDRTV1K's test split, for one), RUDRA's default and Source choices and BT.2408 on it."""
    import torch
    from training.bench_hdrtv1k import canonical_sdr
    out = Path(args.out)
    sdr_files = {p.stem: p for p in Path(args.sdr_dir).iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff")}
    hdr_files = {p.stem: p for p in Path(args.hdr_dir).iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff", ".exr")}
    stems = sorted(set(sdr_files) & set(hdr_files))
    kinds, scene = {}, args.name
    for stem in stems:
        code709 = read_rgb(sdr_files[stem])
        nits = _decode(args.hdr_encoding, read_any(hdr_files[stem]), args.white_nits)
        write_white1(out / "ref" / scene / f"{stem}.exr", nits)
        with torch.inference_mode():   # the canonical SDR RUDRA reads from a BT.709 video frame
            canon = canonical_sdr(code709.astype(np.float32), "cpu")[0].permute(1, 2, 0).double().numpy()
        write_png(out / "sdr" / scene / f"{stem}.png", canon, 16)
        import torch as _t
        from rudra.sdr2hdr import inverse_aces_approx
        aces = inverse_aces_approx(_t.as_tensor(srgb_to_linear(canon), dtype=_t.float32)).double().numpy()
        write_white1(out / "baseline" / scene / f"{stem}.exr", aces * DIFFUSE_WHITE_NITS)
        for name, curve in (("rudra_bt1886", "bt1886"), ("rudra_bt1886_srgb", "bt1886_srgb"), ("rudra_rec709", "clip")):
            write_white1(out / name / scene / f"{stem}.exr", rudra_source_curve(canon, curve) * DIFFUSE_WHITE_NITS)
        bt = apply(matrix("rec709", "rec2020"), np.power(code709, 2.4) * DIFFUSE_WHITE_NITS)
        write_white1(out / "bt2408" / scene / f"{stem}.exr", bt)
        kinds[f"{scene}/{stem}"] = args.name
    (out / "_kinds.json").write_text(json.dumps(kinds, indent=1), encoding="utf-8")
    for name, desc in (("baseline", "RUDRA's analytic inverse (ACES), the default before 13 Oct 2026"),
                       ("rudra_bt1886", "RUDRA, Source = BT.1886 (the default since 13 Oct 2026)"),
                       ("rudra_bt1886_srgb", "RUDRA, BT.1886 taken on the sRGB code (measured, not shipped)"),
                       ("rudra_rec709", "RUDRA, Source = Rec.709 (plain clip)"),
                       ("bt2408", "ITU-R BT.2408 direct mapping of the BT.709 code (BT.1886, 100 % = 203 cd/m2)")):
        _write_run(out, name, dict(method=desc, set=args.name, frames=len(stems)))
    print(f"{args.name}: {len(stems)} pairs in {out}")


def _write_run(out: Path, name: str, info: dict) -> None:
    (out / f"{name}.run.json").write_text(json.dumps(
        dict(info, written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf-8")


def _frames(out: Path) -> list[tuple[str, str]]:
    kinds = json.loads((out / "_kinds.json").read_text(encoding="utf-8"))
    return [tuple(k.split("/", 1)) for k in kinds]


def rudra_source_curve(code: np.ndarray, curve: str) -> np.ndarray:
    """RUDRA's analytic inverse with the Source picker set (native core/source_curve.cpp):
    the ACES inverse times the picked curve's per-code log2 correction, one knot per 8-bit
    code, linear between knots (composite.frag curveLog2). Scene-linear, white 1.0."""
    import torch
    from rudra.sdr2hdr import inverse_aces_approx
    lin = lambda x: srgb_to_linear(np.asarray(x, np.float64))
    aces = lambda d: inverse_aces_approx(torch.as_tensor(d, dtype=torch.float32)).double().numpy()
    knots = np.arange(256) / 255.0
    d = lin(knots)
    if curve == "clip":            # Rec.709, plain clip: the display light itself
        want = np.clip(d, 0.0, 0.995)
    elif curve == "bt1886":        # BT.1886 at zero black: code^2.4 of the BT.709 video's code, which
        x = np.clip(d, 0.0, 0.995)  # RUDRA reads through the inverse BT.709 OETF (13 Oct 2026, the default)
        want = np.power(np.where(x < 0.018, 4.5 * x, 1.099 * np.power(x, 0.45) - 0.099), 2.4)
    elif curve == "bt1886_srgb":   # the same taken on the sRGB code (the first try; measured, not shipped)
        want = np.power(linear_to_srgb(np.clip(d, 0.0, 0.995)), 2.4)
    else:
        raise SystemExit(f"source curve {curve!r}: clip or bt1886 here")
    have = aces(d)
    log2 = np.where((want > 0) & (have > 0), np.log2(np.maximum(want, 1e-30) / np.maximum(have, 1e-30)), 0.0)
    log2[0] = log2[1]
    pos = np.clip(code, 0.0, 1.0) * 255.0
    lo = np.minimum(np.floor(pos).astype(int), 254)
    corr = log2[lo] + (log2[lo + 1] - log2[lo]) * (pos - lo)
    return aces(lin(code)) * np.exp2(corr)


def cmd_builtin(args) -> None:
    out = Path(args.out)
    frames = _frames(out)
    for name, curve, label in (("rudra_rec709", "clip", "Rec.709 (plain clip)"),
                               ("rudra_bt1886_srgb", "bt1886_srgb", "BT.1886 taken on the sRGB code (not shipped)"),
                               ("rudra_bt1886", "bt1886", "BT.1886 of the BT.709 code (display 2.4; the app's default since 13 Oct 2026)")):
        if name not in args.methods:
            continue
        # The Source picker's answers for broadcast SDR (core/source_curve.cpp).
        from training.export_bench_pairs import UNSAFE, slug
        from training.sdr2hdr_dataset import load_rgb
        recs = [json.loads(line) for line in (out / "_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for r in recs:
            scene, asset = slug(r.get("scene_id", "scene")), UNSAFE.sub("_", str(r["asset_id"]))
            kept = out / "sdr" / scene / f"{asset}.png"   # the SDR the network saw, when the tree has it
            code = read_rgb(kept) if kept.exists() else load_rgb(r["sdr_path"], hdr=False).astype(np.float64)
            write_white1(out / name / scene / f"{asset}.exr", rudra_source_curve(code, curve) * DIFFUSE_WHITE_NITS)
        _write_run(out, name, dict(method=f"RUDRA's analytic inverse with Source = {label}, the app's picker "
                                          "(core/source_curve.cpp)",
                                   input="the canonical SDR the network sees", frames=len(recs)))
        print(f"{name}: {len(recs)} frames")
    if "bt2408" in args.methods:
        # ITU-R BT.2408 display-light direct mapping, the broadcast default for SDR in an
        # HDR programme: BT.1886 (gamma 2.4, zero black) at 100 % = 203 cd/m2, BT.709 primaries
        # carried into BT.2020 unchanged. No expansion: the floor any converter starts from.
        for scene, asset in frames:
            code = read_rgb(out / "input709" / "png16" / scene / f"{asset}.png")
            nits = apply(matrix("rec709", "rec2020"), np.power(code, 2.4) * DIFFUSE_WHITE_NITS)
            write_white1(out / "bt2408" / scene / f"{asset}.exr", nits)
        _write_run(out, "bt2408", dict(method="ITU-R BT.2408 display-light direct mapping",
                                       input="input709/png16", frames=len(frames),
                                       note="BT.1886 gamma 2.4, SDR 100% = 203 cd/m2, BT.709 -> BT.2020"))
        print(f"bt2408: {len(frames)} frames")
    if args.hdrtvdm:
        repo = Path(args.hdrtvdm).resolve()
        method = repo / "method"
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        for weights in args.hdrtvdm_weights:
            name = "hdrtvdm" if weights == "params.pth" else "hdrtvdm_" + weights.removeprefix("params_").removesuffix(".pth").lower()
            raw = out / "_raw" / name
            raw.mkdir(parents=True, exist_ok=True)
            # test.py loads params.pth from its folder: run a copy with the chosen weights.
            work = out / "_raw" / f"{name}_code"
            work.mkdir(parents=True, exist_ok=True)
            for f in ("test.py", "network.py"):
                (work / f).write_bytes((method / f).read_bytes())
            (work / "params.pth").write_bytes((method / weights).read_bytes())
            if args.device == "cpu":
                # The authors run the net in fp16 on a GPU; PyTorch's CPU fp16 convolutions are
                # slow, so on CPU the copy runs fp32 (outputs within ~80 of 65535 codes of fp16).
                t = (work / "test.py").read_text(encoding="utf-8")
                t = t.replace("net = TriSegNet().half()", "net = TriSegNet().float()")
                t = t.replace("(2, 0, 1)))).float().half()", "(2, 0, 1)))).float()")
                (work / "test.py").write_text(t, encoding="utf-8")
            # The net halves the frame three times (pixel unshuffle): an input whose sides are not
            # multiples of 8 (a letterboxed crop) is padded by repeating its edge, and the output cropped.
            padded = out / "_raw" / "padded8"
            inputs, sizes = [], {}
            for s_, a_ in frames:
                src = out / "input709" / "png8" / s_ / f"{a_}.png"
                img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                h_, w_ = img.shape[:2]
                if h_ % 8 or w_ % 8:
                    dst = padded / s_ / f"{a_}.png"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dst), cv2.copyMakeBorder(img, 0, (-h_) % 8, 0, (-w_) % 8, cv2.BORDER_REPLICATE))
                    src = dst
                inputs.append(src)
                sizes[a_] = (h_, w_)
            todo = [p for p, (s, a) in zip(inputs, frames) if not (out / name / s / f"{a}.exr").exists()]
            for i in range(0, len(todo), 2):
                batch = todo[i:i + 2]
                subprocess.run([sys.executable, "test.py", *map(str, batch), "-out", str(raw), "-out_format", "tif",
                                "-use_gpu", "false" if args.device == "cpu" else "true"], cwd=work, check=True)
                for p in batch:
                    hits = sorted(raw.glob(p.stem + "_HDR.*"))
                    if len(hits) != 1:
                        raise FileNotFoundError(f"{raw}: expected one output for {p.stem}, found {hits}")
                    h_, w_ = sizes[p.stem]
                    write_white1(out / name / p.parent.name / f"{p.stem}.exr",
                                 _decode("pq2020_hdrtv1k", read_rgb(hits[0])[:h_, :w_], 0.0))
                    hits[0].unlink()
                print(f"{name}: {min(i + 2, len(todo))}/{len(todo)}", flush=True)
            _write_run(out, name, dict(method=f"HDRTVDM (Guo et al., CVPR 2023), {weights}", repo=str(repo),
                                       commit=commit or None, licence="MPL-2.0", input="input709/png8",
                                       frames=len(frames), decode="pq2020_hdrtv1k (PQ white at 65282.5)",
                                       precision="fp32 (CPU)" if args.device == "cpu" else "fp16 (authors')"))


def cmd_import(args) -> None:
    out = Path(args.out)
    frames = _frames(out)
    files = [p for p in Path(args.src).rglob("*") if p.suffix.lower() in (".png", ".tif", ".tiff", ".exr")]
    by_stem: dict[str, list[Path]] = {}
    for p in files:
        by_stem.setdefault(p.stem, []).append(p)
    n, missing = 0, []
    for scene, asset in frames:
        hits = by_stem.get(asset) or [p for p in files if p.stem.startswith(asset + "_") or p.stem.startswith(asset + ".")]
        if len(hits) != 1:
            missing.append(asset)
            continue
        nits = _decode(args.encoding, read_any(hits[0]), args.white_nits)
        write_white1(out / args.name / scene / f"{asset}.exr", nits)
        n += 1
    _write_run(out, args.name, dict(source=str(args.src), encoding=args.encoding, white_nits=args.white_nits,
                                    frames=n, missing=missing, note=args.note))
    print(f"{args.name}: {n} frames" + (f", {len(missing)} missing (first {missing[0]})" if missing else ""))


def _bootstrap(d: np.ndarray, draws: int = 20000, seed: int = 20261011) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(draws, d.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cmd_score(args) -> None:
    from rudra.delivery.bench import run_benchmark
    out = Path(args.out)
    kinds = json.loads((out / "_kinds.json").read_text(encoding="utf-8"))
    skip = {"ref", "input709", "_raw", "results", "sdr"}
    methods = args.methods or sorted(p.name for p in out.iterdir() if p.is_dir() and p.name not in skip)
    results = out / "results"
    for m in methods:
        target = results / f"{m}.json"
        if target.exists() and not args.rescore:
            print(f"{m}: scored already ({target})")
            continue
        print(f"scoring {m} ...", flush=True)
        run_benchmark(out, nits_scale=DIFFUSE_WHITE_NITS, output=target, test_dir=m)
    # Clipped and crushed regions, in stops (needs the SDR the network saw: the manifest's).
    subprocess.run([sys.executable, str(REPO / "training" / "score_regions.py"), str(out), "--test-dir", *methods],
                   check=False)
    per = {}
    for m in methods:
        rows = json.loads((results / f"{m}.json").read_text(encoding="utf-8"))["results"]
        per[m] = {f"{r['clip']}/{r['frame']}": r for r in rows}
        reg = results / f"{m}.regions.csv"
        if reg.exists():
            import csv
            for row in csv.DictReader(reg.read_text(encoding="utf-8").splitlines()):
                key = f"{row.get('scene')}/{row.get('frame')}"
                v = row.get("clipped_err_stops")
                if key in per[m] and v not in (None, "", "nan") and np.isfinite(float(v)):
                    per[m][key]["clipped_err_stops"] = float(v)
    base = args.reference
    table = {}
    parts = ("all",) + tuple(dict.fromkeys(kinds.values()))
    for part in parts:
        keys = [k for k in kinds if part == "all" or kinds[k] == part]
        table[part] = {}
        for m in methods:
            row = {"frames": sum(k in per[m] for k in keys)}
            for metric in ("pu_psnr_db", "cvvdp_jod", "clipped_err_stops"):
                vals = [per[m][k][metric] for k in keys if k in per[m] and per[m][k].get(metric) is not None]
                row[metric] = float(np.mean(vals)) if vals else None
                if m != base and base in per:
                    d = np.array([per[m][k][metric] - per[base][k][metric] for k in keys
                                  if k in per[m] and k in per[base] and per[m][k].get(metric) is not None
                                  and per[base][k].get(metric) is not None])
                    if d.size:
                        lo, hi = _bootstrap(d)
                        row[metric + "_vs_ref"] = dict(mean=float(d.mean()), ci95=[lo, hi], n=int(d.size))
            table[part][m] = row
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    runs = {m: json.loads((out / f"{m}.run.json").read_text(encoding="utf-8"))
            for m in methods if (out / f"{m}.run.json").exists()}
    (report / "compare.json").write_text(json.dumps(dict(reference=base, table=table, runs=runs), indent=2),
                                         encoding="utf-8")
    (report / "compare.md").write_text(markdown(table, base, runs), encoding="utf-8")
    print((report / "compare.md").read_text(encoding="utf-8"))


def _fmt(v, digits=2):
    return "" if v is None else f"{v:.{digits}f}"


def _delta(row, metric, digits=2):
    d = row.get(metric + "_vs_ref")
    if not d:
        return ""
    lo, hi = d["ci95"]
    sig = "" if lo <= 0.0 <= hi else " *"
    return f"{d['mean']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{sig}"


def markdown(table: dict, base: str, runs: dict) -> str:
    lines = [f"# RUDRA against the market (roadmap 2.2/2.3, release row R3), {time.strftime('%Y-%m-%d')}", "",
             "Set: `configs/compare_set_v1.json`. Metrics: PU21-PSNR (dB, higher is better), CVVDP (JOD, "
             "higher is better), error in stops of luminance where the SDR clipped (lower is better). "
             f"Differences are paired, frame by frame, against `{base}` (RUDRA's analytic inverse), with a "
             "bootstrap 95 % interval; * marks an interval that excludes zero. Published whatever it says.", ""]
    for part, rows in table.items():
        lines += [f"## {KIND_LABEL.get(part, part)}", "",
                  "| method | frames | PU21 dB | vs inverse | CVVDP JOD | vs inverse | clipped err (stops) | vs inverse |",
                  "|---|---:|---:|---|---:|---|---:|---|"]
        for m, r in sorted(rows.items(), key=lambda kv: -(kv[1].get("cvvdp_jod") or -1e9)):
            lines.append(f"| {m} | {r['frames']} | {_fmt(r.get('pu_psnr_db'))} | {_delta(r, 'pu_psnr_db')} | "
                         f"{_fmt(r.get('cvvdp_jod'), 3)} | {_delta(r, 'cvvdp_jod', 3)} | "
                         f"{_fmt(r.get('clipped_err_stops'), 3)} | {_delta(r, 'clipped_err_stops', 3)} |")
        lines.append("")
    lines += ["## Methods", ""]
    for m, info in sorted(runs.items()):
        desc = info.get("method") or info.get("note") or info.get("source", "")
        lines.append(f"- `{m}`: {desc}" + (f" ({info['encoding']})" if info.get("encoding") else ""))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def maps(v: str) -> tuple[str, str]:
        old, _, new = v.partition("=")
        if not new:
            raise argparse.ArgumentTypeError("--map OLD=NEW")
        return old, new

    a = sub.add_parser("prepare", help="reference, analytic inverse, the model, and the external input")
    a.add_argument("--set", default=str(REPO / "configs" / "compare_set_v1.json"))
    a.add_argument("--map", type=maps, action="append", default=[],
                   help=r"rewrite a path prefix, e.g. --map G:\corpus_v4b=G:\datasets\corpora\corpus_v4b")
    a.add_argument("--out", required=True)
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--test-name", default="v8")
    a.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    a.add_argument("--device", default="cpu")
    a.set_defaults(fn=cmd_prepare)

    b = sub.add_parser("builtin", help="methods that run here: bt2408, HDRTVDM")
    b.add_argument("--out", required=True)
    b.add_argument("--methods", nargs="*", default=["bt2408", "rudra_rec709", "rudra_bt1886"])
    b.add_argument("--hdrtvdm", help="a clone of github.com/AndreGuo/HDRTVDM")
    b.add_argument("--hdrtvdm-weights", nargs="*", default=["params.pth"],
                   help="also params_3DM.pth params_DaVinci.pth (the authors' other degradations)")
    b.add_argument("--device", default="cpu")
    b.set_defaults(fn=cmd_builtin)

    i = sub.add_parser("import", help="a tool's outputs for input709/, matched by stem")
    i.add_argument("--out", required=True)
    i.add_argument("--name", required=True)
    i.add_argument("--src", required=True)
    i.add_argument("--encoding", choices=ENCODINGS, required=True)
    i.add_argument("--white-nits", type=float, default=DIFFUSE_WHITE_NITS,
                   help="linear encodings: the nits of 1.0 (default 203)")
    i.add_argument("--note", default="", help="the tool, its version and settings, as run")
    i.set_defaults(fn=cmd_import)

    q = sub.add_parser("pairs", help="another real-SDR set: SDR video frames + graded HDR as a bench tree")
    q.add_argument("--out", required=True)
    q.add_argument("--name", required=True, help="the set's name (one scene folder, one part of the table)")
    q.add_argument("--sdr-dir", required=True, help="BT.709 SDR frames (video, BT.709 transfer)")
    q.add_argument("--hdr-dir", required=True)
    q.add_argument("--hdr-encoding", choices=ENCODINGS, default="pq2020")
    q.add_argument("--white-nits", type=float, default=DIFFUSE_WHITE_NITS)
    q.set_defaults(fn=cmd_pairs)

    s = sub.add_parser("score", help="score every method and write the table")
    s.add_argument("--out", required=True)
    s.add_argument("--methods", nargs="*")
    s.add_argument("--reference", default="baseline")
    s.add_argument("--report", default=str(REPO / "reports" / "compare_v1"))
    s.add_argument("--rescore", action="store_true")
    s.set_defaults(fn=cmd_score)

    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
