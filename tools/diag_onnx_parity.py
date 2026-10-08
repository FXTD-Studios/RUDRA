"""Where and why ONNX Runtime disagrees with eager PyTorch on real frames.

The export's parity passes on the 16 small golden frames (<= 300 px) and
failed on 8 Oct 2026 on the 287 Meridian frames at 1080p (ONNX fp32 max |d|
5.6e-3 against atol 3e-4). This runs a few of those frames through eager,
TorchScript and the ONNX sessions of a package (or the .partial one the
failed export left) and prints, per output, the max |d| and where it is,
the values there, and the same for a 512 px centre crop and a 2x downscale
of the frame, so a size-dependent op (Resize, pooling, a conv kernel path)
shows itself.

    python tools/diag_onnx_parity.py --package dist/models/.sdr2hdr_shadow_v1.partial \\
        --checkpoint checkpoints/sdr2hdr_shadow_v1.pt \\
        --bench-dir G:/.../manifest.jsonl --frames 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.export_model import (  # noqa: E402
    OUTPUTS, TOLERANCE, _to_tensor, bench_frames, eager_reference, load_network, run_onnx, run_torchscript,
)


def report(label: str, ref: dict, got: dict) -> None:
    for k in OUTPUTS:
        a = ref[k].astype(np.float64)
        b = got[k].astype(np.float64)
        if not a.size:
            continue
        d = np.abs(a - b)
        i = np.unravel_index(int(np.argmax(d)), d.shape)
        tol = TOLERANCE["onnx" if label.startswith("onnx") else label]
        over = float(np.max(d - (tol["atol"] + tol["rtol"] * np.abs(a))))
        flag = "  OVER" if over > 0 else ""
        print(f"    {label:11s} {k:14s} max|d| {float(d.max()):.3e} at {tuple(int(x) for x in i)}"
              f"  ref {float(a[i]):+.5f} got {float(b[i]):+.5f}  mean|d| {float(d.mean()):.2e}{flag}")


def main(argv=None) -> int:
    import onnxruntime as ort

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--variants", action="store_true",
                    help="also run the full frame under ORT with graph optimisations off, one thread, and "
                         "no pre-packing, to see which knob the error follows")
    args = ap.parse_args(argv)
    net, _ = load_network(args.checkpoint)
    ts = torch.jit.load(str(args.package / "model.ts"))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    f_sess = ort.InferenceSession(str(args.package / "model.frame.onnx"), so, providers=["CPUExecutionProvider"])
    t_sess = ort.InferenceSession(str(args.package / "model.tile.onnx"), so, providers=["CPUExecutionProvider"])
    print(f"onnxruntime {ort.__version__}, torch {torch.__version__}, threads {torch.get_num_threads()}")
    for name, hwc in bench_frames(args.bench_dir, args.frames):
        h, w = hwc.shape[:2]
        variants = [("full", hwc)]
        if h >= 512 and w >= 512:
            y0, x0 = (h - 512) // 2, (w - 512) // 2
            variants.append(("centre 512", np.ascontiguousarray(hwc[y0:y0 + 512, x0:x0 + 512])))
        half = np.ascontiguousarray(hwc[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3)).astype(np.float32))
        variants.append(("half size", half))
        for vname, v in variants:
            sdr = _to_tensor(v)
            ref = eager_reference(net, sdr)
            print(f"{name} [{vname}] {v.shape[1]}x{v.shape[0]}")
            report("torchscript", ref, run_torchscript(ts, sdr))
            report("onnx", ref, run_onnx(f_sess, t_sess, sdr))
        if args.variants:
            sdr = _to_tensor(hwc)
            ref = eager_reference(net, sdr)
            def session(path, **kw):
                o = ort.SessionOptions()
                o.log_severity_level = 3
                for k, v in kw.items():
                    if k == "config":
                        for ck, cv in v.items():
                            o.add_session_config_entry(ck, cv)
                    else:
                        setattr(o, k, v)
                return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
            variants = {
                "opt off": dict(graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                "opt basic": dict(graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                "1 thread": dict(intra_op_num_threads=1),
                "no prepack": dict(config={"session.disable_prepacking": "1"}),
                "no nchwc": dict(config={"session.use_nchwc": "0"}),   # ignored by builds without the knob
            }
            print(f"{name} [full] ORT variants, residual / highlight / shadow max|d| vs eager:")
            for vname, kw in variants.items():
                try:
                    fs = session(args.package / "model.frame.onnx", **kw)
                    tsess = session(args.package / "model.tile.onnx", **kw)
                    got = run_onnx(fs, tsess, sdr)
                    print(f"    {vname:11s} " + "  ".join(f"{k} {np.max(np.abs(ref[k].astype(np.float64) - got[k])):.3e}" for k in ("residual", "highlight", "shadow")))
                except Exception as e:  # noqa: BLE001
                    print(f"    {vname:11s} failed: {e}")
        # The ONNX tile session on the frame's own tiles vs eager tiled: is the
        # error in the untiled full-frame pass only?
        sdr = _to_tensor(hwc)
        from training.infer_sdr2hdr import predict_fields  # noqa: E402
        with torch.inference_mode():
            tiled = predict_fields(net, sdr, tile_size=512, overlap=64)
        full = eager_reference(net, sdr)
        d = float(np.max(np.abs(tiled["residual"].float().numpy() - full["residual"])))
        print(f"  eager tiled(512/64) vs eager untiled: residual max|d| {d:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
