"""Export a trained PeakNet to ONNX (18 Oct 2026): what the app and the torch-free tools run.

Inputs: patch (N, 6, 64, 64) float32 and feats (N, 12) float32, exactly as
build_dataset.region_inputs makes them; output: a (N,) dome heights in stops.
Feature standardisation is inside the graph. Checked against torch on random
inputs before it is written.

    python training/peaknet/export_onnx.py checkpoints/peaknet_v2/peaknet.pt checkpoints/peaknet_v2/peaknet.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.peaknet.train import PeakNet  # noqa: E402


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    src, dst = Path(argv[0]), Path(argv[1])
    ck = torch.load(src, map_location="cpu", weights_only=True)
    model = PeakNet(int(ck["n_feats"]), float(ck["a_grid"][-1]), int(ck["width"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    patch, feats = torch.rand(4, 6, 64, 64), torch.randn(4, int(ck["n_feats"]))
    torch.onnx.export(model, (patch, feats), str(dst), input_names=["patch", "feats"], output_names=["a"],
                      dynamic_axes={"patch": {0: "n"}, "feats": {0: "n"}, "a": {0: "n"}}, opset_version=17,
                      dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    p, f = np.random.rand(16, 6, 64, 64).astype(np.float32), np.random.randn(16, int(ck["n_feats"])).astype(np.float32)
    want = model(torch.from_numpy(p), torch.from_numpy(f)).detach().numpy()
    got = sess.run(["a"], {"patch": p, "feats": f})[0]
    err = float(np.abs(want - got).max())
    if err > 1e-4:
        raise SystemExit(f"ONNX differs from torch by {err}")
    print(f"wrote {dst} (max |onnx - torch| = {err:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
