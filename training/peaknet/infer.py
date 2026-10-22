"""PeakNet at inference: an SDR frame -> clip-fill overrides (17 Oct 2026).

The output is exactly what the app already accepts for a clipped region
(native ClipOverride: normalised x, y and the region's peak in nits), so the
shipped clip fill renders the result and nothing outside fully clipped pixels
can change. Regions PeakNet does not see (smaller than MIN_PIXELS) keep the
default dome.

    from training.peaknet.infer import load_peaknet, predict_overrides
    model = load_peaknet("checkpoints/peaknet_v2/peaknet.onnx")   # or .pt (needs torch)
    overrides = predict_overrides(model, sdr_codes_hwc, probe="build/clip_fill_probe")
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.peaknet.build_dataset import LUMA, MIN_PIXELS, full_clip_labels, region_inputs, run_probe  # noqa: E402


def load_peaknet(path: str | Path, device: str = "cpu"):
    """A callable (patches (N,6,64,64), feats (N,12)) -> a (N,) from peaknet.onnx (onnxruntime,
    no torch) or peaknet.pt (torch)."""
    path = Path(path)
    if path.suffix == ".onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return lambda p, f: sess.run(["a"], {"patch": p.astype(np.float32), "feats": f.astype(np.float32)})[0]
    import torch
    from training.peaknet.train import PeakNet
    ck = torch.load(path, map_location=device, weights_only=True)
    model = PeakNet(int(ck["n_feats"]), float(ck["a_grid"][-1]), int(ck["width"]))
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()

    def run(p, f):
        with torch.no_grad():
            return model(torch.from_numpy(p.astype(np.float32)).to(device),
                         torch.from_numpy(f.astype(np.float32)).to(device)).cpu().numpy()
    return run


def predict_overrides(model, codes: np.ndarray, probe: str, tmp: Path | None = None
                      ) -> list[tuple[float, float, float]]:
    """codes (H, W, 3) float32 SDR code values in [0, 1] -> [(x, y, nits)] for the clip fill."""
    codes = np.ascontiguousarray(codes, np.float32)
    n, labels = full_clip_labels(codes)
    if n <= 1:
        return []
    h, w = labels.shape
    filled_def, regs = run_probe(probe, codes, tmp=tmp)
    filled0, regs0 = run_probe(probe, codes, glow=0.0, tmp=tmp)
    if len(regs) != len(regs0):
        return []
    y0_log = np.log2(np.maximum(filled0 @ LUMA, 1e-9))
    any_full = labels > 0
    feats, patches, where = [], [], []
    for r0, rd in zip(regs0, regs):
        if r0["pixels"] < MIN_PIXELS:
            continue
        lab = labels[int(r0["y"]), int(r0["x"])]
        if lab == 0:
            continue
        mask = labels == lab
        a_def = float(np.log2(max(rd["peak_nits"], 1e-9) / max(r0["peak_nits"], 1e-9)))
        f, p = region_inputs(codes, y0_log, mask, any_full, r0, a_def)
        feats.append(f)
        patches.append(p)
        where.append(((r0["x"] + 0.5) / w, (r0["y"] + 0.5) / h, r0["peak_nits"]))
    if not feats:
        return []
    a = model(np.stack(patches).astype(np.float32), np.stack(feats).astype(np.float32))
    return [(x, y, float(centre * 2.0 ** float(ai))) for (x, y, centre), ai in zip(where, a)]
