#!/usr/bin/env python3
"""The native lock's parity golden (7 Nov 2026): the Python oracle on a made-up frame.

native/core/src/aces_lock.cpp mirrors training/aces_partial_tighten.py (lock_observed,
dequantize_recorded, the appearance guard). This writes one small frame, viewed through
a non-separable toy view (the same MixingView as tests/test_aces_partial_tighten.py, also
written in native/tests/test_aces_lock.cpp), and the oracle's results for three settings:
the registered arm lock_dqs050g, the full lock (strength 1) and the integer-code target.
No footage, no model.

    python tools/make_lock_golden.py native/tests/golden/aces_lock/lock_parity_v1.bin

Format (little endian): b"RLK1", int32 h, w, cases; source f32 HWC; constrained f32 HWC;
weight f32 HW; dequantized f64 HWC; used u8 HWC; then per case: f64 strength,
int32 dequant, int32 changed, kept, raised, dequant_channels, f32 HWC result.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.aces_partial_tighten import dequantize_recorded, partial_support, tighten_partial  # noqa: E402


class MixingView:
    """Not separable, like ACES: a channel-mixing matrix, then a per-channel tone curve."""
    M = np.array([[0.90, 0.08, 0.02], [0.02, 0.96, 0.02], [0.01, 0.04, 0.95]], np.float64)

    def forward(self, x):
        y = np.maximum(np.asarray(x, np.float64), 0) @ self.M.T
        return (y / (1 + y)).astype(np.float32)


def frame(h=40, w=56, seed=7):
    """A warm highlight ramp (red past the clip, green and blue recorded), a textured patch,
    a fully clipped patch, mid-grey, and a strip the weight leaves out."""
    v = MixingView()
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    scene = np.empty((h, w, 3))
    scene[..., 0] = 40.0 + 900.0 * xx / (w - 1)
    scene[..., 1] = 1.4 + 1.6 * yy / (h - 1) + 0.3 * np.sin(xx / 9.0)
    scene[..., 2] = 0.8 + 0.5 * xx / (w - 1)
    tex = (slice(4, 16), slice(30, 46))
    scene[tex] *= np.exp(rng.normal(0, 0.12, scene[tex].shape))
    scene[24:32, 40:52] = [5000.0, 4000.0, 3500.0]   # fully clipped
    scene[30:40, 0:12] = [0.18, 0.18, 0.18]           # mid-grey
    src = (np.rint(v.forward(scene.astype(np.float32)) * 255) / 255).astype(np.float32)
    drift = np.exp(rng.normal(0, 0.012, scene.shape)) * np.where(rng.random((h, w, 1)) < 0.15, 1.04, 1.0)
    cur = (scene * drift).astype(np.float32)
    weight = np.ones((h, w), np.float32)
    weight[:, 20:22] = 0.0
    return v, src, cur, weight


CASES = [(0.5, True), (1.0, True), (0.5, False)]


def main(argv=None) -> int:
    out = Path((argv or sys.argv[1:])[0])
    v, src, cur, weight = frame()
    sup = partial_support(src, weight)
    dq, used = dequantize_recorded(src, region=sup, iters=200, max_lap=1.5)
    blob = [b"RLK1", struct.pack("<iii", src.shape[0], src.shape[1], len(CASES)),
            src.astype("<f4").tobytes(), cur.astype("<f4").tobytes(), weight.astype("<f4").tobytes(),
            dq.astype("<f8").tobytes(), used.astype(np.uint8).tobytes()]
    for strength, dequant in CASES:
        kw = dict(target="dequant", dq_iters=200, dq_max_lap=1.5) if dequant else {}
        res, d = tighten_partial(src, cur, src, weight, v, mode="lock", iters=4, band_codes=0.0, strength=strength, **kw)
        blob.append(struct.pack("<diiiii", strength, int(dequant), d["changed_pixels"], d["kept_pixels"],
                                d.get("strength_raised_pixels", 0), d.get("dequant_channels", 0)))
        blob.append(res.astype("<f4").tobytes())
        print(f"strength {strength} dequant {dequant}: support {d['support_pixels']} changed {d['changed_pixels']} "
              f"kept {d['kept_pixels']} raised {d.get('strength_raised_pixels', 0)} dequant ch {d.get('dequant_channels', 0)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"".join(blob))
    print(f"{out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
