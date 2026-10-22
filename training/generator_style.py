"""run04: generator-style SDR degradations (22 Oct 2026). No LTX model, weight or data.

Why: RUDRA's main use is SDR from video/image generators. Every training SDR
in the run02 snapshot comes from an ACES output transform on a real plate
(sdr_curve aces1.3_1000 / aces2.1_*), so the model has only ever seen
camera-pipeline SDR. Generated SDR fails differently: highlights are
clipped and smeared rather than rolled off, shadows and skin are
over-smoothed ("plastic"), gradients band, and the files are heavily
compressed for delivery. The ideas are the ordinary ones from the
inverse-tone-mapping literature (and listed in arXiv 2604.11788 among
others); the code is ours.

Every operation acts on the SDR side only and the HDR target is unchanged, so
the conditional-mean answer stays the true scene: under a smeared or clipped
highlight the model is rewarded for the real highlight, under plastic shadows
for the real shadow. Nothing here changes exposure, white balance or the
tone curve as a whole -- that would teach the model to re-grade the picture,
which breaks the SDR-preservation contract. Contrast and saturation jitter
are kept within +/-6 %, a codec/grade wobble rather than a new look.

Deterministic: one NumPy generator, saved and restored with the resume
state exactly like ShadowCodecAugment.
"""
from __future__ import annotations

import numpy as np
import torch

LUMA2020 = np.asarray((0.2627, 0.6780, 0.0593), np.float32)

# Per-operation probabilities, applied only to a crop already selected (p).
DEFAULT_OPS = {
    "highlight_clip": 0.45,     # hard clip at 0.90-0.98 code, no rescale
    "highlight_smear": 0.40,    # blur only where luma > ~0.8
    "shadow_smooth": 0.35,      # blur only where luma < ~0.2
    "plastic": 0.30,            # edge-preserving smoothing, whole crop
    "banding": 0.20,            # requantise to 6-7 bits
    "contrast_sat": 0.40,       # +/-6 % about mid-grey; +/-6 % saturation
    "jpeg": 0.50,               # q 15-55 block compression (delivery codec proxy)
}
ORDER = tuple(DEFAULT_OPS)


def _smooth(x: float, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class GeneratorStyleAugment:
    def __init__(self, probability: float, ops: dict | None = None, seed: int = 0):
        ops = dict(DEFAULT_OPS if ops is None else ops)
        if not 0.0 <= probability <= 1.0 or set(ops) - set(DEFAULT_OPS) \
                or any(not 0.0 <= float(v) <= 1.0 for v in ops.values()):
            raise ValueError("bad generator-style augmentation settings")
        self.p = float(probability)
        self.ops = {k: float(ops.get(k, 0.0)) for k in ORDER}
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- resume state
    def state(self) -> dict:
        s = self.rng.bit_generator.state
        return {"bit_generator": s["bit_generator"], "state": {k: str(v) for k, v in s["state"].items()},
                "has_uint32": int(s["has_uint32"]), "uinteger": str(s["uinteger"])}

    def restore(self, saved: dict) -> None:
        self.rng.bit_generator.state = {"bit_generator": saved["bit_generator"],
                                        "state": {k: int(v) for k, v in saved["state"].items()},
                                        "has_uint32": int(saved["has_uint32"]), "uinteger": int(saved["uinteger"])}

    # ------------------------------------------------------------------ ops
    def _one(self, sdr: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """(H,W,3) float32 code in [0,1] -> degraded copy and the ops applied.
        Every op draws its coin and parameters whether or not it fires, so the
        generator advances by a fixed amount per selected crop."""
        import cv2
        x = sdr.astype(np.float32).copy()
        draws = {k: (self.rng.random() < self.ops[k], self.rng.random(), self.rng.random()) for k in ORDER}
        applied = []
        for k in ORDER:
            on, u, v = draws[k]
            if not on:
                continue
            luma = x @ LUMA2020
            if k == "highlight_clip":
                x = np.minimum(x, 0.90 + 0.08 * u)
            elif k == "highlight_smear":
                sigma = 1.0 + 3.0 * u
                m = _smooth(luma, 0.70, 0.85)[..., None]
                x = x * (1 - m) + cv2.GaussianBlur(x, (0, 0), sigma) * m
            elif k == "shadow_smooth":
                sigma = 0.8 + 2.2 * u
                m = (1.0 - _smooth(luma, 0.12, 0.25))[..., None]
                x = x * (1 - m) + cv2.GaussianBlur(x, (0, 0), sigma) * m
            elif k == "plastic":
                d = 5 + 2 * int(u * 3)
                x = cv2.bilateralFilter(x, d, 0.04 + 0.08 * v, 2.0 + 4.0 * v)
            elif k == "banding":
                levels = 63.0 if u < 0.5 else 127.0
                x = np.round(x * levels) / levels
            elif k == "contrast_sat":
                c, s = 0.94 + 0.12 * u, 0.94 + 0.12 * v
                x = 0.5 + c * (x - 0.5)
                g = (x @ LUMA2020)[..., None]
                x = g + s * (x - g)
            elif k == "jpeg":
                q = 15 + int(u * 41)
                code = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
                ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(code[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    x = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
            x = np.clip(x, 0.0, 1.0).astype(np.float32)
            applied.append(k)
        # No final requantisation: training SDR is canonical BT.2020 code, which is
        # not on the 8-bit grid either; JPEG and banding quantise where they fire.
        return x, applied

    def __call__(self, sdr_batch: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) canonical sRGB code in [0,1], CPU. One selection draw per sample."""
        out = sdr_batch.clone()
        for i in range(sdr_batch.shape[0]):
            if self.rng.random() < self.p:
                img = sdr_batch[i].permute(1, 2, 0).numpy().astype(np.float32)
                out[i] = torch.from_numpy(self._one(img)[0]).permute(2, 0, 1)
        return out
