"""run03 experiment pieces for the measured near-black defect (22 Oct 2026).

Finding: on codec-crushed, noisy near-black SDR (Mataha A_0027C021, H.265) the
run02 model lifts noise into saturated blobs, up to +10 stops over the
baseline; on clean photographic shadows (SI-HDR) shadow reconstruction is good.
A luminance-only guard could not tell the two apart (pre-registered C3 failed).
Two learned remedies, both anchored to the TARGET (so genuine shadow detail is
still rewarded) and both outside the pinned evaluation/decoder files:

1. ShadowCodecAugment -- with probability p, degrade only the shadows of a
   training SDR crop: blend in a low-quality JPEG round trip and a small black
   crush below a soft luma threshold. The HDR target is unchanged, so the
   conditional-mean answer for noisy crushed blacks is the smooth true shadow,
   never a coloured blob. Mid-tones and highlights are untouched (no tone-curve
   inconsistency is introduced). Its NumPy generator state is part of the
   resume state, so exact continuation still holds.

2. near_black_log_loss -- the existing shadow-chroma term is absolute
   (network units ~1e-5 at 0.1 nit), so blobs in deep blacks cost nearly
   nothing. This term works in LOG space on pixels whose SDR max code is at or
   below `code_threshold`: |log Y_out - log Y_target| plus log-opponent chroma
   difference, floor 0.001 nit.

3. pu_min selection -- best.pt is chosen on the trailing median of
   -min(clean PU gain, hard PU gain) instead of the log-domain composite.
"""
from __future__ import annotations

import numpy as np
import torch
from rudra.torch_compat import clamp_inclusive

LUMA2020 = (0.2627, 0.6780, 0.0593)
FLOOR = 1e-7          # 0.001 nit in network units (nits / 10,000)


class ShadowCodecAugment:
    def __init__(self, probability: float, jpeg_quality=(10, 40), crush_max: float = 4 / 255,
                 shadow_lo: float = 16 / 255, shadow_hi: float = 48 / 255, seed: int = 0):
        if not 0.0 <= probability <= 1.0 or not 0 < shadow_lo < shadow_hi <= 1:
            raise ValueError("bad shadow-codec augmentation settings")
        self.p, self.q, self.crush = float(probability), tuple(int(v) for v in jpeg_quality), float(crush_max)
        self.lo, self.hi = float(shadow_lo), float(shadow_hi)
        self.rng = np.random.default_rng(seed)

    def state(self) -> dict:
        s = self.rng.bit_generator.state
        return {"bit_generator": s["bit_generator"], "state": {k: str(v) for k, v in s["state"].items()},
                "has_uint32": int(s["has_uint32"]), "uinteger": str(s["uinteger"])}

    def restore(self, saved: dict) -> None:
        self.rng.bit_generator.state = {"bit_generator": saved["bit_generator"],
                                        "state": {k: int(v) for k, v in saved["state"].items()},
                                        "has_uint32": int(saved["has_uint32"]), "uinteger": int(saved["uinteger"])}

    def _one(self, sdr: np.ndarray) -> np.ndarray:
        import cv2
        q = int(self.rng.integers(self.q[0], self.q[1] + 1))
        crush = float(self.rng.uniform(0.0, self.crush))
        code = np.clip(np.rint(sdr * 255.0), 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(code[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return sdr
        jpeg = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
        jpeg = np.clip((jpeg - crush) / (1.0 - crush), 0.0, 1.0)
        luma = sdr @ np.asarray(LUMA2020, np.float32)
        t = np.clip((self.hi - luma) / (self.hi - self.lo), 0.0, 1.0)
        m = (t * t * (3 - 2 * t))[..., None]                 # 1 in deep shadow, 0 above shadow_hi
        return (sdr * (1 - m) + jpeg * m).astype(np.float32)

    def __call__(self, sdr_batch: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) canonical sRGB code in [0,1], CPU. Draws are made for every sample."""
        out = sdr_batch.clone()
        for i in range(sdr_batch.shape[0]):
            if self.rng.random() < self.p:
                img = sdr_batch[i].permute(1, 2, 0).numpy().astype(np.float32)
                out[i] = torch.from_numpy(self._one(img)).permute(2, 0, 1)
        return out


def near_black_log_loss(pred: torch.Tensor, target: torch.Tensor, sdr: torch.Tensor,
                        code_threshold: float = 16 / 255) -> torch.Tensor:
    mask = (sdr.amax(1, keepdim=True) <= code_threshold).to(pred.dtype)
    if float(mask.sum()) == 0.0:
        return pred.sum() * 0.0
    w = torch.tensor(LUMA2020, device=pred.device, dtype=pred.dtype).view(1, 3, 1, 1)
    lp = torch.log(clamp_inclusive(pred, 0.0) + FLOOR)
    lt = torch.log(target.clamp_min(0) + FLOOR)
    lum = (torch.log((clamp_inclusive(pred, 0.0) * w).sum(1, keepdim=True) + FLOOR)
           - torch.log((target.clamp_min(0) * w).sum(1, keepdim=True) + FLOOR)).abs()
    opp_p = torch.cat((lp[:, 0:1] - lp[:, 1:2], lp[:, 2:3] - lp[:, 1:2]), 1)
    opp_t = torch.cat((lt[:, 0:1] - lt[:, 1:2], lt[:, 2:3] - lt[:, 1:2]), 1)
    chroma = (opp_p - opp_t).abs().mean(1, keepdim=True)
    return ((lum + chroma) * mask).sum() / mask.sum()


def pu_min_score(metrics: dict) -> float:
    """Lower is better (the trainer minimises): minus the worse of the two PU gains."""
    return -min(float(metrics["clean_pu_gain_db"]), float(metrics["hard_pu_gain_db"]))
