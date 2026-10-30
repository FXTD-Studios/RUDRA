"""How bright a clipped area was: a small, deterministic, gateable estimator (layer 3 v3, 30 Oct 2026).

v1-v2 asked the generative model for both the SHAPE inside a clipped area and its BRIGHTNESS. The
shape helped; the brightness was off by +-0.3..1.5 stops either way (claude/TASK_GENFILL_LAYER3).
v3 splits them:

    fill log2 Y(x) = log2 Y_master(x) + max(0, b_c + alpha * s(x))

    b_c   the mean excess (stops over the master) of clipped component c: this estimator, plus the
          artist's offset
    s(x)  the generated fill's own shape inside c, zero mean (rudra.genfill.run)

The estimator reads only what the 8-bit source holds around and inside the component (its size and
outline, how fast the picture rises towards it, the channels that did not clip, the frame's exposure),
never the truth, and is a tiny MLP (14 -> 32 -> 32 -> 1) trained with an asymmetric loss: an
overshoot (more light than there was) costs `over_weight` times an undershoot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

AP1_Y = np.array([.2722287168, .6740817658, .0536895174], np.float32)
FEATURES = ['log2_area', 'level', 'min_chan', 'all_three', 'ring1_rel', 'ring1_near', 'ring2_rel', 'slope',
            'ring_std', 'compact', 'frame_rel', 'frame_clipped', 'centroid_y', 'ring_sat']
MIN_PIXELS = 16


def log2_lum(rgb: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(rgb @ AP1_Y, 1e-6)).astype(np.float32)


def components(codes: np.ndarray, min_pixels: int = MIN_PIXELS):
    """codes (H, W, 3) uint8 -> (labels (H, W) int32, [component ids]) of the fully clipped areas
    (max code 255), 8-connected, at least `min_pixels`."""
    import cv2
    core = (codes.max(-1) >= 255).astype(np.uint8)
    n, lab = cv2.connectedComponents(core, connectivity=8)
    ids = [c for c in range(1, n) if (lab == c).sum() >= min_pixels]
    return lab.astype(np.int32), ids


def _ring(mask: np.ndarray, r0: int, r1: int) -> np.ndarray:
    import cv2
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    outer = cv2.dilate(mask.astype(np.uint8), k(r1)) > 0
    inner = cv2.dilate(mask.astype(np.uint8), k(r0)) > 0 if r0 > 0 else mask
    return outer & ~inner


def frame_features(codes: np.ndarray, master: np.ndarray):
    """Per-frame context shared by its components."""
    ly = log2_lum(master)
    mx = codes.max(-1)
    return {'ly': ly, 'median': float(np.median(ly)), 'clipped': float((mx >= 255).mean()), 'mx': mx}


def component_features(codes: np.ndarray, master: np.ndarray, mask: np.ndarray, frame=None) -> np.ndarray:
    """The FEATURES of one clipped component, from the SDR codes and the master (exact inverse) only."""
    import cv2
    fr = frame or frame_features(codes, master)
    ly, mx = fr['ly'], fr['mx']
    h, w = mask.shape
    area = float(mask.sum())
    level = float(ly[mask].mean())
    cm = codes[mask].astype(np.float32)
    min_chan = float(cm.min(-1).mean() / 255.0)
    all_three = float((cm.min(-1) >= 255).mean())
    r1, r2 = _ring(mask, 0, 3), _ring(mask, 4, 10)
    r1 &= mx < 255
    r2 &= mx < 255
    l1 = float(ly[r1].mean()) if r1.any() else level
    l2 = float(ly[r2].mean()) if r2.any() else l1
    near = float((mx[r1] >= 250).mean()) if r1.any() else 1.0
    std = float(ly[r1 | r2].std()) if (r1 | r2).any() else 0.0
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = float(sum(cv2.arcLength(c, True) for c in cnts)) or 1.0
    ys = np.nonzero(mask)[0]
    rr = master[r1] if r1.any() else master[mask]
    sat = float(((rr.max(-1) - rr.min(-1)) / np.maximum(rr.max(-1), 1e-6)).mean())
    return np.array([np.log2(area / (h * w)), level, min_chan, all_three, l1 - level, near, l2 - level,
                     (l1 - l2) / 5.5, std, min(1.0, 4 * np.pi * area / per ** 2), fr['median'] - level,
                     fr['clipped'], float(ys.mean() / h), sat], np.float32)


class BrightnessEstimator:
    """features -> b (stops over the master), >= 0. A tiny MLP with standardised inputs."""

    def __init__(self, state: dict | None = None):
        self._build()
        self.mu = np.zeros(len(FEATURES), np.float32)
        self.sd = np.ones(len(FEATURES), np.float32)
        self.meta: dict = {}
        if state:
            self.load_state(state)

    def _build(self):
        import torch
        self.net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.GELU(),
                                       torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))

    @staticmethod
    def _out(raw):
        import torch
        return torch.nn.functional.softplus(raw[:, 0])          # >= 0 and never a dead gradient

    def predict(self, feats: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(((np.atleast_2d(feats) - self.mu) / self.sd).astype(np.float32))
        with torch.no_grad():
            return self._out(self.net(x)).numpy()

    def fit(self, X: np.ndarray, y: np.ndarray, weights=None, over_weight: float = 4.0, epochs: int = 300,
            lr: float = 3e-3, seed: int = 0):
        import torch
        torch.manual_seed(seed)
        self._build()                                            # seeded weights: the fit is reproducible
        self.mu = X.mean(0).astype(np.float32)
        self.sd = (X.std(0) + 1e-6).astype(np.float32)
        xt = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
        yt = torch.from_numpy(y.astype(np.float32))
        wt = torch.from_numpy((weights if weights is not None else np.ones(len(y))).astype(np.float32))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-3)
        g = torch.Generator().manual_seed(seed)
        n = len(y)
        for _ in range(epochs):
            for idx in torch.randperm(n, generator=g).split(512):
                d = self._out(self.net(xt[idx])) - yt[idx]
                loss = (wt[idx] * (over_weight * torch.relu(d) ** 2 + torch.relu(-d) ** 2)).sum() / wt[idx].sum()
                opt.zero_grad(); loss.backward(); opt.step()
        self.meta.update({'over_weight': over_weight, 'epochs': epochs, 'lr': lr, 'seed': seed, 'n': int(n)})
        return self

    def state(self) -> dict:
        return {'features': FEATURES, 'mu': self.mu.tolist(), 'sd': self.sd.tolist(), 'meta': self.meta,
                'weights': {k: v.detach().cpu().numpy().tolist() for k, v in self.net.state_dict().items()},
                'format': 'rudra_genfill_brightness_v1'}

    def load_state(self, s: dict):
        import torch
        if s.get('format') != 'rudra_genfill_brightness_v1' or s.get('features') != FEATURES:
            raise ValueError('not a RUDRA brightness estimator for these features')
        self.mu = np.array(s['mu'], np.float32)
        self.sd = np.array(s['sd'], np.float32)
        self.meta = s.get('meta', {})
        self.net.load_state_dict({k: torch.tensor(v) for k, v in s['weights'].items()})

    def save(self, path):
        Path(path).write_text(json.dumps(self.state()))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))


from dataclasses import dataclass, field


@dataclass
class BrightnessSettings:
    """How run.fill_master sets the fill's brightness (v3): the estimator (None = b 0, shape only), the
    artist's offset in stops, the generated shape's gain, and the time smoothing of b."""
    estimator: 'BrightnessEstimator | None' = None
    offset: float = 0.0
    shape_gain: float = 1.0
    ema: float = 0.7
    identity: dict = field(default_factory=dict)


def excess_map(codes: np.ndarray, master: np.ndarray, estimator: BrightnessEstimator | None, offset: float = 0.0):
    """(H, W) b in stops for every fully clipped pixel (its component's estimate + the artist's offset),
    NaN elsewhere; plus the component labels."""
    lab, ids = components(codes)
    out = np.full(codes.shape[:2], np.nan, np.float32)
    if not ids:
        return out, lab
    fr = frame_features(codes, master)
    feats = np.stack([component_features(codes, master, lab == c, fr) for c in ids])
    b = estimator.predict(feats) if estimator is not None else np.zeros(len(ids), np.float32)
    for c, bc in zip(ids, b):
        out[lab == c] = max(0.0, float(bc) + offset)
    return out, lab


def compose(master: np.ndarray, generated: np.ndarray, b_map: np.ndarray, labels: np.ndarray,
            shape_gain: float = 1.0) -> np.ndarray:
    """The v3 fill: master brightness lifted by b, with the generated fill's zero-mean shape inside each
    component and the generated colour (its RGB ratios); nothing changes where b_map is NaN."""
    out = np.array(master, np.float32, copy=True)
    lm, lg = log2_lum(master), log2_lum(generated)
    for c in np.unique(labels[np.isfinite(b_map)]):
        s = (labels == c) & np.isfinite(b_map)
        rel = lg[s] - lm[s]
        shape = rel - rel.mean()
        ly = lm[s] + np.maximum(0.0, b_map[s] + shape_gain * shape)
        g = generated[s]
        yg = np.maximum(g @ AP1_Y, 1e-6)[:, None]
        ratio = np.where(yg > 1e-5, g / yg, master[s] / np.maximum(master[s] @ AP1_Y, 1e-6)[:, None])
        out[s] = ratio * (2.0 ** ly)[:, None]
    return out
