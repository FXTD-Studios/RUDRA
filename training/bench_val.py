"""Bench-like validation for best.pt selection (v10, 13 Oct 2026).

Every best.pt from v4 to v9 was selected on 256-px centre crops scored in
log-PSNR, which does not predict the bench (full frames, PU21/CVVDP; STATUS
line E). v9's best.pt never left step 0, because the selection set had no
exposure push while the model was learning exactly that. This scores what the
gates score, on held-out validation frames:

  real      live-action real SDR (sdr_kind "real") at full frame (long side
            ``max_side``): PU21-PSNR of the model minus that of its baseline
            (the Source picker's BT.1886), in dB.
  rendered  rendered validation frames the same way, each on its own curve.
  clip      the real frames' HDR re-rendered with ACES at +E EV
            (sdr2hdr_dataset.exposure_push, the clip benches' render): the
            error inside the SDR's clipped pixels, baseline minus model, in
            stops of luminance (training/score_regions.py's measure).
  crushed   the same inside crushed pixels (max code <= 1/255) of the real
            frames: baseline minus model, stops.

    score = real + min(0, rendered) + 4 x clip + 4 x min(0, crushed)

higher is better (train_sdr2hdr --best-metric bench_gain negates it). One
stop inside the clip weighs 4 dB: the size of the v8-to-clip-fill gap.
Frames are read once and kept (half precision) for every later evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

LUMA_2020 = torch.tensor([0.2627, 0.6780, 0.0593])
FLOOR_NITS = 0.005
PEAK_NITS = 10000.0
CLIP_WEIGHT_DB_PER_STOP = 4.0
CLIPPED, CRUSHED = 254.0 / 255.0, 1.0 / 255.0
MIN_REGION_PIXELS = 256

# PU21 banding+glare (rudra/delivery/bench.py), in torch.
_PU21 = (0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142)


def pu21(nits: torch.Tensor) -> torch.Tensor:
    p0, p1, p2, p3, p4, p5, p6 = _PU21
    yp = nits.double().clamp(FLOOR_NITS, PEAK_NITS).pow(p3)
    return p6 * (((p0 + p1 * yp) / (1.0 + p2 * yp)).pow(p4) - p5)


_PU_PEAK = float(pu21(torch.tensor(PEAK_NITS)))


def pu_psnr(test_nits: torch.Tensor, ref_nits: torch.Tensor) -> float:
    mse = float(((pu21(test_nits) - pu21(ref_nits)) ** 2).mean())
    return float("inf") if mse <= 0 else 10.0 * math.log10(_PU_PEAK ** 2 / mse)


def log_luma(network_units: torch.Tensor) -> torch.Tensor:
    """(3,H,W) network units -> (H,W) log2 Rec.2020 luminance in nits, floored."""
    y = (network_units.double().clamp_min(0.0) * PEAK_NITS * LUMA_2020.double().to(network_units.device)[:, None, None]).sum(0)
    return torch.log2(y.clamp_min(FLOOR_NITS))


def region_error(test: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> float | None:
    if int(mask.sum()) < MIN_REGION_PIXELS:
        return None
    return float((log_luma(test) - log_luma(ref)).abs()[mask].mean())


def fit(x: torch.Tensor, max_side: int) -> torch.Tensor:
    """(3,H,W) -> long side at most ``max_side`` (area), as export_bench_pairs --max-side."""
    h, w = x.shape[-2:]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        x = F.interpolate(x[None].float(), size=(max(1, round(h * s)), max(1, round(w * s))), mode="area")[0]
    return x


@dataclass
class Frame:
    sdr: torch.Tensor          # (3,H,W) half, codes
    hdr: torch.Tensor          # (3,H,W) half, network units
    real: bool
    baseline_curve: int
    source_curve: int


@dataclass
class BenchVal:
    """Held-out frames chosen once, in a fixed order, and kept in memory."""
    records: list[dict]
    max_side: int = 960
    push_evs: tuple[float, ...] = (1.0, 2.0)
    frames: list[Frame] = field(default_factory=list)

    @classmethod
    def from_manifest(cls, manifest: str, split: str = "val", real_frames: int = 48, rendered_frames: int = 48,
                      max_side: int = 960, push_evs=(1.0, 2.0), seed: int = 20261013) -> "BenchVal":
        from training.sdr2hdr_dataset import read_jsonl
        rows = [r for r in read_jsonl(manifest) if r.get("split") == split]
        rng = np.random.default_rng(seed)
        real = [r for r in rows if r.get("sdr_kind") == "real"]
        rendered = [r for r in rows if r.get("sdr_kind") != "real"]
        pick = lambda xs, n: [xs[i] for i in sorted(rng.permutation(len(xs))[:n])] if n > 0 else []
        chosen = pick(real, real_frames) + pick(rendered, rendered_frames)
        if not chosen:
            raise ValueError(f"no {split!r} records in {manifest} for the bench validation")
        return cls(chosen, max_side, tuple(float(e) for e in push_evs))

    def load(self) -> None:
        if self.frames:
            return
        from training.sdr2hdr_dataset import (effective_baseline_curve, effective_source_curve, load_rgb)
        for r in self.records:
            sdr = torch.from_numpy(np.ascontiguousarray(load_rgb(r["sdr_path"], hdr=False))).permute(2, 0, 1)
            hdr = torch.from_numpy(np.ascontiguousarray(load_rgb(r["hdr_path"], hdr=True))).permute(2, 0, 1)
            sdr, hdr = fit(sdr, self.max_side), fit(hdr, self.max_side)
            self.frames.append(Frame(sdr.half(), hdr.clamp_min(0.0).half(), r.get("sdr_kind") == "real",
                                     effective_baseline_curve(r, False, False, False, 0.0),
                                     effective_source_curve(r, False, False, 0.0)))

    @torch.no_grad()
    def evaluate(self, model, device: torch.device) -> dict[str, float]:
        from rudra.source_baseline import baseline_curve_index
        from training.sdr2hdr_dataset import exposure_push

        self.load()
        was = model.training
        model.eval()
        use_picker = bool(getattr(model, "baseline_source", False))
        use_source = bool(getattr(model, "source_curve", False))

        def run(sdr: torch.Tensor, baseline_curve: int, source_curve: int):
            x = sdr.float()[None].to(device)
            kw = {}
            if use_picker:
                kw["baseline_curve"] = torch.tensor([baseline_curve], device=device)
            if use_source and source_curve:
                kw["source_curve"] = torch.tensor([source_curve], device=device)
            out = model(x, **kw)
            return out.hdr[0].float(), out.analytic_baseline[0].float() if out.analytic_baseline is not None \
                else out.baseline[0].float()

        gains = {"real": [], "rendered": []}
        clip, crushed = [], []
        aces = baseline_curve_index("aces")
        for f in self.frames:
            ref = f.hdr.float().to(device)
            pred, base = run(f.sdr, f.baseline_curve, f.source_curve)
            gains["real" if f.real else "rendered"].append(
                pu_psnr(pred * PEAK_NITS, ref * PEAK_NITS) - pu_psnr(base * PEAK_NITS, ref * PEAK_NITS))
            if f.real:
                top = f.sdr.float().amax(0).to(device)
                e_m, e_b = region_error(pred, ref, top <= CRUSHED), region_error(base, ref, top <= CRUSHED)
                if e_m is not None:
                    crushed.append(e_b - e_m)
                for ev in self.push_evs:
                    sdr_p, hdr_p, _ = exposure_push(f.hdr.float(), ev)
                    ref_p = hdr_p.float().to(device)
                    pred_p, base_p = run(sdr_p, aces if use_picker else 0, 0)
                    mask = sdr_p.float().amax(0).to(device) >= CLIPPED
                    e_m, e_b = region_error(pred_p, ref_p, mask), region_error(base_p, ref_p, mask)
                    if e_m is not None:
                        clip.append(e_b - e_m)
        model.train(was)
        mean = lambda xs: float(np.mean(xs)) if xs else 0.0
        out = {
            "bench_real_gain_db": mean(gains["real"]), "bench_real_frames": len(gains["real"]),
            "bench_rendered_gain_db": mean(gains["rendered"]), "bench_rendered_frames": len(gains["rendered"]),
            "bench_clip_gain_stops": mean(clip), "bench_clip_frames": len(clip),
            "bench_crushed_gain_stops": mean(crushed), "bench_crushed_frames": len(crushed),
        }
        out["bench_gain"] = (out["bench_real_gain_db"] + min(0.0, out["bench_rendered_gain_db"])
                             + CLIP_WEIGHT_DB_PER_STOP * out["bench_clip_gain_stops"]
                             + CLIP_WEIGHT_DB_PER_STOP * min(0.0, out["bench_crushed_gain_stops"]))
        return out
