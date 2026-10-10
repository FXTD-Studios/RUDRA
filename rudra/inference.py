"""Frame inference for SDR2HDRNet: tiling, feathering and per-frame heads.

The one implementation of how a frame is cut into tiles, how the tiles are
blended back, and which quantities are computed ONCE per frame and handed to
every tile (residual scale, shadow weight, tone curve). It lived in
training/infer_sdr2hdr.py, which the installed package does not ship, and
`rudra video` carried a second copy of the tiler that drifted from it -- the
per-tile curve fixed on 9 Oct 2026 was that drift. training/infer_sdr2hdr.py
re-exports everything here, so existing imports keep working.

tools/export_model.py records ``_tile_weight`` in the model package as the
feathering reference; native/infer/src/tiler.cpp mirrors it.
"""

from __future__ import annotations

import contextlib

import torch

from .sdr2hdr import SDR2HDRNet

__all__ = ["predict_image", "predict_fields"]


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _tile_weight(height: int, width: int, overlap: int, y: int, x: int,
                 full_h: int, full_w: int, device: torch.device) -> torch.Tensor:
    wy = torch.ones(height, device=device, dtype=torch.float32)
    wx = torch.ones(width, device=device, dtype=torch.float32)
    fy, fx = min(overlap, height // 2), min(overlap, width // 2)
    if y > 0 and fy:
        wy[:fy] = torch.linspace(1e-3, 1.0, fy, device=device)
    if y + height < full_h and fy:
        wy[-fy:] = torch.linspace(1.0, 1e-3, fy, device=device)
    if x > 0 and fx:
        wx[:fx] = torch.linspace(1e-3, 1.0, fx, device=device)
    if x + width < full_w and fx:
        wx[-fx:] = torch.linspace(1.0, 1e-3, fx, device=device)
    return (wy[:, None] * wx[None, :])[None, None]


@torch.inference_mode()
def predict_image(model: SDR2HDRNet, sdr: torch.Tensor, preserve_outside: bool,
                  tile_size: int, overlap: int, recovery_mode: str = "all",
                  recovery_strength: float = 1.0, bf16: bool = True,
                  source_curve: int | None = None,
                  shadow_weight: float | torch.Tensor | None = None) -> torch.Tensor:
    """Memory-bounded image inference with overlap feathering.

    ``source_curve``: an id from rudra.sdr2hdr.SOURCE_CURVES for a model built
    with ``source_curve=True``; None or 0 is unknown (the blind estimate).

    ``bf16=False`` runs CUDA in fp32. The analytic baseline is always fp32, and
    bf16 keeps 8 mantissa bits (0.4 % steps), so on a clean frame the baseline
    scores near-exactly and a bf16 prediction cannot. Use fp32 whenever the two
    are compared.

    ``shadow_weight``: the per-frame shadow weight to use instead of the one
    the model predicts from this frame. `rudra video` passes its temporally
    smoothed value here; None (the default) predicts it, as before.
    """
    if sdr.shape[0] != 1:
        raise ValueError("predict_image expects one image at a time")
    _, _, height, width = sdr.shape
    # The conditioning head reads whole-frame statistics, so its scale is
    # computed once here and handed to every tile. Left to itself each tile
    # would predict from its own window and a patch of sky inside a dim
    # interior would reconstruct as if the whole frame were a sunset.
    scale = model.predict_residual_scale(sdr) if hasattr(model, "predict_residual_scale") else None
    if shadow_weight is None and hasattr(model, "predict_shadow_weight"):
        shadow_weight = model.predict_shadow_weight(sdr)
    # Same reason for the curve: one tone-curve estimate per FRAME, or every
    # tile would invert its own guess and the seams would show.
    source = None if not source_curve else torch.tensor([int(source_curve)], device=sdr.device)
    curve = (model.predict_curve(sdr, source) if source is not None else model.predict_curve(sdr)) \
        if hasattr(model, "predict_curve") else None
    if source is not None and curve is None:
        raise ValueError("a source curve was given but this model has no CurveHead")
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if (sdr.is_cuda and bf16) else contextlib.nullcontext()
        with amp:
            return model(sdr, preserve_outside=preserve_outside,
                         recovery_mode=recovery_mode,
                         residual_strength=recovery_strength,
                         residual_scale=scale, shadow_weight=shadow_weight,
                         curve_params=curve).hdr.float()
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must be >= 0 and smaller than tile size")
    result = torch.zeros((1, 3, height, width), device=sdr.device, dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), device=sdr.device, dtype=torch.float32)
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile = sdr[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if (tile.is_cuda and bf16) else contextlib.nullcontext()
            with amp:
                prediction = model(tile, preserve_outside=preserve_outside,
                                   recovery_mode=recovery_mode,
                                   residual_strength=recovery_strength,
                                   residual_scale=scale, shadow_weight=shadow_weight,
                         curve_params=curve).hdr.float()
            weight = _tile_weight(tile.shape[-2], tile.shape[-1], overlap, y, x,
                                  height, width, sdr.device)
            result[..., y:y + tile.shape[-2], x:x + tile.shape[-1]] += prediction * weight
            weights[..., y:y + tile.shape[-2], x:x + tile.shape[-1]] += weight
    return result / weights.clamp_min(1e-6)


@torch.inference_mode()
def predict_fields(model: SDR2HDRNet, sdr: torch.Tensor, tile_size: int,
                   overlap: int) -> dict[str, torch.Tensor]:
    """The raw head fields, stitched: log residual, highlight mask, shadow mask.

    ``recovery_mode``, ``residual_strength`` and ``preserve_outside`` never
    touch these three -- they only enter the composition that happens after.
    So a client that holds the fields can rebuild any combination of them
    without another forward pass, which is what lets RUDRA Studio's controls
    run at frame rate instead of at one HTTP round trip each.

    Feathering is applied to the fields rather than to the composed
    prediction, so inside an overlap band a tiled frame differs from
    ``predict_image`` by the difference between blending before and after
    ``expm1``. Measured on a 160x160 frame with 64/16 tiles that reaches a few
    percent at the worst pixel of a band -- not a rounding error -- while an
    untiled pass agrees with ``predict_image`` to about 3e-6. That is why the
    viewer asks for an untiled pass and falls back to tiles only when it runs
    out of memory, and why the header it gets back says which happened.
    tests/test_frame_fields_2026_08_28.py pins both numbers.
    """
    if sdr.shape[0] != 1:
        raise ValueError("predict_fields expects one image at a time")
    _, _, height, width = sdr.shape

    scale = model.predict_residual_scale(sdr) if hasattr(model, "predict_residual_scale") else None
    # The three fields do not depend on the shadow weight (it scales the
    # prior in the composite, after them), so this only saves forward() an
    # encode it would otherwise run per tile. The weight itself is what
    # ui/server.py sends the page beside the fields.
    # (Named in full: `shadow` is the stitched mask further down, and a
    # closure reads the name at call time, not at definition.)
    shadow_weight = model.predict_shadow_weight(sdr) \
        if hasattr(model, "predict_shadow_weight") else None
    curve = model.predict_curve(sdr) if hasattr(model, "predict_curve") else None

    def _run(tile: torch.Tensor):
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if tile.is_cuda \
            else contextlib.nullcontext()
        with amp:
            out = model(tile, preserve_outside=False, recovery_mode="all",
                        residual_strength=1.0, residual_scale=scale,
                        shadow_weight=shadow_weight, curve_params=curve)
        residual = out.log_residual.float()
        # The viewer composes from these three fields alone and knows nothing
        # about a conditioning head, so the per-frame scale is folded into the
        # residual here. The GLSL composite then stays a line-for-line port of
        # forward()'s tail, and the Studio's strength slider remains a control
        # on top of the model's judgement rather than a replacement for it.
        if out.residual_scale is not None:
            residual = residual * out.residual_scale.float()
        return (residual, out.highlight_mask.float(), out.shadow_mask.float())

    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        residual, highlight, shadow = _run(sdr)
        return {"residual": residual, "highlight": highlight,
                "shadow": shadow, "tiled": False, "curve": curve}

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must be >= 0 and smaller than tile size")
    device = sdr.device
    residual = torch.zeros((1, 3, height, width), device=device, dtype=torch.float32)
    highlight = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    shadow = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile = sdr[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            r, h, s = _run(tile)
            th, tw = tile.shape[-2], tile.shape[-1]
            weight = _tile_weight(th, tw, overlap, y, x, height, width, device)
            residual[..., y:y + th, x:x + tw] += r * weight
            highlight[..., y:y + th, x:x + tw] += h * weight
            shadow[..., y:y + th, x:x + tw] += s * weight
            weights[..., y:y + th, x:x + tw] += weight
    weights = weights.clamp_min(1e-6)
    return {"residual": residual / weights, "highlight": highlight / weights,
            "shadow": shadow / weights, "tiled": True, "curve": curve}
