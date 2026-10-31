#!/usr/bin/env python3
"""run05 corpus: "virtual RAW" pairs for generated-content SDR (22 Oct 2026).

Product target (docs/production/PRODUCT_TARGET.md): the master must look exactly
like the source under the DECLARED VIEW (round trip within 1 code value) except
where range was truly lost, and there it carries the real scene. This builder
makes training pairs that satisfy that contract by construction:

    plate     scene-linear BT.2020 frame from an existing corpus row
              (camera plates, Poly Haven, CC0 renders), network units
    exposure  auto key toward a random mid-grey key, clamped
    look      one per pair, drawn from many families (ACES, Hable, Reinhard,
              AgX, camera log, clip; "filmic_s" is held out for evaluation),
              with contrast / saturation jitter, in Rec.709
    input     8-bit quantisation, optional JPEG / video codec, optional
              generator-VAE round trip (pluggable), then the canonical
              BT.2020 sRGB code the app and trainer use (uint16 PNG)
    target    where the look did not lose range:
                  declared_inverse(canonical(look before quantisation))
              i.e. the dequantised, deblocked inverse of the view -- its round
              trip IS the source, and it is smooth (the "grade headroom" truth)
              where the look clipped / crushed:
                  the true plate, rescaled so it joins the inverse at the
                  boundary (one luminance gain fitted just below the clip /
                  just above the crush), blended in log over a short band
    mask      0 = exact inverse, 255 = reconstructed (PNG, beside the pair)

The declared view is a Source-picker curve (rudra/source_baseline.py); the
default "bt1886" is the app's default, so the same pairs train the model the
app runs. HDR targets use pipeline/hdr_io log2_extended with a sentinel.

    python pipeline/run05_corpus.py build --sources sources.jsonl --out G:/datasets/corpora/rudra_run05_v1 \
        --variants 2 --workers 8
    python pipeline/run05_corpus.py build --sources sources.jsonl --out ..._heldout --families filmic_s --splits val

sources.jsonl rows: {source_id, scene_id, split, asset_id, hdr_path (decodable
by training/sdr2hdr_dataset.load_rgb), ceiling_nits?, licence_source?,
licence_licence?, commercial_ok?}. Evaluation-only scenes (Tears of Steel,
Mataha, SI-HDR) are refused by name.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BUILDER_VERSION = "run05-corpus-v1"
LUMA2020 = np.array([0.2627, 0.6780, 0.0593], np.float32)
NETWORK_PER_SCENE = 203.0 / 10000.0          # scene 1.0 (diffuse white) in network units
TRAIN_FAMILIES = {"aces": 0.22, "hable": 0.15, "reinhard": 0.12, "agx": 0.17, "camera_log": 0.20, "clip": 0.14}
HELDOUT_FAMILIES = ("filmic_s",)
FORBIDDEN_SCENES = ("tears", "tos_", "mataha", "sihdr", "si-hdr", "si_hdr")
HI_BAND, LO_BAND = (0.94, 0.99), (1.0 / 255, 5.0 / 255)    # code where reconstruction blends in
HI_FIT, LO_FIT = (0.80, 0.94), (6.0 / 255, 24.0 / 255)     # code band where the gain is fitted

M2020_TO_709 = np.array([[1.6605, -0.5876, -0.0728],
                         [-0.1246, 1.1329, -0.0083],
                         [-0.0182, -0.1006, 1.1187]], np.float32)


# ------------------------------------------------------------------ looks
def curve_filmic_s(x: np.ndarray, contrast: float = 1.0, **_) -> np.ndarray:
    """Held-out family: a log-space logistic S with a soft toe (resembles NLE "filmic" looks)."""
    t = (np.log2(np.maximum(x, 2.0 ** -10)) + 2.47) / 6.5              # mid grey 0.18 -> ~0
    s = 1.0 / (1.0 + np.exp(-3.2 * contrast * t))
    s0, s1 = 1.0 / (1.0 + np.exp(3.2 * contrast * (10 - 2.47) / 6.5)), 1.0
    return np.clip((s - s0) / (s1 - s0), 0.0, 1.0) ** 1.15


def look_curves() -> dict:
    from pipeline.sdr_render import CURVES
    return {**CURVES, "filmic_s": curve_filmic_s}


def draw_look(rng: np.random.Generator, families: list[str], codec_p: float, codecs: list[str],
              vae_p: float) -> dict:
    weights = np.array([TRAIN_FAMILIES.get(f, 1.0) for f in families], np.float64)
    look = {
        "family": str(rng.choice(families, p=weights / weights.sum())),
        "key": float(np.exp(rng.uniform(np.log(0.09), np.log(0.30)))),
        "key_strength": float(rng.uniform(0.5, 1.0)),
        "contrast": float(rng.uniform(0.85, 1.2)),
        "saturation": float(rng.uniform(0.85, 1.2)),
        "white": float(rng.uniform(4.0, 16.0)),
        "oetf": str(rng.choice(["srgb", "rec709"], p=[0.7, 0.3])),
        "codec": "none", "quality": None, "vae": False,
    }
    if rng.random() < codec_p:
        pool = ["jpeg", *codecs]
        look["codec"] = str(rng.choice(pool))
        look["quality"] = int(rng.integers(30, 90)) if look["codec"] == "jpeg" else int(rng.integers(20, 36))
    look["vae"] = bool(rng.random() < vae_p)
    return look


def exposure_ev(scene: np.ndarray, look: dict) -> float:
    y = np.maximum(scene @ LUMA2020, 0.0)
    key = float(np.exp(np.mean(np.log(y + 1e-4))))
    return float(np.clip(look["key_strength"] * np.log2(look["key"] / max(key, 1e-6)), -4.0, 4.0))


def render_look(scene2020: np.ndarray, look: dict, ev: float) -> np.ndarray:
    """Scene-linear BT.2020 (1.0 = diffuse white) -> Rec.709 code, FLOAT [0,1], unquantised."""
    from pipeline.sdr_render import _rec709_oetf, _srgb_oetf
    x = np.maximum(scene2020 * np.float32(2.0 ** ev), 0.0) @ M2020_TO_709.T
    x = np.maximum(x, 0.0)
    disp = look_curves()[look["family"]](x, white=look["white"], contrast=look["contrast"])
    grey = (disp @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    disp = np.clip(grey + look["saturation"] * (disp - grey), 0.0, 1.0)
    code = _srgb_oetf(disp) if look["oetf"] == "srgb" else _rec709_oetf(disp)
    return np.clip(code, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------- canonical + declared view
def canonical(code709: np.ndarray) -> np.ndarray:
    """Rec.709 code (treated as sRGB, as the app does) -> canonical BT.2020 sRGB code, float."""
    import torch
    from rudra.delivery.colorspace import rgb_to_rgb_matrix
    from rudra.sdr2hdr import linear_to_srgb, srgb_to_linear
    x = torch.from_numpy(np.ascontiguousarray(code709, np.float32)).permute(2, 0, 1)[None]
    m = torch.as_tensor(rgb_to_rgb_matrix("rec709", "rec2020"), dtype=x.dtype)
    out = linear_to_srgb(torch.einsum("ij,bjhw->bihw", m, srgb_to_linear(x))).clamp(0, 1)
    return out[0].permute(1, 2, 0).numpy().astype(np.float32)


def declared_inverse(canon: np.ndarray, view: str = "bt1886", corpus_ev: float = 0.0) -> np.ndarray:
    """The app's analytic baseline for the picked Source curve: canonical code -> network units."""
    import torch
    from rudra.sdr2hdr import sdr_to_baseline_hdr
    from rudra.source_baseline import baseline_curve_index, correction_log2, knots_table
    x = torch.from_numpy(np.ascontiguousarray(canon, np.float32)).permute(2, 0, 1)[None]
    ids = torch.tensor([baseline_curve_index(view)])
    out = sdr_to_baseline_hdr(x, corpus_ev) * torch.exp2(correction_log2(x, ids, knots_table()))
    return out[0].permute(1, 2, 0).numpy().astype(np.float32)


# ------------------------------------------------------------------ the pair
def _smooth(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _gain(base_y, plate_y, sel, minimum=200):
    ok = sel & (plate_y > 1e-9) & (base_y > 1e-9)
    if ok.sum() < minimum:
        return None
    return float(np.exp(np.median(np.log(base_y[ok] / plate_y[ok]))))


def degrade(code8: np.ndarray, look: dict, vae=None) -> np.ndarray:
    from pipeline.sdr_render import _jpeg, _video_codec
    out = code8
    if look["codec"] == "jpeg":
        out = _jpeg(out, look["quality"])
    elif look["codec"] != "none":
        out = _video_codec(out, look["codec"], look["quality"])
    if look.get("vae") and vae is not None:
        out = np.clip(np.asarray(vae(out)), 0, 255).astype(np.uint8)
    return out


def build_pair(plate_net: np.ndarray, look: dict, view: str = "bt1886", vae=None,
               ceiling_net: float | None = None, target_mode: str = "restoration",
               reconstruction_policy: str = 'legacy_nearwhite') -> dict:
    """plate_net: (H,W,3) scene-linear BT.2020 in network units. Returns arrays + stats."""
    if target_mode not in ('restoration', 'source_preserving'):
        raise ValueError('unknown target mode')
    from rudra.reconstruction_policy import reconstruction_bands
    hi_band, lo_band = reconstruction_bands(reconstruction_policy)
    scene = plate_net / np.float32(NETWORK_PER_SCENE)
    ev = exposure_ev(scene, look)
    code_f = render_look(scene, look, ev)                         # float, pre-quantisation
    code8 = np.clip(np.rint(code_f * 255.0), 0, 255).astype(np.uint8)
    code8_in = degrade(code8, look, vae)
    canon_in = canonical(code8_in.astype(np.float32) / 255.0)     # the model's input
    canon_f = canonical(code_f)
    base_f = declared_inverse(canon_f, view)                      # smooth inverse = round-trip truth
    exposed = plate_net * np.float32(2.0 ** ev)                   # true scene, network units
    by, py = base_f @ LUMA2020, exposed @ LUMA2020
    cmax = code_f.max(axis=2)
    stats = {"render_ev": round(ev, 4)}
    target = base_f.copy()
    w_hi = _smooth(cmax, *hi_band)
    k_hi = _gain(by, py, (cmax >= HI_FIT[0]) & (cmax < HI_FIT[1]))
    if k_hi is not None and w_hi.max() > 0:
        ext = np.maximum(exposed * np.float32(k_hi), base_f)      # never darker than the inverse
        lt = np.log(np.maximum(target, 1e-9)) * (1 - w_hi[..., None]) + np.log(np.maximum(ext, 1e-9)) * w_hi[..., None]
        target = np.exp(lt).astype(np.float32)
    else:
        w_hi = np.zeros_like(cmax)
    w_lo = 1.0 - _smooth(cmax, *lo_band)
    k_lo = _gain(by, py, (cmax >= LO_FIT[0]) & (cmax < LO_FIT[1]))
    if k_lo is not None and w_lo.max() > 0:
        cap = float(declared_inverse(canonical(np.full((1, 1, 3), LO_BAND[1], np.float32)), view)[0, 0] @ LUMA2020)
        ext = exposed * np.float32(k_lo)
        ey = ext @ LUMA2020
        ext = ext * np.minimum(1.0, cap / np.maximum(ey, 1e-12))[..., None]   # never lift shadows
        lt = np.log(np.maximum(target, 1e-9)) * (1 - w_lo[..., None]) + np.log(np.maximum(ext, 1e-9)) * w_lo[..., None]
        target = np.where(w_lo[..., None] > 0, np.exp(lt), target).astype(np.float32)
    else:
        w_lo = np.zeros_like(cmax)
    mask = np.maximum(w_hi, w_lo)
    if target_mode == 'source_preserving':
        from training.contract_checks import enforce_contract, reconstruction_mask
        target = enforce_contract(target, canon_in, view, reconstruction_policy)
        mask = reconstruction_mask(canon_in, reconstruction_policy)
    stats.update(k_hi=k_hi, k_lo=k_lo, hi_fraction=float((w_hi > 0.5).mean()),
                 lo_fraction=float((w_lo > 0.5).mean()), target_peak_nits=float(target.max() * 1e4),
                 input_peak_code=int(code8_in.max()))
    if ceiling_net is not None and k_hi is not None:
        stats["ceiling_nits"] = float(ceiling_net * (2.0 ** ev) * k_hi * 1e4)
    return {"canon_in": canon_in, "target": target, "mask": mask, "code8_in": code8_in, "stats": stats}


# ------------------------------------------------------------------ IO
def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def storage():
    from pipeline.hdr_io import HDRStorage
    # Preserve the inverse view's near-black values rather than lifting every
    # nonzero channel to the legacy 0.005-nit storage floor.
    return HDRStorage(mode="log2_extended", floor_nits=1e-6)


def write_pair(out: Path, asset: str, pair: dict) -> dict:
    import cv2
    from pipeline.hdr_io import encode_hdr_u16
    paths = {k: out / k / f"{asset}.png" for k in ("sdr", "hdr", "mask")}
    sdr16 = np.clip(np.rint(pair["canon_in"] * 65535.0), 0, 65535).astype(np.uint16)
    hdr16, enc = encode_hdr_u16(pair["target"] / np.float32(NETWORK_PER_SCENE), storage())
    m8 = np.clip(np.rint(pair["mask"] * 255.0), 0, 255).astype(np.uint8)
    for key, img in (("sdr", sdr16[..., ::-1]), ("hdr", hdr16[..., ::-1]), ("mask", m8)):
        if not cv2.imwrite(str(paths[key]), np.ascontiguousarray(img)):
            raise RuntimeError(f"write failed: {paths[key]}")
    return {"paths": paths, "encode": enc}


def check_source(row: dict) -> None:
    text = f"{row.get('scene_id', '')} {row.get('asset_id', '')} {row.get('source_id', '')}".lower()
    if any(f in text for f in FORBIDDEN_SCENES):
        raise ValueError(f"evaluation-only scene refused: {row.get('asset_id')}")
    for k in ("source_id", "scene_id", "split", "asset_id", "hdr_path"):
        if not row.get(k):
            raise ValueError(f"source row missing {k}: {row}")


def _seed(asset: str, variant: int, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{asset}:{variant}".encode()).digest()[:8], "little")


def _work(job: tuple) -> list[dict]:
    row, variants, cfg = job
    plate = load_plate(row)
    ceiling = row.get("ceiling_nits")
    out, results = Path(cfg["out"]), []
    vae = _load_vae(cfg.get("vae"), json.dumps(cfg.get('vae_identity'), sort_keys=True)) if cfg.get("vae") else None
    for v in range(variants):
        asset = _slug(f"{row['asset_id']}__r05v{v}")
        if (out / "rows" / f"{asset}.json").exists():
            saved = json.loads((out / "rows" / f"{asset}.json").read_text(encoding="utf-8"))
            for kind in ("sdr", "hdr", "mask"):
                path = Path(saved[f"{kind}_path"])
                if not path.is_file() or _sha(path) != saved[f"{kind}_sha256"]:
                    raise ValueError(f"corrupt completed row {asset}: {kind}")
            results.append(saved)
            continue
        rng = np.random.default_rng(_seed(row["asset_id"], v, cfg["seed"]))
        look = draw_look(rng, cfg["families"], cfg["codec_p"], cfg["codecs"], cfg["vae_p"] if vae else 0.0)
        pair = build_pair(plate, look, cfg["view"], vae, None if ceiling is None else float(ceiling) / 1e4,
                          cfg['target_mode'], cfg['reconstruction_policy'])
        written = write_pair(out, asset, pair)
        p = written["paths"]
        rec = {
            "asset_id": asset, "scene_id": row["scene_id"], "split": row["split"], "source_id": row["source_id"],
            "project": row.get("project"), "parent_asset_id": row["asset_id"], "parent_hdr_sha256": row.get("hdr_sha256"),
            "sdr_path": str(p["sdr"].resolve()), "hdr_path": str(p["hdr"].resolve()), "mask_path": str(p["mask"].resolve()),
            "sdr_sha256": _sha(p["sdr"]), "hdr_sha256": _sha(p["hdr"]), "mask_sha256": _sha(p["mask"]),
            "width": int(pair["canon_in"].shape[1]), "height": int(pair["canon_in"].shape[0]),
            "sdr_kind": "rendered", "render_family": "run05_virtual_raw", "tonemap_ev": 0.0,
            "source_curve": "unknown", "sdr_curve": look["family"], "sdr_codec": look["codec"],
            "declared_view": cfg["view"], "look": look, **pair["stats"],
            "target_mode": cfg['target_mode'],
            "reconstruction_policy": cfg['reconstruction_policy'],
            "ceiling_nits": pair["stats"].get("ceiling_nits"),
            "is_video": False, "excluded": False, "builder": BUILDER_VERSION,
            "licence_source": row.get("licence_source"), "licence_licence": row.get("licence_licence"),
            "commercial_ok": row.get("commercial_ok"), "dataset_scope": row.get("dataset_scope"),
        }
        tmp = out / "rows" / f"{asset}.json.partial"
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, out / "rows" / f"{asset}.json")        # a finished row is the resume marker
        results.append(rec)
    return results


M709_TO_2020 = np.array([[0.6274, 0.3293, 0.0433],
                         [0.0691, 0.9195, 0.0114],
                         [0.0164, 0.0880, 0.8956]], np.float32)


def load_plate(row: dict) -> np.ndarray:
    """Scene-linear BT.2020 plate in network units. Primaries must be stated per row
    (`plate_primaries` rec2020 | rec709); EXR sources also need `exr_scale` (network units
    per EXR unit). PNG targets decode through their ingest sentinel. Nothing is guessed."""
    prim = row.get("plate_primaries")
    if prim not in ("rec2020", "rec709"):
        raise ValueError(f"{row.get('asset_id')}: plate_primaries must be 'rec2020' or 'rec709'")
    if str(row["hdr_path"]).lower().endswith(".exr"):
        from training.exr_any import read_rgb
        if not row.get("exr_scale"):
            raise ValueError(f"{row.get('asset_id')}: EXR source needs exr_scale")
        img = read_rgb(row["hdr_path"]).astype(np.float32) * np.float32(row["exr_scale"])
    else:
        from training.sdr2hdr_dataset import load_rgb
        img = load_rgb(row["hdr_path"], hdr=True, ceiling=1e6)
    if not np.isfinite(img).all():
        raise ValueError(f"{row.get('asset_id')}: nonfinite source plate")
    img = np.maximum(img, 0)
    if prim == "rec709":
        img = np.maximum(img @ M709_TO_2020.T, 0)
    return img.astype(np.float32)


@lru_cache(maxsize=1)
def _load_vae(spec: str, identity: str = ''):
    """`module:function` returning a callable uint8(H,W,3) -> uint8(H,W,3); see pipeline/generator_vae.py."""
    import importlib
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)()


def build(sources: Path, out: Path, variants: int = 1, families: list[str] | None = None,
          splits: list[str] | None = None, view: str = "bt1886", seed: int = 20261022,
          codec_p: float = 0.5, vae: str | None = None, vae_p: float = 0.3, workers: int = 1,
          limit: int | None = None, target_mode: str = 'restoration',
          reconstruction_policy: str = 'legacy_nearwhite') -> dict:
    from pipeline.hdr_io import selftest
    from pipeline.sdr_render import available_video_codecs
    from rudra.reconstruction_policy import reconstruction_bands
    reconstruction_bands(reconstruction_policy)
    families = list(families or TRAIN_FAMILIES)
    if target_mode not in ('restoration', 'source_preserving'):
        raise ValueError('unknown target mode')
    unknown = set(families) - set(look_curves())
    if unknown:
        raise ValueError(f"unknown look families {unknown}")
    rows = [json.loads(l) for l in sources.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not splits or r.get("split") in splits]
    for r in rows:
        check_source(r)
    if set(families) & set(HELDOUT_FAMILIES) and any(r["split"] == "train" for r in rows):
        raise ValueError("held-out look families must never be rendered into train rows")
    if limit:
        rows = rows[:limit]
    scenes, assets = {}, set()
    for row in rows:
        if scenes.setdefault(row['scene_id'], row['split']) != row['split']:
            raise ValueError(f"scene {row['scene_id']} crosses splits")
        asset = _slug(row['asset_id'])
        if asset in assets:
            raise ValueError(f"duplicate output asset: {asset}")
        assets.add(asset)
    codecs = available_video_codecs()
    vae_identity = None
    if vae:
        if vae != 'pipeline.generator_vae:wan_vae':
            raise ValueError('VAE factory requires a supported weight-identity provider')
        if workers != 1:
            raise ValueError('Wan VAE builds require workers=1 to avoid duplicate GPU models')
        from pipeline.generator_vae import wan_vae_identity
        vae_identity = wan_vae_identity()
    for d in ("sdr", "hdr", "mask", "rows"):
        (out / d).mkdir(parents=True, exist_ok=True)
    selftest(storage())
    sentinel = out / "_ingest_config.json"
    block = {"storage": storage().as_dict(), "builder": BUILDER_VERSION, "declared_view": view,
             "families": families, "seed": seed,
             "sources_sha256": _sha(sources), "variants": variants,
             "target_mode": target_mode,
             "reconstruction_policy": reconstruction_policy,
             "reconstruction_policy_sha256": _sha(REPO / 'rudra' / 'reconstruction_policy.py'),
             "dependency_sha256": {name: _sha(REPO / name) for name in (
                 'training/contract_checks.py', 'pipeline/sdr_render.py', 'pipeline/hdr_io.py',
                 'rudra/sdr2hdr.py', 'rudra/source_baseline.py', 'rudra/delivery/colorspace.py')},
             "splits": splits, "limit": limit, "codec_p": codec_p,
             "codecs": codecs, "vae": vae, "vae_p": vae_p, "vae_identity": vae_identity,
             "builder_sha256": _sha(Path(__file__))}
    if sentinel.exists():
        old = json.loads(sentinel.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in block} != block:
            raise ValueError("output directory was built with other settings; use a new --out")
    else:
        sentinel.write_text(json.dumps(block, indent=2), encoding="utf-8")
    cfg = {"out": str(out), "families": families, "view": view, "seed": seed, "codec_p": codec_p,
           "codecs": codecs, "vae": vae, "vae_p": vae_p, "target_mode": target_mode,
           "reconstruction_policy": reconstruction_policy,
           "vae_identity": vae_identity}
    jobs = [(r, variants, cfg) for r in rows]
    records = []
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, res in enumerate(pool.imap(_work, jobs, chunksize=2), 1):
                records += res
                if i % 50 == 0:
                    print(f"[run05] {i}/{len(jobs)} sources", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            records += _work(job)
            if i % 50 == 0:
                print(f"[run05] {i}/{len(jobs)} sources", flush=True)
    records.sort(key=lambda r: r["asset_id"])
    scenes = {}
    for r in records:
        if scenes.setdefault(r["scene_id"], r["split"]) != r["split"]:
            raise ValueError(f"scene {r['scene_id']} crosses splits")
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    summary = {"builder": BUILDER_VERSION, "rows": len(records), "sources": len(rows), "view": view,
               "families": {f: sum(r["sdr_curve"] == f for r in records) for f in families},
               "splits": {s: sum(r["split"] == s for r in records) for s in sorted({r["split"] for r in records})},
               "codecs": {c: sum(r["sdr_codec"] == c for r in records) for c in sorted({r["sdr_codec"] for r in records})},
               "vae_rows": sum(bool(r["look"].get("vae")) for r in records),
               "hi_rows": sum(r["hi_fraction"] > 0 for r in records), "lo_rows": sum(r["lo_fraction"] > 0 for r in records),
               "manifest_sha256": _sha(manifest), "release_authorized": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--variants", type=int, default=1)
    b.add_argument("--families", nargs="*")
    b.add_argument("--splits", nargs="*")
    b.add_argument("--view", default="bt1886")
    b.add_argument("--seed", type=int, default=20261022)
    b.add_argument("--codec-p", type=float, default=0.5)
    b.add_argument("--vae", help="module:function factory, e.g. pipeline.generator_vae:wan_vae")
    b.add_argument("--vae-p", type=float, default=0.3)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--limit", type=int)
    b.add_argument('--target-mode', choices=('restoration', 'source_preserving'), default='restoration')
    b.add_argument('--reconstruction-policy', choices=('legacy_nearwhite', 'observed_clip_v1'), default='legacy_nearwhite')
    a = ap.parse_args()
    print(json.dumps(build(a.sources, a.out, a.variants, a.families, a.splits, a.view, a.seed,
                           a.codec_p, a.vae, a.vae_p, a.workers, a.limit, a.target_mode, a.reconstruction_policy), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
