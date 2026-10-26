import json

import cv2
import numpy as np
import pytest

from pipeline import run05_corpus as rc
from training import contract_checks as cc

NET = rc.NETWORK_PER_SCENE


def scene(h=96, w=160, seed=0):
    """Scene-linear BT.2020 (1.0 = diffuse white): sky ramp, a sun far above white, a dark corner, texture."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.25 + 0.6 * x + 0.3 * y
    img[..., 1] = 0.30 + 0.5 * x + 0.2 * y
    img[..., 2] = 0.45 + 0.4 * x
    img *= (1 + 0.05 * rng.standard_normal((h, w, 1))).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    sun = (yy - 20) ** 2 + (xx - 120) ** 2 < 100
    img[sun] = [400.0, 380.0, 330.0]                         # ~80,000 nits
    img[60:, :40] *= 0.004                                   # crushed corner
    return np.maximum(img, 0)


LOOK = {"family": "aces", "key": 0.18, "key_strength": 0.0, "contrast": 1.0, "saturation": 1.0,
        "white": 8.0, "oetf": "srgb", "codec": "none", "quality": None, "vae": False}


def test_target_roundtrips_to_input_outside_mask_and_extends_the_sun():
    pair = rc.build_pair(scene() * np.float32(NET), LOOK)
    r = cc.roundtrip(pair["target"], pair["canon_in"], pair["mask"])
    assert r["r_pass"], r
    assert cc.roundtrip(pair["target"], pair["canon_in"])["r_pass"]     # input-derived mask too
    yy, xx = np.mgrid[:96, :160]
    sun = (yy - 20) ** 2 + (xx - 120) ** 2 < 49
    base_white = rc.declared_inverse(np.ones((1, 1, 3), np.float32))[0, 0, 1]
    assert pair["mask"][sun].min() > 0.99
    assert pair["target"][sun].mean() > 20 * base_white       # true range restored, far above SDR white
    assert pair["stats"]["k_hi"] is not None


def test_shadow_target_never_lifted_above_cap():
    pair = rc.build_pair(scene() * np.float32(NET), LOOK)
    corner = (slice(70, None), slice(0, 30))
    cap = rc.declared_inverse(rc.canonical(np.full((1, 1, 3), rc.LO_BAND[1], np.float32)))[0, 0] @ rc.LUMA2020
    assert (pair["target"][corner] @ rc.LUMA2020).max() <= cap * 1.001


@pytest.mark.parametrize('codec,quality', [('none', None), ('jpeg', 30)])
def test_source_preserving_target_passes_input_contract(codec, quality):
    look = {**LOOK, 'codec': codec, 'quality': quality}
    pair = rc.build_pair(scene() * np.float32(NET), look, target_mode='source_preserving')
    result = cc.roundtrip(pair['target'], pair['canon_in'])
    assert result['r_pass'], result
    assert np.array_equal(pair['mask'], cc.reconstruction_mask(pair['canon_in']))


def test_saved_dark_target_does_not_acquire_a_storage_black_lift(tmp_path):
    for name in ('sdr', 'hdr', 'mask'):
        (tmp_path / name).mkdir()
    (tmp_path / '_ingest_config.json').write_text(json.dumps({'storage': rc.storage().as_dict()}))
    source = np.broadcast_to(np.array([4, 4, 1], np.float32) / 255, (16, 16, 3)).copy()
    canon = rc.canonical(source)
    pair = {'canon_in': canon, 'target': rc.declared_inverse(canon),
            'mask': cc.reconstruction_mask(canon)}
    paths = rc.write_pair(tmp_path, 'dark', pair)['paths']
    from training.sdr2hdr_dataset import load_rgb
    result = cc.roundtrip(cc.load_reference(paths['hdr']), load_rgb(paths['sdr'], False))
    assert result['r_pass'], result


def test_baseline_passes_roundtrip_regrade_fails():
    pair = rc.build_pair(scene() * np.float32(NET), LOOK)
    base = rc.declared_inverse(pair["canon_in"])
    assert cc.roundtrip(base, pair["canon_in"], pair["mask"])["r_pass"]
    assert not cc.roundtrip(base * 1.25, pair["canon_in"], pair["mask"])["r_pass"]


def test_target_is_smoother_than_quantised_baseline_under_push():
    ramp = np.repeat(np.linspace(0.05, 0.2, 1200, dtype=np.float32)[None, :, None], 3, 2)
    ramp = np.repeat(ramp, 48, 0)
    pair = rc.build_pair(ramp * np.float32(NET), {**LOOK, "family": "clip"})
    base = rc.declared_inverse(pair["canon_in"])
    b_base = cc.banding_index(base, 2)["banding"]
    b_tgt = cc.banding_index(pair["target"], 2)["banding"]
    assert b_base is not None and b_tgt is not None and b_base > b_tgt + 0.3


def _sources(tmp_path, names=(("ph_a", "train"), ("ph_b", "val"))):
    from pipeline.hdr_io import HDRStorage, encode_hdr_u16
    d = tmp_path / "src" / "hdr"
    d.mkdir(parents=True)
    st = HDRStorage(mode="log2_extended")
    (tmp_path / "src" / "_ingest_config.json").write_text(json.dumps({"storage": st.as_dict()}))
    rows = []
    for i, (scene_id, split) in enumerate(names):
        code, _ = encode_hdr_u16(scene(seed=i), st)
        p = d / f"{scene_id}.png"
        cv2.imwrite(str(p), code[..., ::-1])
        rows.append({"source_id": "polyhaven", "scene_id": scene_id, "split": split, "asset_id": f"{scene_id}_f0",
                     "hdr_path": str(p.resolve()), "licence_licence": "CC0",
                     "plate_primaries": "rec2020"})
    m = tmp_path / "sources.jsonl"
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return m


def test_build_end_to_end_resumable_and_loadable(tmp_path):
    from training.sdr2hdr_dataset import load_rgb
    src = _sources(tmp_path)
    out = tmp_path / "corpus"
    s1 = rc.build(src, out, variants=2, codec_p=0.0)
    assert s1["rows"] == 4 and s1["splits"] == {"train": 2, "val": 2}
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    t = load_rgb(rows[0]["hdr_path"], hdr=True, ceiling=1e6)        # decodes through the sentinel
    assert t.shape == (96, 160, 3) and t.max() > 0.02
    s2 = rc.build(src, out, variants=2, codec_p=0.0)                  # resume: identical
    assert s2["manifest_sha256"] == s1["manifest_sha256"]
    with pytest.raises(ValueError):
        rc.build(src, out, variants=2, codec_p=0.0, view="aces")      # other settings, same dir
    with pytest.raises(ValueError, match="other settings"):
        rc.build(src, out, variants=2, codec_p=0.5)
    src.write_text(src.read_text() + "\n")
    with pytest.raises(ValueError, match="other settings"):
        rc.build(src, out, variants=2, codec_p=0.0)


def test_resume_rejects_damaged_completed_media(tmp_path):
    src = _sources(tmp_path)
    out = tmp_path / "corpus"
    rc.build(src, out, codec_p=0.0)
    next((out / "mask").glob("*.png")).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="corrupt completed row"):
        rc.build(src, out, codec_p=0.0)


def test_resume_pins_target_math_dependencies(tmp_path, monkeypatch):
    src = _sources(tmp_path)
    out = tmp_path / 'corpus'
    rc.build(src, out, codec_p=0)
    original = rc._sha
    monkeypatch.setattr(rc, '_sha', lambda p: 'changed' if p.name == 'contract_checks.py' else original(p))
    with pytest.raises(ValueError, match='other settings'):
        rc.build(src, out, codec_p=0)


def test_observed_clip_saved_targets_preserve_nearwhite_and_pin_policy(tmp_path):
    from training.sdr2hdr_dataset import load_rgb
    src = _sources(tmp_path)
    out = tmp_path / 'strict'
    rc.build(src, out, variants=3, codec_p=0, target_mode='source_preserving', reconstruction_policy='observed_clip_v1')
    for row in map(json.loads, (out / 'manifest.jsonl').read_text().splitlines()):
        assert row['reconstruction_policy'] == 'observed_clip_v1'
        x = load_rgb(row['sdr_path'], hdr=False)
        target = load_rgb(row['hdr_path'], hdr=True, ceiling=1e6)
        source = cc.canonical_to_source(x) * 255
        ordinary = (source.max(2) < 254.5) & (source.max(2) >= 5)
        result = cc.roundtrip(target, x, policy='observed_clip_v1')
        assert result['r_pass']
        viewed = cc.canonical_to_source(cc.view_forward(target)) * 255
        assert np.abs(viewed - source).max(2)[ordinary].max() < 1
    with pytest.raises(ValueError, match='other settings'):
        rc.build(src, out, variants=3, codec_p=0, target_mode='source_preserving')


def test_duplicate_assets_rejected_before_output(tmp_path):
    src = _sources(tmp_path)
    rows = [json.loads(line) for line in src.read_text().splitlines()]
    rows[1]["asset_id"] = rows[0]["asset_id"]
    src.write_text("\n".join(json.dumps(row) for row in rows))
    out = tmp_path / "corpus"
    with pytest.raises(ValueError, match="duplicate output asset"):
        rc.build(src, out)
    assert not out.exists()


def test_heldout_family_never_in_train_and_eval_scenes_refused(tmp_path):
    src = _sources(tmp_path)
    with pytest.raises(ValueError):
        rc.build(src, tmp_path / "x", families=["filmic_s"])
    rc.build(src, tmp_path / "y", families=["filmic_s"], splits=["val"], codec_p=0.0)
    bad = _sources(tmp_path / "b", names=(("mataha_A001", "train"), ("ph_c", "val")))
    with pytest.raises(ValueError):
        rc.build(bad, tmp_path / "z")


def test_vae_hook_roundtrip_shape():
    import torch
    from pipeline.generator_vae import make_roundtrip

    class Dist:
        def __init__(self, z):
            self.z = z

        def mode(self):
            return self.z

    class FakeVAE:
        def encode(self, x):
            return type("E", (), {"latent_dist": Dist(x * 0.9)})()

        def decode(self, z):
            return type("D", (), {"sample": z})()

    rt = make_roundtrip(FakeVAE(), torch.device("cpu"))
    img = np.random.default_rng(0).integers(0, 255, (37, 50, 3)).astype(np.uint8)
    out = rt(img)
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)


def test_enforce_contract_makes_any_output_roundtrip_but_keeps_reconstruction():
    pair = rc.build_pair(scene() * np.float32(NET), LOOK)
    regraded = rc.declared_inverse(pair["canon_in"]) * 1.3
    regraded[pair["mask"] > 0.99] = 3.0                        # a model's highlight guess
    fixed = cc.enforce_contract(regraded, pair["canon_in"])
    assert cc.roundtrip(fixed, pair["canon_in"])["r_pass"]
    sel = (cc.reconstruction_mask(pair["canon_in"]) > 0.99) & (pair["mask"] > 0.99) & (pair["canon_in"].max(axis=2) > 0.5)
    assert sel.any() and np.allclose(fixed[sel], 3.0)


def test_sources_drop_test_split_and_noncommercial():
    from pipeline.run05_sources import collect
    rows = [{"scene_id": "a", "asset_id": "a1", "split": "train", "hdr_path": "/x/a.png", "commercial_ok": True},
            {"scene_id": "m", "asset_id": "m1", "split": "test", "hdr_path": "/x/m.png", "commercial_ok": True},
            {"scene_id": "n", "asset_id": "n1", "split": "train", "hdr_path": "/x/n.png", "commercial_ok": False}]
    out, dropped = collect(rows, "s")
    assert [r["asset_id"] for r in out] == ["a1"] and dropped["test"] == 1 and dropped["not_commercial"] == 1


@pytest.mark.parametrize('permission', [None, False, 'true', 1])
def test_sources_require_explicit_commercial_permission(permission):
    from pipeline.run05_sources import collect
    row = dict(scene_id='s', asset_id='a', split='train', hdr_path='/x/a.png', commercial_ok=permission)
    out, dropped = collect([row], 's')
    assert not out and dropped['not_commercial'] == 1


def test_sources_must_state_units_and_primaries(tmp_path):
    with pytest.raises(ValueError):
        rc.load_plate({"asset_id": "e", "hdr_path": str(tmp_path / "x.exr"), "plate_primaries": "rec709"})
    with pytest.raises(ValueError):
        rc.load_plate({"asset_id": "p", "hdr_path": str(tmp_path / "x.png")})


def test_collect_rudra_split_authority_and_panorama_keys(tmp_path):
    from pipeline.run05_sources import collect_rudra, panorama_key
    assert panorama_key("tif:aarfontein_dusk_4k") == panorama_key("hl_ph_aarfontein_dusk") == "ph:aarfontein_dusk"
    r02 = [{"scene_id": "fxtd_e_a", "split": "val", "source_id": "elmatareed"},
           {"scene_id": "hl_ph_aarfontein_dusk", "split": "train", "source_id": "polyhaven_highlight_pipeline",
            "source_path": "G:\\x\\linear_exr\\hl.exr", "asset_id": "h"},
           {"scene_id": "fxtd_lab_1", "split": "train", "source_id": "lab", "source_path": "F:\\RED\\cam.exr"},
           {"scene_id": "fxtd_mataha_x", "split": "test", "source_id": "mataha"}]
    a = [{"scene_id": "fxtd_e_a", "split": "train", "source_id": "elmatareed", "asset_id": "e1", "hdr_path": "G:\\a.png"},
         {"scene_id": "fxtd_mataha_x", "split": "train", "source_id": "mataha", "asset_id": "m1", "hdr_path": "G:\\m.png"}]
    mv = [{"scene_id": "tif:aarfontein_dusk_4k", "hdr_frames": ["D:\\v\\f0.png", "D:\\v\\f8.png"]}]
    for name, rows in (("r02", r02), ("a", a), ("mv", mv)):
        for row in rows:
            row['commercial_ok'] = True
        if name == 'a':
            rows.append(dict(scene_id='unknown_rights', split='train', source_id='camera',
                             asset_id='unreviewed', hdr_path='G:/unknown.png'))
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out, notes = collect_rudra(tmp_path / "r02.jsonl", tmp_path / "a.jsonl", tmp_path / "mv.jsonl",
                               root_map={"D:\\v": "G:\\v"})
    by = {r["asset_id"]: r for r in out}
    assert by["e1"]["split"] == "val"                    # run02 split wins over Corpus A's
    assert "m1" not in by and notes["dropped_forbidden"] == 1
    assert by["f0"]["split"] == "train" and by["f0"]["hdr_path"].startswith("G:\\v")
    assert by["hl"]["plate_primaries"] == "rec709" and by["hl"]["exr_scale"] > 0
    assert "cam" not in by                               # camera EXRs are not Rec.709 renders
    assert 'unreviewed' not in by and notes['dropped_noncommercial'] == 1
