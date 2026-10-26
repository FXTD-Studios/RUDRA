"""Missing gradient coverage must not masquerade as zero banding error."""
import numpy as np

from training.contract_checks import headroom


def test_flat_image_has_no_banding_evidence():
    flat = np.full((64, 64, 3), 0.005, dtype=np.float32)
    result = headroom(flat, flat)
    assert result["banding_measured_pushes"] == 0
    assert result["banding_excess_worst"] is None


def test_gradient_has_measured_banding_evidence():
    ramp = np.linspace(0.001, 0.015, 1200, dtype=np.float32)
    image = np.broadcast_to(ramp[None, :, None], (48, 1200, 3)).copy()
    result = headroom(image, image)
    assert result["banding_measured_pushes"] > 0
    assert result["banding_excess_worst"] == 0.0


def test_reference_not_clipped_to_model_ceiling(tmp_path):
    import cv2
    import json
    from pipeline.hdr_io import HDRStorage, encode_hdr_u16
    from training.contract_checks import load_reference
    storage = HDRStorage(mode='log2_extended')
    (tmp_path / '_ingest_config.json').write_text(json.dumps({'storage': storage.as_dict()}))
    # 3000 scene units * 203 nits / 10000 = 60.9 network units, above model max 4.
    code, _ = encode_hdr_u16(np.full((8, 8, 3), 3000, np.float32), storage)
    path = tmp_path / 'reference.png'
    assert cv2.imwrite(str(path), code)
    assert load_reference(path).min() > 60
