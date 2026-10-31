import pytest
from pipeline.generator_vae import wan_vae_identity


def test_identity_changes_with_weights_and_config(tmp_path):
    pytest.importorskip('diffusers')   # the identity records the diffusers version
    root = tmp_path / 'wan'
    root.mkdir()
    weights = root / 'model.safetensors'
    config = root / 'config.json'
    weights.write_bytes(b'first')
    config.write_text('{}')
    first = wan_vae_identity(str(root))
    weights.write_bytes(b'second')
    second = wan_vae_identity(str(root))
    assert first != second
    config.write_text('{"changed":true}')
    assert second != wan_vae_identity(str(root))


def test_identity_requires_weights_and_refuses_ltx(tmp_path):
    with pytest.raises(ValueError, match='safetensors'):
        wan_vae_identity(str(tmp_path))
    p = tmp_path / 'ltx.safetensors'
    p.write_bytes(b'x')
    with pytest.raises(ValueError, match='LTX'):
        wan_vae_identity(str(p))
