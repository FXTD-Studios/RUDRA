"""tools/check_package_weights.py: a public package carries no model weights (1.0.3)."""
import zipfile

from tools.check_package_weights import main, offending

# What the published 1.0.1 Windows ZIP held (listed from the release asset, 1 Nov 2026).
SHIPPED_101 = [
    "RUDRA-1.0.1-windows-x64/RUDRA.exe",
    "RUDRA-1.0.1-windows-x64/onnxruntime.dll",
    "RUDRA-1.0.1-windows-x64/LICENSE",
    "RUDRA-1.0.1-windows-x64/LICENSE-weights",
    "RUDRA-1.0.1-windows-x64/models/models.json",
    "RUDRA-1.0.1-windows-x64/models/sdr2hdr_image_v8/manifest.json",
    "RUDRA-1.0.1-windows-x64/models/sdr2hdr_image_v8/model.tile.onnx",
    "RUDRA-1.0.1-windows-x64/models/sdr2hdr_image_v8/model.ts",
    "RUDRA-1.0.1-windows-x64/models/sdr2hdr_image_v8/golden/flat_grey.sdr.npy",
]

CLEAN = [
    "RUDRA-1.0.3-windows-x64/RUDRA.exe",
    "RUDRA-1.0.3-windows-x64/rudra-native.exe",
    "RUDRA-1.0.3-windows-x64/onnxruntime.dll",
    "RUDRA-1.0.3-windows-x64/DirectML.dll",
    "RUDRA-1.0.3-windows-x64/OpenColorIO_2_5.dll",
    "RUDRA-1.0.3-windows-x64/LICENSE",
    "RUDRA-1.0.3-windows-x64/NOTICE",
    "RUDRA-1.0.3-windows-x64/models/",
    "RUDRA-1.0.3-windows-x64/ffmpeg/ffmpeg.exe",
]


def test_the_101_package_is_refused():
    bad = offending(SHIPPED_101)
    assert any("sdr2hdr_image_v8/model.tile.onnx" in b for b in bad)
    assert any("sdr2hdr_image_v8/model.ts" in b for b in bad)
    assert any(b.endswith("LICENSE-weights") for b in bad)
    assert not any(b.endswith("RUDRA.exe") or b.endswith("onnxruntime.dll") for b in bad)


def test_a_model_free_package_passes():
    assert offending(CLEAN) == []


def test_any_beta2_name_is_refused_whatever_its_extension():
    assert offending(["RUDRA.app/Contents/Resources/sdr2hdr_shadow_v1.config.json"])
    assert offending(["pkg/extra/sdr2hdr_temporal_v1.pt"])


def test_weights_outside_models_are_refused():
    assert offending(["pkg/somewhere/rudra_genfill.safetensors"])
    assert offending(["pkg/RUDRA_ACES_2/rudra_g1.onnx"])


def test_cli_on_a_zip(tmp_path):
    good, bad = tmp_path / "good.zip", tmp_path / "bad.zip"
    with zipfile.ZipFile(good, "w") as z:
        for n in CLEAN:
            if not n.endswith("/"):
                z.writestr(n, b"x")
    with zipfile.ZipFile(bad, "w") as z:
        for n in SHIPPED_101:
            z.writestr(n, b"x")
    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1


def test_cli_on_a_folder(tmp_path):
    app = tmp_path / "RUDRA.app" / "Contents" / "Resources" / "models"
    app.mkdir(parents=True)
    assert main([str(tmp_path / "RUDRA.app")]) == 0
    (app / "sdr2hdr_image_v8").mkdir()
    (app / "sdr2hdr_image_v8" / "model.ts").write_bytes(b"x")
    assert main([str(tmp_path / "RUDRA.app")]) == 1
