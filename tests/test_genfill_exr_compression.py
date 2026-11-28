"""1.3.2 (28 Nov 2026): an AI-filled frame keeps its master's EXR compression, whichever OpenEXR one Deliver chose."""
import numpy as np
import pytest

OpenEXR = pytest.importorskip("OpenEXR")

from rudra.genfill import run


@pytest.mark.parametrize("name", ["NO_COMPRESSION", "ZIP_COMPRESSION", "ZIPS_COMPRESSION", "PIZ_COMPRESSION",
                                  "DWAA_COMPRESSION", "B44_COMPRESSION"])
def test_the_filled_frame_keeps_its_masters_compression(tmp_path, name):
    comp = getattr(OpenEXR, name)
    h, w = 8, 12
    rgb = {c: np.full((h, w), v, np.float16) for c, v in zip("RGB", (0.5, 0.25, 2.0))}
    master = tmp_path / "f.000000.exr"
    OpenEXR.File({"compression": comp, "type": OpenEXR.scanlineimage}, rgb).write(str(master))
    fr = run.read_frame(master)
    alpha = np.zeros((h, w), np.float32)
    alpha[2:4, 3:6] = 1.0
    out = tmp_path / "out.exr"
    run.write_frame(out, fr, np.ones((h, w, 3), np.float32), alpha, {"rudra:test": "1"})
    with OpenEXR.File(str(out)) as f:
        assert f.header()["compression"] == comp
        assert "rudra.generated" in f.channels()
