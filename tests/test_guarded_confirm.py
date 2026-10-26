import json

import numpy as np
import pytest

from training import guarded_confirm as gc
from training import sihdr_eval as se
from tests.test_sihdr_eval import FakeRunner, make_dataset


def selection(tmp_path, ckpt, selected="lo0.05_hi0.3_f0"):
    s = tmp_path / "sel.json"
    s.write_text(json.dumps({"selected": selected, "checkpoint_sha256": gc.sha(ckpt),
                             "grid": [{"lo": 0.05, "hi": 0.3, "feather_px": 0}]}))
    return s


def test_register_freezes_and_refuses_failed_selection(tmp_path):
    ckpt = tmp_path / "c.pt"; ckpt.write_bytes(b"c")
    p = gc.register(selection(tmp_path, ckpt), tmp_path / "pre.json")
    assert p["guard"] == {"lo": 0.05, "hi": 0.3, "feather_px": 0} and set(p["criteria"]) == {"C1", "C2", "C3", "C4"}
    with pytest.raises(ValueError, match="never rewritten"):
        gc.register(selection(tmp_path, ckpt), tmp_path / "pre.json")
    with pytest.raises(ValueError, match="failed on validation"):
        gc.register(selection(tmp_path, ckpt, selected=None), tmp_path / "pre2.json")


def test_sihdr_run_keeps_unguarded_and_scores_guarded(tmp_path):
    ids = ["001", "002", "003", "004", "005", "006"]
    root = make_dataset(tmp_path, ids)
    proto = tmp_path / "protocol.json"
    se.freeze(tmp_path / "prep.json", tmp_path / "anchors.json", proto)
    ckpt = tmp_path / "c.pt"; ckpt.write_bytes(b"c")
    pre = tmp_path / "pre.json"
    gc.register(selection(tmp_path, ckpt), pre)
    summ = gc.run_sihdr(pre, proto, root, ckpt, tmp_path / "out", runner=FakeRunner(20.0))
    row = json.loads((tmp_path / "out" / "per_image.jsonl").read_text().splitlines()[0])
    assert set(row["methods"]) == {"model", "unguarded", "baseline_bt1886"}
    assert summ["release_authorized"] is False
    other = tmp_path / "other.pt"; other.write_bytes(b"x")
    with pytest.raises(ValueError, match="checkpoint differs"):
        gc.run_sihdr(pre, proto, root, other, tmp_path / "out2", runner=FakeRunner(1.0))


def test_guarded_runner_only_changes_dark_pixels():
    class Inner:
        def predict(self, code):
            base = np.full((4, 4, 3), 1e-6); base[:, 2:] = 0.05      # network units: 0.01 nit vs 500 nits
            return {"model": base * 4, "baseline_bt1886": base}
    out = gc.GuardedRunner(Inner(), {"lo": 0.05, "hi": 0.3, "feather_px": 0}).predict(None)
    assert np.allclose(out["model"][:, :2], 1e-6) and np.allclose(out["model"][:, 2:], 0.2)
    assert np.allclose(out["unguarded"][:, :2], 4e-6)
