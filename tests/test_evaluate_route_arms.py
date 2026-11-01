"""tools/evaluate_route_arms.py end to end: a stub model, the real ACES 2.0 view.

Runs without PyTorch: a minimal stand-in covers the few torch calls the
evaluator makes (from_numpy, permute, indexing, load, inference_mode), so the
evaluator's own code, the routing, the colour constraint and OCIO all run.
"""
import contextlib
import hashlib
import importlib
import json
import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyOpenColorIO")
from rudra.aces_view import ACES2View


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def permute(self, *axes):
        return _T(self.a.transpose(axes))

    def __getitem__(self, i):
        return _T(self.a[i])

    def numpy(self):
        return self.a


def _fake_torch():
    t = ModuleType("torch")
    t.from_numpy = lambda a: _T(a)
    t.set_num_threads = lambda n: None
    t.inference_mode = contextlib.nullcontext
    t.load = lambda p, map_location=None, weights_only=True: pickle.loads(open(p, "rb").read())
    return t


def _view():
    v = ACES2View(threads=4)
    v.set_threads = lambda n: None
    return v


@pytest.fixture
def ev(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    ref_mod = ModuleType("training.aces_reference")
    ref_mod.reference_view = _view
    monkeypatch.setitem(sys.modules, "training.aces_reference", ref_mod)
    model_mod = ModuleType("training.aces_model")
    model_mod.model_from_checkpoint = lambda st: None
    monkeypatch.setitem(sys.modules, "training.aces_model", model_mod)
    monkeypatch.delitem(sys.modules, "tools.evaluate_route_arms", raising=False)
    return importlib.import_module("tools.evaluate_route_arms")


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class StubModel:
    """The analytic inverse as the baseline; a 'reconstruction' that is right in
    fully clipped pixels and wrong (too bright) in partially clipped ones."""

    def __init__(self, view, refs):
        self.view, self.refs = view, refs

    def eval(self):
        return self

    def __call__(self, x):
        src = x[0].permute(1, 2, 0).numpy()
        base = np.asarray(self.view.inverse(src), np.float32)
        ref = self.refs[src.tobytes()]
        full = src.min(-1) >= 254.5 / 255
        part = (src.max(-1) >= 254.5 / 255) & ~full
        pred = base.copy()
        pred[full] = ref[full]
        pred[part] = base[part] * 1.6
        w = (src.max(-1) > 0.5).astype(np.float32)
        t = lambda a: _T(np.ascontiguousarray(a).transpose(2, 0, 1)[None])
        return SimpleNamespace(acescg=t(pred), baseline=t(base), reconstruction_weight=_T(w[None, None]))


def test_full_only_arm_end_to_end(tmp_path, monkeypatch, ev):
    view = _view()
    rng = np.random.default_rng(3)
    refs, rows = {}, []
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    grade_reg = tmp_path / "grade_reg.json"
    grade_reg.write_text(json.dumps({"limits": "test"}))
    for i in range(3):
        ref = rng.uniform(0.05, 0.6, (24, 24, 3)).astype(np.float32)
        ref[6:18, 6:18] = 8.0 + rng.uniform(0, 4, (12, 12, 1))     # a bright, fully clipped area
        ref[1:4, 1:4] = [0.3, 0.3, 6.0]                             # blue-only clip: partial
        src = np.rint(np.clip(np.asarray(view.forward(ref), np.float32), 0, 1) * 255) / 255
        src = src.astype(np.float32)
        refs[src.tobytes()] = ref
        sp, rp = corpus / f"s{i}.npy", corpus / f"r{i}.npy"
        np.save(sp, src)
        np.save(rp, ref)
        rows.append({"case_id": f"c{i}", "scene_id": f"sc{i}", "project": "p", "clip_target": 0, "kept": True,
                     "source_path": str(sp), "source_sha256": _sha(sp), "reference_path": str(rp), "reference_sha256": _sha(rp)})
    (corpus / "report.json").write_text(json.dumps({"registration_sha256": _sha(grade_reg), "rows": rows}))
    reg = tmp_path / "arms.json"
    reg.write_text(json.dumps({"grade_set_registration_sha256": _sha(grade_reg), "limits": "test",
                               "arms": {"full_only_feathers_px": [0, 8], "primary": "full_only_f0"}}))
    ckpt = tmp_path / "stub.pt"
    ckpt.write_bytes(pickle.dumps({"stub": 0}))
    monkeypatch.setattr(ev, "model_from_checkpoint", lambda st: StubModel(view, refs))
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["x", "--corpus", str(corpus), "--registration", str(reg),
                                      "--candidate", f"stub={ckpt}", "--out", str(out)])
    ev.main()
    r = json.loads(out.read_text())
    assert r["primary"] == "stub:full_only_f0"
    s = r["summary"]
    # The stub's partial error is gone in the routed arms and present in the shipped one.
    assert s["stub:full_only_f0"]["grade_0"] <= s["analytic"]["grade_0"] + 1e-9
    assert s["stub:learned"]["grade_0"] > s["stub:full_only_f0"]["grade_0"]
    assert r["gates"]["grade_0"] and r["gates"]["highlight_H"]
    for row in r["rows"]:
        assert row["methods"]["stub:full_only_f0"]["H_partial"] == pytest.approx(row["methods"]["analytic"]["H_partial"])
