"""The 0 EV training kit: the bin hinge and its decision table."""
import json

import pytest

from tools.summarize_bin_hinge import main as summarize


def test_bin_hinge_penalises_only_beyond_the_tolerance():
    torch = pytest.importorskip("torch")
    from training.aces_bin_hinge import bin_hinge
    src = torch.full((1, 3, 2, 2), 0.8)
    view0 = src.clone()
    view0[0, 0, 0, 0] += 0.3 / 255     # inside half a code: free
    view0[0, 0, 0, 1] += 1.5 / 255     # a code beyond it
    view0[0, 1, 1, 0] -= 0.75 / 255    # a quarter code beyond it
    mask = torch.ones(1, 1, 2, 2)
    v, over_tol, over_one = bin_hinge(view0, src, mask, 0.5)
    assert v.item() == pytest.approx((1.0 ** 2 + 0.25 ** 2) / 4, rel=1e-3)
    assert over_tol == pytest.approx(0.5) and over_one == pytest.approx(0.25)
    mask[0, 0, 0, 1] = 0               # masked out: not counted
    v2, _, _ = bin_hinge(view0, src, mask, 0.5)
    assert v2.item() == pytest.approx(0.25 ** 2 / 3, rel=1e-3)
    view0.requires_grad_(True)
    bin_hinge(view0, src, torch.ones(1, 1, 2, 2), 0.5)[0].backward()
    assert view0.grad[0, 0, 0, 0] == 0 and view0.grad[0, 0, 0, 1] > 0


def _arm(tmp_path, name, aligned, grade0, fresh0):
    out = tmp_path / name
    (out / "paired_development").mkdir(parents=True)
    hist = [{"step": i, "over_one_code_share": 0.3 - i * 1e-4, "over_tol_share": 0.6 - i * 1e-4} for i in range(300)]
    (out / "report.json").write_text(json.dumps({"history": hist}))
    (out / "paired_development" / "acceptance.json").write_text(json.dumps(
        {"new_aligned_residual_mse": {"candidate": aligned},
         "checks": {"new_H_no_worse_analytic": True, "outside_full_exact": False, "aligned_residual_preserved": aligned <= .010367}}))
    (out / "grade.json").write_text(json.dumps({"summary": {"grades": {
        "-3": {"constrained": 11.0, "analytic": 18.6}, "0": {"constrained": grade0, "analytic": .0604},
        "3": {"constrained": .002, "analytic": .0025}}}}))
    (out / "video.json").write_text(json.dumps({"checks": {"temporal": True}, "max_transition_mean_ev": 0.015}))
    (out / "fresh.json").write_text(json.dumps({"summary": {
        "analytic": {"grade_-3": 526.5, "grade_0": 0.0568, "grade_3": 0.3927, "H": 0.0832},
        f"{name}:learned": {"grade_-3": 490.0, "grade_0": fresh0, "grade_3": 0.35, "H": 0.078}}}))
    return out


def test_decision_table_reads_every_file_and_judges_each_gate(tmp_path, capsys):
    exp = tmp_path / "exp"
    exp.mkdir()
    for name, aligned, g0, f0 in (("w4", .0095, .055, .10), ("w16", .0120, .050, .09)):
        out = _arm(tmp_path, name, aligned, g0, f0)
        (exp / f"claude_bin_hinge_{name}_20261102.json").write_text(json.dumps(
            {"out": str(out), "bin_weight": 4 if name == "w4" else 16, "bin_tol_codes": 0.5}))
    summarize(["--arms", "w4", "w16", "--experiments", str(exp)])
    text = capsys.readouterr().out
    assert "VERDICT w4: passes" in text
    assert "VERDICT w16: does not pass" in text
    assert "aligned residual        0.012000" in text and "FAIL" in text
    assert "pixels over 1 code 0.295 -> 0.275" in text
