"""tools/holdout_selection.py and tools/holdout_verdict.py (7 Nov 2026): the Crew holdout's mapping
and its registered rule, on made-up records (nothing of the holdout is read)."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from holdout_kit import seal  # noqa: E402
from holdout_selection import main as selection_main, selection  # noqa: E402
from holdout_verdict import verdict  # noqa: E402


def sealed(rows):
    """A holdout record with the given rows and a valid seal."""
    rec = {"schema": "rudra_holdout_registration_v1", "rows": rows}
    rec["seal_sha256"] = seal(rec)
    return rec


ROWS = [{"id": "a_f1", "scene_id": "a", "path": "G:/h/hdr/a_f1.png", "project": "P", "camera": "C", "sha256": "1" * 64},
        {"id": "b_f2", "scene_id": "b", "path": "G:/h/hdr/b_f2.png", "project": "P", "camera": "C", "sha256": "2" * 64}]


def test_every_row_in_order_with_the_builder_fields():
    """Each registered row becomes one selection row, in order, with what the builder reads."""
    out = selection(sealed(ROWS), "rec2020")
    assert [s["asset_id"] for s in out] == ["a_f1", "b_f2"]
    assert out[0] == {"asset_id": "a_f1", "scene_id": "a", "project": "P", "camera": "C", "hdr_path": "G:/h/hdr/a_f1.png",
                      "hdr_sha256": "1" * 64, "plate_primaries": "rec2020"}


def test_an_edited_registration_or_bad_primaries_is_refused(tmp_path):
    """A record changed after sealing, unknown primaries, and an existing output all stop it."""
    rec = sealed(ROWS)
    rec["rows"] = rec["rows"][:1]
    with pytest.raises(SystemExit, match="edited"):
        selection(rec, "rec2020")
    with pytest.raises(SystemExit):
        selection(sealed(ROWS), "p3")
    reg = tmp_path / "h.json"
    reg.write_text(json.dumps(sealed(ROWS)))
    out = tmp_path / "sel.json"
    assert selection_main([str(reg), "--plate-primaries", "rec2020", "--out", str(out)]) == 0
    with pytest.raises(SystemExit, match="exists"):
        selection_main([str(reg), "--plate-primaries", "rec2020", "--out", str(out)])


def result(lock0, lock_m3=480.0, lock_p3=0.30, h=0.07, negy=0, rde=0):
    """An evaluate_route_arms result with analytic 0 EV RMS 0.24 code and the given lock numbers."""
    an = {"grade_-3": 500.0, "grade_0": 0.24 ** 2, "grade_3": 0.35, "H": 0.08}
    lk = {"grade_-3": lock_m3, "grade_0": lock0 ** 2, "grade_3": lock_p3, "H": h, "new_negative_y": negy,
          "remaining_display_errors": rde}
    return {"primary": "w4:lock_dqs050g", "cases": 90, "scenes": 24, "summary": {"analytic": an, "w4:lock_dqs050g": lk}}


def test_the_tolerant_0ev_rule_is_relative_to_this_sets_analytic():
    """0 EV passes at analytic RMS + 0.1 code and fails just past it; the others are no worse than analytic."""
    assert verdict(result(0.339))["passes"]
    v = verdict(result(0.3405))
    assert not v["passes"] and not v["gates"]["grade_0_rms"]["pass"]
    assert math.isclose(v["gates"]["grade_0_rms"]["limit"], 0.34)
    assert not verdict(result(0.30, lock_m3=501.0))["passes"]
    assert not verdict(result(0.30, lock_p3=0.36))["passes"]
    assert not verdict(result(0.30, h=0.081))["passes"]
    assert not verdict(result(0.30, negy=1))["passes"]
    assert not verdict(result(0.30, rde=1))["passes"]
