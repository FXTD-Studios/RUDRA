import json

import numpy as np
import pytest

from rudra.dark_guard import apply_dark_guard, guard_weight
from training import dark_guard_eval as dge


def test_guard_keeps_baseline_in_blacks_and_model_in_highlights():
    base = np.full((8, 8, 3), 0.01)
    base[:, 4:] = 500.0
    model = base * 3.0
    out = apply_dark_guard(model, base, lo=0.05, hi=0.3)
    assert np.allclose(out[:, :4], base[:, :4]) and np.allclose(out[:, 4:], model[:, 4:])
    w = guard_weight(base, 0.05, 0.3, feather_px=2)
    assert 0 < w[0, 3] < 1 and w.min() >= 0 and w.max() <= 1
    with pytest.raises(ValueError):
        guard_weight(base, 0.3, 0.05)


def make_run(tmp_path):
    from rudra.delivery.bench import load_frame, pu_psnr
    from rudra.delivery.exr import write_exr
    run = tmp_path / "run"
    rng = np.random.default_rng(1)
    for cond in ("clean", "hard"):
        res = {"test": [], "baseline_bt1886": []}
        for s in range(4):
            for f in range(2):
                ref = rng.uniform(0.5, 300, (32, 48, 3)) / 203.0
                ref[:12] = 0.0002                          # dark region (< 0.1 nits)
                base = ref * 1.02
                model = ref * 1.0
                model[:12] = 0.05                          # lifted blacks: the measured defect
                for tree, img in (("ref", ref), ("test", model), ("baseline_bt1886", base)):
                    p = run / cond / tree / f"s{s}" / f"f{f}.exr"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    write_exr(p, img.astype(np.float32), half=True)
                for tree in ("test", "baseline_bt1886"):
                    a = load_frame(run / cond / tree / f"s{s}" / f"f{f}.exr", 203.0)
                    r = load_frame(run / cond / "ref" / f"s{s}" / f"f{f}.exr", 203.0)
                    res[tree].append(dict(clip=f"s{s}", frame=f"f{f}", pu_psnr_db=pu_psnr(a, r)))
        for tree, rows in res.items():
            (run / cond / f"{tree}.json").write_text(json.dumps({"results": rows}))
    return run


def test_guard_fixes_lifted_blacks_and_rule_selects(tmp_path):
    run = make_run(tmp_path)
    rows = {c: dge.score_condition(run / c, dge.GRID) for c in ("clean", "hard")}
    summary = dge.summarize(rows, dge.GRID)
    un = summary["settings"]["unguarded"]
    assert un["clean"]["dark_frames"] == 8 and un["clean"]["dark_scenes"] == 4
    assert un["clean"]["dark_mean_gain_db"] < -1
    best = summary["settings"][summary["selected"]]
    assert best["clean"]["dark_mean_gain_db"] > un["clean"]["dark_mean_gain_db"] + 1
    assert summary["release_authorized"] is False


def test_recorded_scores_must_reproduce(tmp_path):
    run = make_run(tmp_path)
    data = json.loads((run / "clean" / "test.json").read_text())
    data["results"][0]["pu_psnr_db"] += 0.5
    (run / "clean" / "test.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not the same frames"):
        dge.score_condition(run / "clean", dge.GRID[:1])
