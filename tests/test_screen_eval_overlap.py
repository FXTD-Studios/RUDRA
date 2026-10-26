import numpy as np

from pipeline.screen_eval_overlap import screen, thumb, variants


def test_duplicates_and_mirrors_flagged_distinct_not():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.01, 50, (64, 96, 3))
    b = rng.uniform(0.01, 50, (64, 96, 3))
    r = screen({"eval_a": variants(a), "eval_b": variants(b)},
               {"train_copy": [thumb(a[:, ::-1] * 3.0)], "train_other": [thumb(rng.uniform(0.01, 50, (64, 96, 3)))]})
    assert [f["eval"] for f in r["flags"]] == ["eval_a"]
    assert r["best_match"]["eval_a"]["train"] == "train_copy"
