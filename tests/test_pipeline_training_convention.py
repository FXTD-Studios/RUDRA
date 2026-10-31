import pytest
from training.sdr2hdr_dataset import corpus_ev_of

PIPE = dict(sdr_kind="rendered", render_family="pipeline", tonemap_ev=None,
            pipeline_recipe={"family": "aces2.1_1000"})

def test_pipeline_requires_explicit_convention_and_preserves_metadata():
    with pytest.raises(ValueError, match="explicit"):
        corpus_ev_of("unused", [PIPE])
    assert corpus_ev_of("unused", [PIPE], pipeline_only_ev=0) == 0
    assert PIPE["tonemap_ev"] is None
    assert PIPE["sdr_kind"] == "rendered"

@pytest.mark.parametrize("rows,ev,real", [
    ([PIPE, dict(sdr_kind="real")], 0, None),
    ([PIPE, dict(sdr_kind="rendered",tonemap_ev=0)], 0, None),
    ([PIPE], float("nan"), None),
    ([PIPE], 0, 0),
    ([], 0, None),
    ([dict(PIPE,tonemap_ev=1)], 0, None),
])
def test_pipeline_override_rejects_mixed_or_invalid_conventions(rows,ev,real):
    with pytest.raises(ValueError, match="pipeline_only_ev"):
        corpus_ev_of("unused", rows, real_only_ev=real, pipeline_only_ev=ev)

def test_existing_real_and_scene_linear_conventions_unchanged():
    assert corpus_ev_of("unused", [dict(sdr_kind="real")], real_only_ev=0) == 0
    assert corpus_ev_of("unused", [dict(sdr_kind="rendered",tonemap_ev=0)]) == 0
