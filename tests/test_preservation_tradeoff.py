import pytest
import torch
import json

from rudra.sdr2hdr import SDR2HDRNet
from tools.diagnose_preservation_tradeoff import error_partition, gradient_relation, parameter_probe
from tools.diagnose_preservation import verify_historical


def test_error_partitions_conserve_full_frame_error_and_coverage():
    ref = torch.tensor([.00001, .0005, .005, .05, .5]).reshape(1, 1, 1, 5).expand(1, 3, 1, 5)
    sdr = torch.tensor([0., .2, .5, .8, 1.]).reshape(1, 1, 1, 5).expand_as(ref)
    result = error_partition(ref * 1.2, ref * .8, ref, sdr)
    for group in ("luminance_bands", "sdr_regions"):
        assert sum(x["pixel_fraction"] for x in result[group].values()) == pytest.approx(1)
        for kind in ("model", "baseline"):
            assert sum(x[f"{kind}_mse"] for x in result[group].values()) == pytest.approx(result[f"full_{kind}_mse"], rel=1e-6)
    assert all(x["pixel_fraction"] == pytest.approx(.2) for x in result["luminance_bands"].values())


def test_gradient_relation_handles_opposition_and_absent_gradient():
    a = torch.tensor([3., 4.])
    assert gradient_relation(a, -2*a) == {"cosine": -1., "norm_ratio": .5}
    assert gradient_relation(a, torch.zeros_like(a)) == {"cosine": None, "norm_ratio": None}


@pytest.mark.parametrize("amp", [False, True])
def test_parameter_probe_preserves_weights_flags_and_does_not_accumulate_gradients(amp):
    torch.manual_seed(23)
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0).eval().requires_grad_(False)
    with torch.no_grad():
        model.head.bias[:3].fill_(-.05)
    before = {k: v.clone() for k,v in model.state_dict().items()}
    x = torch.full((1, 3, 32, 32), 1/255)
    result = parameter_probe(model, x, torch.full_like(x, .0001), 1, amp=amp)
    assert result["terms"]["post_hold"]["gradient_norm"] == 0
    assert result["terms"]["pre_hold"]["gradient_norm"] > 0
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("corrupt", [False, True])
def test_new_pair_historical_parity_rejects_changed_scores(tmp_path, corrupt):
    rows = []
    for arm in ("control", "preclamp"):
        directory = tmp_path / arm
        directory.mkdir()
        frame = {"asset_id": "frame", "condition": "real", "ev": None,
                 "pu_gain_db": -2., "clip_gain_stops": None, "crushed_gain_stops": .5}
        (directory / "bench_val.jsonl").write_text(json.dumps({"step": 200, "frames": [frame]}) + "\n")
        measured = {k: frame[k] for k in ("pu_gain_db", "clip_gain_stops", "crushed_gain_stops")}
        if corrupt and arm == "preclamp":
            measured["pu_gain_db"] += .01
        rows.append({"arm": arm, **frame, "variants": {"original": measured}})
    if corrupt:
        with pytest.raises(ValueError, match="historical score mismatch"):
            verify_historical(rows, tmp_path, ("control", "preclamp"))
    else:
        assert verify_historical(rows, tmp_path, ("control", "preclamp")) == 0
