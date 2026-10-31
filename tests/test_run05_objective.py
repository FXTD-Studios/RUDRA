import pytest
import torch
from training.run05_objective import restoration_total
from training.resumable_training import parse, experiment_of


def test_correct_restoration_is_not_pulled_back_to_zero():
    correction = torch.tensor(0.2, requires_grad=True)
    error = (correction - 0.2).square()
    prior = correction.abs()
    total = restoration_total({'total': error + 0.1 * prior, 'residual_outside': prior})
    total.backward()
    assert correction.grad.item() == pytest.approx(0.0)


def test_wrong_restoration_still_receives_target_gradient():
    correction = torch.tensor(0.1, requires_grad=True)
    prior = correction.abs()
    total = restoration_total({'total': (correction - 0.2).square() + 0.1 * prior,
                               'residual_outside': prior})
    total.backward()
    assert correction.grad.item() < 0


@pytest.mark.parametrize('focus,hold', [(0.5, 0), (0, 1)])
def test_conflicting_preservation_penalties_refused(focus, hold):
    with pytest.raises(ValueError):
        restoration_total({}, focus, hold)


def test_flag_recorded_in_checkpoint_experiment():
    mine, _ = parse(['--manifest', 'unused.jsonl', '--run05-restoration'])
    assert experiment_of(mine)['run05_restoration'] is True
