import json

import pytest
import torch

from pipeline.run05_corpus import build
from training import resumable_training as rt
from tests.test_run05_corpus import _sources
from tests.test_resumable_training import argv, load, same_tensors, same_optimizer, one_thread  # noqa: F401


EXTRA = ['--bounded-source', '--reconstruction-policy', 'observed_clip_v1',
         '--crushed-hold', '0', '--crushed-hold-domain', 'postclamp',
         '--augmentation-strength', '0', '--degradation-probability', '0',
         '--run05-restoration', '--dequant-code-weight', '.003']


def dataset(tmp_path):
    source = _sources(tmp_path / 'sources')
    build(source, tmp_path / 'pairs', variants=3, codec_p=0,
          target_mode='source_preserving', reconstruction_policy='observed_clip_v1')
    return tmp_path / 'pairs/manifest.jsonl'


def parse_run(manifest, out, *extra):
    mine, args = rt.parse(argv(manifest, out, *EXTRA, *extra))
    args.pipeline_corpus_ev = None  # run05 rows carry their own known tonemap_ev
    return mine, args


def test_dequant_training_resumes_exactly_and_records_loss(tmp_path):
    manifest = dataset(tmp_path)
    ref, split = tmp_path / 'ref', tmp_path / 'split'
    mine, args = parse_run(manifest, ref)
    rt.train(args, experiment=rt.experiment_of(mine))
    mine, args = parse_run(manifest, split)
    rt.train(args, experiment=rt.experiment_of(mine), control=rt.Control(split / 'STOP', None, stop_at_step=5))
    rt.train(args, resume=True, experiment=rt.experiment_of(mine))
    a, b = load(ref / 'step_0000009.pt'), load(split / 'step_0000009.pt')
    same_tensors(a['model'], b['model'])
    same_optimizer(a['optimizer'], b['optimizer'])
    assert a['config']['experiment']['dequant_code_weight'] == .003
    logs = [json.loads(line) for line in (ref / 'train.jsonl').read_text().splitlines()]
    assert any(row.get('dequant_code', 0) > 0 for row in logs)
    # The paired control must see the same crop stream and RNG trajectory.
    _, control_args = parse_run(manifest, tmp_path / 'control')
    rt.train(control_args, experiment={'run05_restoration': True})
    control = load(tmp_path / 'control/step_0000009.pt')
    assert a['rng']['python'] == control['rng']['python']
    assert a['rng']['numpy'] == control['rng']['numpy']
    assert torch.equal(a['rng']['torch_cpu'], control['rng']['torch_cpu'])
    changed = dict(rt.experiment_of(mine), dequant_code_weight=.004)
    with pytest.raises(rt.ResumeError, match='configuration differs'):
        rt.train(args, resume=True, experiment=changed)


def test_dequant_rejects_regraded_or_wrong_view_data(tmp_path):
    manifest = dataset(tmp_path)
    mine, args = parse_run(manifest, tmp_path / 'run', '--augmentation-strength', '1')
    with pytest.raises(ValueError, match='unaugmented'):
        rt.train(args, experiment=rt.experiment_of(mine))
    args.augmentation_strength = 0
    rows = [dict(json.loads(line), declared_view='aces') for line in manifest.read_text().splitlines()]
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='BT1886'):
        rt.train(args, experiment=rt.experiment_of(mine))


@pytest.mark.parametrize('weight', ['-1', 'nan', 'inf'])
def test_invalid_dequant_weights_are_rejected(weight):
    mine, _ = rt.parse(['--manifest', 'unused', '--dequant-code-weight', weight])
    with pytest.raises(ValueError, match='finite and nonnegative'):
        rt.experiment_of(mine)
