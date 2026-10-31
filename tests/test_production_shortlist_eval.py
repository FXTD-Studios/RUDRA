import json

import pytest

from training.production_eval_kit import digest
from training.production_shortlist_eval import load_shortlist, run

SCENES = {'s1': 3, 's2': 2, 's3': 2}


def make_kit(tmp_path):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'validation.jsonl').write_text('{"asset_id": "x"}\n')
    contract = dict(validation_sha256=digest(kit / 'validation.jsonl'), frames=7, scenes=3,
                    conditions=['clean', 'hard'])
    (kit / 'contract.json').write_text(json.dumps(contract))
    return kit


def rows(offset, per_scene=None):
    out = []
    for scene, n in SCENES.items():
        for i in range(n):
            out.append(dict(clip=scene, frame=f'{scene}_{i}',
                            pu_psnr_db=30.0 + i + offset + (per_scene or {}).get(scene, 0.0)))
    return {'results': out}


def fake_evaluator(gains, calls):
    """Stands in for production_eval_kit.evaluate: same files, synthetic scores."""
    def evaluate(kit, checkpoint, out):
        calls.append(checkpoint.name)
        assert not out.exists()
        offset = gains[checkpoint.name]
        conditions = {}
        for cond in ('clean', 'hard'):
            (out / cond).mkdir(parents=True)
            (out / cond / 'test.json').write_text(json.dumps(rows(offset[cond], offset.get('scene'))))
            (out / cond / 'baseline_bt1886.json').write_text(json.dumps(rows(0.0)))
            mean = offset[cond] + sum((offset.get('scene') or {}).values()) / 3
            ci = [mean - 0.05, mean + 0.05]
            passed = ci[0] >= -0.1 if cond == 'clean' else ci[0] > 0
            conditions[cond] = dict(metrics={'pu_psnr_db': dict(mean=mean, ci95=ci, frames=7, scenes=3)},
                                    passed=passed)
        (out / 'comparison.json').write_text(json.dumps(dict(
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            contract_sha256=digest(kit / 'contract.json'), conditions=conditions,
            validation_pu_gate_passed=all(c['passed'] for c in conditions.values()),
            release_authorized=False)))
    return evaluate


def setup(tmp_path, gains):
    kit = make_kit(tmp_path)
    entries = {}
    for name in gains:
        p = tmp_path / 'ckpt' / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(name.encode() * 10)
        entries[name] = dict(label=name.replace('.pt', ''), path=str(p), sha256=digest(p))
    cands = [entries[n] for n in gains if n.startswith('cand')]
    refs = [entries[n] for n in gains if n.startswith('ref')]
    shortlist = tmp_path / 'shortlist.json'
    shortlist.write_text(json.dumps(dict(candidates=cands, references=refs)))
    return kit, shortlist


GAINS = {'cand_a.pt': dict(clean=0.9, hard=0.30), 'cand_b.pt': dict(clean=1.0, hard=0.28),
         'cand_c.pt': dict(clean=0.8, hard=-0.2), 'ref_old.pt': dict(clean=0.7, hard=0.1)}


def test_batch_scores_compares_and_applies_frozen_rule(tmp_path):
    kit, shortlist = setup(tmp_path, GAINS)
    calls = []
    summary = run(shortlist, kit, tmp_path / 'eval', evaluate=fake_evaluator(GAINS, calls))
    assert sorted(calls) == sorted(GAINS)
    sel = summary['selection']
    # a and b tie within 0.05 dB on hard; the rule prefers b's higher clean gain
    assert sel['proposed_for_next_stage'] == 'cand_b'
    assert sel['not_selected']['cand_c'] == 'failed validation PU gate'
    assert summary['release_authorized'] is False and summary['promotion'] is False
    vs = summary['candidates']['cand_a']['vs_references']['ref_old']
    assert vs['hard']['ci_verdict'] == 'a_better'
    assert 'cand_a__vs__cand_b' in summary['candidate_pairs']
    assert summary['candidates']['cand_c']['worst']['hard']['scenes_below_baseline'] == 3
    # rerun reuses finished evidence and evaluates nothing again
    calls.clear()
    again = run(shortlist, kit, tmp_path / 'eval', evaluate=fake_evaluator(GAINS, calls))
    assert calls == [] and again['selection'] == sel


def test_candidate_worse_than_reference_is_not_proposed(tmp_path):
    gains = {'cand_a.pt': dict(clean=0.2, hard=0.2), 'ref_old.pt': dict(clean=0.9, hard=0.6)}
    kit, shortlist = setup(tmp_path, gains)
    summary = run(shortlist, kit, tmp_path / 'eval', evaluate=fake_evaluator(gains, []))
    assert summary['selection']['proposed_for_next_stage'] is None
    assert 'CI-worse than reference' in summary['selection']['not_selected']['cand_a']


def test_changed_checkpoint_bytes_refused_before_any_work(tmp_path):
    kit, shortlist = setup(tmp_path, GAINS)
    (tmp_path / 'ckpt' / 'cand_a.pt').write_bytes(b'tampered')
    calls = []
    with pytest.raises(ValueError, match='bytes changed'):
        run(shortlist, kit, tmp_path / 'eval', evaluate=fake_evaluator(GAINS, calls))
    assert calls == [] and not (tmp_path / 'eval').exists()


def test_incomplete_or_foreign_output_is_never_overwritten(tmp_path):
    kit, shortlist = setup(tmp_path, GAINS)
    (tmp_path / 'eval' / 'cand_a' / 'clean').mkdir(parents=True)
    with pytest.raises(ValueError, match='incomplete evaluation'):
        run(shortlist, kit, tmp_path / 'eval', evaluate=fake_evaluator(GAINS, []))
    assert (tmp_path / 'eval' / 'cand_a' / 'clean').is_dir()


def test_dry_run_plans_without_writing(tmp_path):
    kit, shortlist = setup(tmp_path, GAINS)
    plan = run(shortlist, kit, tmp_path / 'eval', dry_run=True, evaluate=fake_evaluator(GAINS, []))
    assert plan['dry_run'] and not (tmp_path / 'eval').exists()
    assert set(plan['plan']) == {'cand_a', 'cand_b', 'cand_c', 'ref_old'}


@pytest.mark.parametrize('mutate,match', [
    (lambda d: d['candidates'].append(dict(d['candidates'][0])), 'duplicate labels'),
    (lambda d: d.update(references=[]), 'at least one'),
    (lambda d: d['candidates'][0].update(sha256='x'), 'malformed'),
])
def test_bad_shortlists_rejected(tmp_path, mutate, match):
    kit, shortlist = setup(tmp_path, GAINS)
    data = json.loads(shortlist.read_text())
    mutate(data)
    shortlist.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_shortlist(shortlist)
