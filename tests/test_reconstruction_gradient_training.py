import json
import pytest
import torch
from training import resumable_training as rt
from tests.test_dequant_training import dataset, parse_run
from tests.test_resumable_training import load, same_tensors, same_optimizer, one_thread  # noqa: F401


def test_reconstruction_gradient_resume_and_sampler_match(tmp_path):
    manifest=dataset(tmp_path)
    def run_args(path):
        return parse_run(manifest,path,'--dequant-code-weight','0','--reconstruction-gradient-weight','.01')
    mine,args=run_args(tmp_path/'full')
    experiment=rt.experiment_of(mine)
    rt.train(args,experiment=experiment)
    _,split=run_args(tmp_path/'split')
    rt.train(split,experiment=experiment,control=rt.Control(tmp_path/'STOP',None,stop_at_step=5))
    rt.train(split,resume=True,experiment=experiment)
    a,b=load(tmp_path/'full/step_0000009.pt'),load(tmp_path/'split/step_0000009.pt')
    same_tensors(a['model'],b['model']);same_optimizer(a['optimizer'],b['optimizer'])
    assert a['config']['experiment']['reconstruction_gradient_weight']==.01
    assert 'dequant_code_weight' not in a['config']['experiment']
    logs=[json.loads(line) for line in (tmp_path/'full/train.jsonl').read_text().splitlines()]
    assert any('reconstruction_gradient' in r for r in logs)
    _,control=run_args(tmp_path/'control')
    rt.train(control,experiment={'run05_restoration':True})
    c=load(tmp_path/'control/step_0000009.pt')
    assert a['rng']['python']==c['rng']['python']
    assert a['rng']['numpy']==c['rng']['numpy']
    assert torch.equal(a['rng']['torch_cpu'],c['rng']['torch_cpu'])
    with pytest.raises(rt.ResumeError,match='configuration differs'):
        rt.train(split,resume=True,experiment=dict(experiment,reconstruction_gradient_weight=.02))


@pytest.mark.parametrize('weight',['-1','nan','inf'])
def test_invalid_weight(weight):
    mine,_=rt.parse(['--manifest','unused','--reconstruction-gradient-weight',weight])
    with pytest.raises(ValueError,match='finite and nonnegative'):
        rt.experiment_of(mine)
