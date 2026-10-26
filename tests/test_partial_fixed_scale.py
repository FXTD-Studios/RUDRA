import json
import pytest
from tools.train_partial_display import load_fixed_display_scale


def test_fixed_scale_accepts_float_equivalent_keys(tmp_path):
    p=tmp_path/'scale.json';p.write_text(json.dumps({'denominators':{'-3':50,'0':.1,'3':.01}}))
    assert load_fixed_display_scale(p,[-3.,0.,3.])=={-3.:50.,0.:.1,3.:.01}


@pytest.mark.parametrize('values',[{'0':0},{'0':float('nan')},{'0':float('inf')},{'0':-.1},{'3':1}])
def test_fixed_scale_rejects_invalid_or_missing_values(tmp_path,values):
    p=tmp_path/'scale.json';p.write_text(json.dumps({'denominators':values}))
    with pytest.raises(ValueError):load_fixed_display_scale(p,[0.])
