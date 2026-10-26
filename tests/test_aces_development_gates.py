from copy import deepcopy
from tools.compare_aces_development import assess


def test_per_image_roundtrip_and_shadow_lift_cannot_be_hidden_by_lower_mean_loss():
    row={'target_errors':{'analytic':{'all_mse':1.,'highlight_mse':1.,'shadow_mse':1.},
                          'model':{'all_mse':.5,'highlight_mse':.5,'shadow_mse':.5}},
         'roundtrip':{'ordinary':{'within1_fraction':1.}},'shadow_lift_gt1code':0}
    report={'evaluation_role':'scene-disjoint development','rows':[row,deepcopy(row)]}
    assert assess(report)['numerical_pass']
    report['rows'][1]['shadow_lift_gt1code']=1
    assert not assess(report)['numerical_pass']
    report['rows'][1]['shadow_lift_gt1code']=0
    report['rows'][1]['roundtrip']['ordinary']['within1_fraction']=.998
    assert not assess(report)['numerical_pass']
