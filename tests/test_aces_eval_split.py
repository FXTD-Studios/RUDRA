import pytest
from tools.evaluate_aces_smoke import validate_evaluation_rows


def test_dev_evaluation_requires_disjoint_scenes_and_content():
    train={'rows':[{'split':'train','scene_id':'a','sdr_sha256':'one'}]}
    valid={'rows':[{'split':'val','scene_id':'b','sdr_sha256':'two'}]}
    validate_evaluation_rows(train,valid)
    valid['rows'][0]['scene_id']='a'
    with pytest.raises(ValueError,match='scene overlap'):validate_evaluation_rows(train,valid)
    valid['rows'][0]['scene_id']='b';valid['rows'][0]['sdr_sha256']='one'
    with pytest.raises(ValueError,match='content overlap'):validate_evaluation_rows(train,valid)
