import numpy as np
import pytest
from tools.train_aces_spatial import load_training_rows,sample_crop


def test_validation_rows_are_rejected_before_file_access():
    with pytest.raises(ValueError,match='TRAIN'):
        load_training_rows({'rows':[{'split':'val'}]})


def test_random_crop_sampling_varies_location_and_keeps_source_target_aligned():
    source=np.arange(64*80*3,dtype=np.float32).reshape(64,80,3)
    points=np.array([[0,0],[63,79],[32,40]])
    data=[(source,source+100,points,points)]
    rng=np.random.default_rng(22);locations=[]
    for _ in range(30):
        s,t,loc=sample_crop(data,rng,16)
        np.testing.assert_array_equal((t-s).numpy(),np.full(s.shape,100,np.float32))
        locations.append(loc[1:3])
    assert len(set(locations))>20
