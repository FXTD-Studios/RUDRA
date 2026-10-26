import numpy as np
from tools.prepare_aces_consistent_pairs import save_source


def test_saved_source_round_trips_saturated_colours_through_training_loader(tmp_path):
    colours=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0,0,0],[1,1,1],[.2,.4,.6]],np.float32)
    source=np.tile(colours[None],(16,1,1))
    decoded,error=save_source(tmp_path/'source.png',source)
    assert decoded.shape==source.shape and error<=.1
    np.testing.assert_allclose(decoded,source,atol=.1/255,rtol=0)
