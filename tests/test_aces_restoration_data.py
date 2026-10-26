import numpy as np
from tools.prepare_aces_restoration_data import clean_frame


def test_balanced_procedural_batch_contains_neutral_dark_and_bright_examples():
    rng=np.random.default_rng(20261027);counts=np.zeros(3,int)
    for i in range(24):
        x,_=clean_frame(rng,i,size=64,coverage='balanced')
        assert np.isfinite(x).all() and x.min()>=.0199 and x.max()<.981
        neutral=x.max(-1)-x.min(-1)<.01
        counts+=[neutral.sum(),(neutral&(x.max(-1)<.12)).sum(),(neutral&(x.min(-1)>.75)).sum()]
    assert (counts>0).all()
