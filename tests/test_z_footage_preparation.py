"""Safety boundaries for the new footage preparation, using tiny fixtures."""
import json
from pathlib import Path
import numpy as np
import pytest
from pipeline import prepare_z_footage as z


def test_take_variants_stay_together():
    def row(name): return {'source_path':name}
    assert z.take_id(row('A031C043_240806_RQBY.mxf')) == z.take_id(row('A031C043_240806_RQBY_S001.mxf'))
    assert z.take_id(row('A_0027C015_251224_002111_p1F4S.mxf.mov')) == z.take_id(row('A_0027C015_251224_002111_p1F4S.mxf_004.mov'))
    assert z.take_id(row('A031C043_240806_RQBY.mxf')) != z.take_id(row('A031C043_240807_RQBY.mxf'))


def test_pq_cannot_enter_scene_camera_decoder():
    with pytest.raises(ValueError,match='Unsupported scene input'):
        z.linear_camera(np.zeros((2,2,3),np.float32),{'source_interpretation':'PQ / BT.2020'})


def test_installed_recipe_is_monotone_and_preserves_hdr_headroom():
    recipe=z.shot_recipe('test_take')
    recipe.update(codec='none',trim_ev=0)
    ramp=np.geomspace(.001,200,64,dtype=np.float32)[None,:,None].repeat(3,2)
    sdr,hdr,_=z.render.render_pair(ramp,recipe)
    assert np.all(np.diff(sdr[0,:,1].astype(float))>=0)
    assert np.all(np.diff(hdr[0,:,1])>=-1e-5)
    assert 500 < hdr.max()*203 <= 1000.1
    encoded,_=z.encode_hdr_u16(hdr,z.render.STORAGE)
    decoded=z.decode_hdr_u16(encoded,z.render.STORAGE)
    assert np.allclose(decoded,hdr,rtol=.001,atol=.0001)


def test_planner_excludes_pq_and_cross_split_flags_and_deduplicates(tmp_path):
    sources=[]
    for name,group,split,frames,curve in [
        ('A001C001_260101_a.mov','mataha','test',10,'ARRI LogC4'),
        ('A001C001_260101_a_001.mov','mataha','test',20,'ARRI LogC4'),
        ('pq.mov','amin_elrihany','train',30,'PQ / BT.2020'),
        ('blocked.mov','sudan','val',10,'ARRI LogC4')]:
        r=dict(source_path=name,production_group=group,split=split,frame_count=frames,
               width=100,height=100,source_interpretation=curve)
        r['take_id']=z.take_id(r)
        sources.append(r)
    z.write_rows(tmp_path/'qc_results.jsonl',[dict(source=r,error=None) for r in sources])
    z.dump(tmp_path/'similarity_flags.json',[dict(a=z.key(sources[3]),b=z.key(sources[2]),cross_split=True)])
    p=z.plan(tmp_path)
    assert len(p['sources'])==1
    assert p['sources'][0]['frame_count']==20
    assert p['sources'][0]['split']=='test'


def test_changed_source_is_rejected(tmp_path):
    p=tmp_path/'source.mov';p.write_bytes(b'original')
    r=dict(source_path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    z.unchanged(r)
    p.write_bytes(b'replaced media')
    with pytest.raises(ValueError,match='Source changed'):
        z.unchanged(r)
