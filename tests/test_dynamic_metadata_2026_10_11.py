"""Dynamic HDR metadata in movie exports (10 Oct 2026).

A PQ master (HDR10, ProRes in PQ) gets, beside it and per shot of the export's
own cuts: <name>_hdr10plus.json (x265's --dhdr10-info / hdr10plus_tool), the
Dolby Vision `dovi_tool generate` config, the RUDRA analysis and the HDR10+
scene statistics. HDR10 carries HDR10+ in the HEVC when this ffmpeg's x265 was
built with it (QC then checks the first frame); otherwise the report says why.
Validated by hand with hdr10plus_tool extract/inject and dovi_tool generate.
"""
import argparse
import json
import shutil

import numpy as np
import pytest

from rudra import video
from rudra.delivery import metadata


def stats_of(levels):
    out = []
    for i, level in enumerate(levels):
        frame = np.full((8, 8, 3), level, np.float64)
        frame[0, 0] = (level*4, level*2, level)
        out.append(metadata.analyze_frame(frame, i))
    return out


def test_shots_come_from_the_exports_cuts():
    assert metadata.shots_from_cuts([0, 5, 9], 12) == [(0, 5), (5, 4), (9, 3)]
    assert metadata.shots_from_cuts([], 3) == [(0, 3)]
    assert metadata.shots_from_cuts([0, 0, 2, 7], 4) == [(0, 2), (2, 2)]   # beyond the end, twice: ignored


def test_the_dovi_config_is_dovi_tools_shape_and_clamps():
    stats = stats_of([0.001, 0.001, 2000.0])
    config = metadata.to_dovi_generate_json(stats, [(0, 2), (2, 1)])
    dark, bright = (shot["metadata_blocks"][0]["Level1"] for shot in config["shots"])
    assert dark == {"min_pq": 12, "max_pq": 2081, "avg_pq": 1229}       # dovi_tool's CM v4.0 clamps
    assert bright["max_pq"] == metadata._pq12(8000.0) and bright["avg_pq"] < bright["max_pq"] and bright["min_pq"] <= 12


def test_hdr10plus_for_x265_is_per_frame_per_shot_in_tenths_of_a_nit():
    stats = stats_of([100.0, 100.0, 400.0])
    data = metadata.to_hdr10plus_x265_json(stats, [(0, 2), (2, 1)])
    frames = data["SceneInfo"]
    assert len(frames) == 3 and [f["SceneId"] for f in frames] == [0, 0, 1]
    assert [f["SequenceFrameIndex"] for f in frames] == [0, 1, 2]
    lum = frames[2]["LuminanceParameters"]
    assert lum["MaxScl"] == [16000, 8000, 4000]                           # 1600, 800, 400 nits
    assert lum["LuminanceDistributions"]["DistributionIndex"] == [1, 5, 10, 25, 50, 75, 90, 95, 99]
    values = lum["LuminanceDistributions"]["DistributionValues"]
    assert values == sorted(values) and values[4] == 4000
    assert frames[0]["LuminanceParameters"] == frames[1]["LuminanceParameters"]   # constant within a shot
    assert data["SceneInfoSummary"] == {"SceneFirstFrameIndex": [0, 2], "SceneFrameNumbers": [2, 1]}


def test_x265_param_paths_escape_drive_letters_and_backslashes():
    assert video.x265_param_path(r"C:\work\a.json") == r"C\:\\work\\a.json"
    assert video.x265_param_path("/o'b dir/a.json") == "/o\\'b\\ dir/a.json"   # a quote and a blank, kept


class StandInPredictor:
    def __init__(self, *_):
        pass

    def predict(self, rgb, contract, smoother):
        weight, cut = smoother.update(rgb[::4, ::4].transpose(2, 0, 1), 1.0)
        return rgb*.03, weight, cut


@pytest.fixture
def media_tools():
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')


def export(tmp_path, monkeypatch, name, *flags):
    source = tmp_path/'source.mp4'
    if not source.exists():
        video.run([video.executable('ffmpeg'), '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i',
                   'testsrc2=size=96x64:rate=24:duration=0.25', '-c:v', 'libx264', '-color_range', 'tv',
                   '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709', str(source)])
    monkeypatch.setattr(video, 'Predictor', StandInPredictor)
    parser = argparse.ArgumentParser()
    video.add_arguments(parser)
    args = parser.parse_args([str(source), '--output', str(tmp_path/name), '--checkpoint', str(source),
                              '--preset', 'ultrafast', '--audio', 'none', *flags])
    args.input_transfer = args.input_primaries = 'rec709'
    assert video.convert_video(args) == 0
    return json.loads((tmp_path/(name+'.json')).read_text())


def test_an_hdr10_export_writes_its_dynamic_metadata(tmp_path, monkeypatch, media_tools):
    report = export(tmp_path, monkeypatch, 'master.mp4')
    meta = report['hdr_metadata']
    assert report['dynamic_metadata'] == 'on' and meta['shots'] >= 1
    for name in ('master.mp4_hdr10plus.json', 'master.mp4_dovi_generate.json', 'master.mp4_hdr_analysis.json',
                 'master.mp4_hdr10plus_scenes.json'):
        assert (tmp_path/name).exists(), name
    frames = json.loads((tmp_path/'master.mp4_hdr10plus.json').read_text())['SceneInfo']
    assert len(frames) == len(report['frames'])
    embedded = meta['hdr10plus']['embedded']
    assert embedded == video.hdr10plus_capable()
    if embedded:
        assert report['qc']['hdr10plus'] == {'passed': True} and video.has_hdr10plus(tmp_path/'master.mp4')
    else:
        assert 'hdr10plus_tool' in meta['hdr10plus']['reason']
    assert not list(tmp_path.glob('rudra-*'))
    # Another container is another name; metadata already there for this one is refused before any work.
    assert export(tmp_path, monkeypatch, 'master.mkv')['hdr_metadata']['hdr10plus']['file'] == 'master.mkv_hdr10plus.json'
    (tmp_path/'again.mp4_dovi_generate.json').write_text('{}')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        export(tmp_path, monkeypatch, 'again.mp4')


def test_hlg_and_off_write_none(tmp_path, monkeypatch, media_tools):
    assert export(tmp_path, monkeypatch, 'hlg.mp4', '--format', 'hlg')['hdr_metadata'] is None
    assert export(tmp_path, monkeypatch, 'off.mp4', '--dynamic-metadata', 'off')['hdr_metadata'] is None
    assert not list(tmp_path.glob('*_hdr10plus.json'))
