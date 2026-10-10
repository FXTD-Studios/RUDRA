"""Temporal consistency for video (10 Oct 2026).

`rudra video` holds the per-frame values -- the shadow weight, the residual
scale and the curve -- steady within a shot, with a time constant in seconds
(--temporal-smoothing, default 0.25), and every report carries a flicker
measure: how far the HDR's average exposure moved from one frame to the next
beyond what the source's did, in EV. The native port (core/video_predict)
holds the same values the same way and is held to these goldens.
"""
import argparse
import json
import math
import shutil

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rudra import video


def test_hold_follows_the_weights_retention_and_resets_at_a_cut():
    smoother = video.ShadowSmoother(.5, .15)
    black = np.zeros((3, 36, 64))
    smoother.update(black, 1.0)
    assert smoother.hold('scale', 2.0) == 2.0                      # a shot starts at its own value
    smoother.update(black, 1.0)
    assert smoother.hold('scale', 4.0) == pytest.approx(3.0)       # .5*2 + .5*4
    curve = smoother.hold('curve', np.array([0.0, 1.0]))
    assert np.array_equal(curve, [0.0, 1.0])                       # each key its own history
    smoother.update(np.ones_like(black), 1.0)                      # a cut
    assert smoother.hold('scale', 8.0) == 8.0


def test_retention_is_a_time_constant():
    assert video.retention_for('24', 0) == 0.0
    assert video.retention_for('24', .25) == math.exp(-1/6)
    assert video.retention_for('30000/1001', .25) == math.exp(-1/(30000/1001*.25))
    # A longer constant or a higher rate holds more of the past each frame.
    assert video.retention_for('60', .25) > video.retention_for('24', .25)


@pytest.mark.parametrize('shape', [(36, 64), (72, 128), (101, 173), (20, 30)])
def test_area_mean_is_torchs_area_interpolation(shape):
    plane = np.random.default_rng(1).random(shape)
    want = F.interpolate(torch.from_numpy(plane)[None, None], size=video.EXPOSURE_THUMB, mode='area')[0, 0].numpy()
    assert np.allclose(video.area_mean(plane), want, rtol=1e-12, atol=1e-12)


def test_exposure_measures_in_ev():
    flat = np.full((90, 160), .18)
    assert video.exposure_ev(flat*2)-video.exposure_ev(flat) == pytest.approx(1.0, abs=1e-2)
    contract = dict(transfer='srgb', primaries='rec709')
    white = np.ones((8, 8, 3), np.float32)
    assert np.allclose(video.source_luminance(white, contract), 1.0)
    assert np.allclose(video.hdr_luminance(white*.01), 1.0)        # 100 nits is diffuse white


class PumpingPredictor:
    """The analytic picture times a gain that jumps every frame: the shot's
    picture pumps while the source holds still."""
    def __init__(self, *_):
        self.n = 0

    def predict(self, rgb, contract, smoother):
        weight, cut = smoother.update(rgb[::4, ::4].transpose(2, 0, 1), 1.0)
        gain = smoother.hold('gain', 1.0 + .5*(self.n % 2))
        self.n += 1
        return rgb*.01*gain, weight, cut


@pytest.fixture
def media_tools():
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')


def convert(tmp_path, monkeypatch, seconds):
    source, target = tmp_path/'still.mp4', tmp_path/f'out_{seconds}.mp4'
    if not source.exists():
        video.run([video.executable('ffmpeg'), '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i',
                   'testsrc2=size=96x64:rate=24:duration=0.5', '-vf', 'tblend=all_mode=average,format=yuv420p',
                   '-frames:v', '1', '-c:v', 'libx264', '-color_range', 'tv', '-x264-params',
                   'colorprim=bt709:transfer=bt709:colormatrix=bt709', str(tmp_path/'one.mp4')])
        video.run([video.executable('ffmpeg'), '-v', 'error', '-nostdin', '-n', '-stream_loop', '11', '-i',
                   str(tmp_path/'one.mp4'), '-c:v', 'libx264', '-qp', '0', '-color_range', 'tv', '-x264-params',
                   'colorprim=bt709:transfer=bt709:colormatrix=bt709', '-r', '24', str(source)])
    monkeypatch.setattr(video, 'Predictor', PumpingPredictor)
    parser = argparse.ArgumentParser()
    video.add_arguments(parser)
    args = parser.parse_args([str(source), '--output', str(target), '--checkpoint', 'checkpoints/sdr2hdr_shadow_v1.pt',
                              '--preset', 'ultrafast', '--audio', 'none', '--temporal-smoothing', str(seconds)])
    args.input_transfer = args.input_primaries = 'rec709'
    assert video.convert_video(args) == 0
    return json.loads(target.with_suffix('.mp4.json').read_text())


def test_the_report_measures_flicker_and_the_hold_reduces_it(tmp_path, monkeypatch, media_tools):
    off = convert(tmp_path, monkeypatch, 0)
    held = convert(tmp_path, monkeypatch, .25)
    assert off['temporal_smoothing'] == 0 and held['temporal_smoothing'] == .25
    assert off['temporal']['retention'] == 0 and held['temporal']['retention'] == math.exp(-1/6)
    assert off['frames'][0]['flicker_ev'] is None                  # a shot's first frame is not scored
    n = off['temporal']['frames_scored']
    assert n == len(off['frames'])-len(off['scene_cuts'])   # the first frame is a cut
    # The source holds still and the gain alternates 1 and 1.5: log2(1.5) EV a frame.
    assert off['temporal']['flicker_ev_mean'] == pytest.approx(math.log2(1.5), rel=.05)
    assert held['temporal']['flicker_ev_mean'] < off['temporal']['flicker_ev_mean']*.3
    assert held['warnings'] == []
