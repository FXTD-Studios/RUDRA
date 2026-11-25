"""1.3.1 (25 Nov 2026): AI fill's flicker fallback one region at a time (fill_master fallback='regions').

A frame holds a steady lamp and an arc whose fill jumps for one frame. 'spikes' (RUDRA 1.1) would give the whole
frame back to the master; 'regions' gives back only the arc, bit for bit, and keeps the lamp's fill."""
from __future__ import annotations

import numpy as np

from rudra.genfill import run as gr


def scene(arc_gain: float, h: int = 96, w: int = 128):
    master = np.full((h, w, 3), 0.2, np.float32)
    alpha = np.zeros((h, w), np.float32)
    alpha[10:30, 10:40] = 1.0          # the lamp
    alpha[60:90, 80:120] = 1.0         # the arc, far from it
    fill = master.copy()
    fill[10:30, 10:40] = 4.0           # steady
    fill[60:90, 80:120] = 4.0 * arc_gain
    out = master + alpha[..., None] * (fill - master)
    return master, out, alpha


def tri(master, out, alpha):
    return (gr.small(master), gr.small(out), gr.small(alpha))


def test_regions_are_the_separate_fill_areas_and_near_specks_join():
    _, _, alpha = scene(1.0)
    labels, n = gr.fill_regions(alpha)
    assert n == 2 and labels[0, 0] == 0 and labels[15, 15] != labels[70, 100]
    alpha[32, 20] = 1.0                # a glint 2 px under the lamp: the lamp's region
    labels, n = gr.fill_regions(alpha)
    assert n == 2 and labels[32, 20] == labels[15, 15]


def test_only_the_spiking_region_falls_back_bit_for_bit():
    prev, cur, nxt = scene(1.0), scene(4.0), scene(1.0)   # the arc jumps 2 stops for one frame
    labels, n = gr.fill_regions(cur[2])
    bad = gr.flickering_regions(tri(*prev), tri(*cur), tri(*nxt), labels, n)
    arc = labels[70, 100]
    assert bad == [arc]
    out, alpha = gr.revert_regions(cur[0], cur[1], cur[2], labels, bad)
    np.testing.assert_array_equal(out[60:90, 80:120], cur[0][60:90, 80:120])   # the master's own pixels
    assert (alpha[60:90, 80:120] == 0).all()
    np.testing.assert_array_equal(out[10:30, 10:40], cur[1][10:30, 10:40])     # the lamp keeps its fill
    assert (alpha[10:30, 10:40] == 1).all()


def test_a_fill_that_starts_or_stays_steady_is_kept():
    steady = scene(1.0)
    labels, n = gr.fill_regions(steady[2])
    assert gr.flickering_regions(tri(*steady), tri(*steady), tri(*steady), labels, n) == []
    # a jump in only one direction (the arc brightens and stays): kept, as 'spikes' keeps it
    assert gr.flickering_regions(tri(*scene(1.0)), tri(*scene(4.0)), tri(*scene(4.0)), labels, n) == []
    # the first and last frames have nothing on one side
    assert gr.flickering_regions(None, tri(*steady), tri(*steady), labels, n) == []


def test_regions_is_a_fallback_choice():
    assert 'regions' in gr.FALLBACKS and gr.FALLBACKS[:2] == ('none', 'spikes')
