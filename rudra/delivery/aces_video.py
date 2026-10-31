"""Signed ACEScg grading master to a derived ACES2 HDR1000 RGB16 video."""
from pathlib import Path
import math
from rudra.aces_view import ACES2View
from rudra.delivery.video import encode_sequence


class ACES2HDR1000View(ACES2View):
    DISPLAY='Rec.2100-PQ - Display'
    VIEW='ACES 2.0 - HDR 1000 nits (Rec.2020)'


def encode_acescg_rgb16(frames,output:Path,fps=24.,ocio_threads=1):
    """Apply the official display transform, then encode its PQ codes losslessly.

    Scene-linear signed values are passed directly to OCIO without clipping.
    This video is display-referred; keep the float32 EXR as grading master.
    """
    if not math.isfinite(fps) or fps<=0:raise ValueError('positive finite fps required')
    output=Path(output).with_suffix('.mkv')
    if output.exists():raise ValueError('new output required')
    view=ACES2HDR1000View(threads=ocio_threads)
    return encode_sequence((view.forward(frame) for frame in frames),output,
                           target='ffv1rgb16',fps=fps,peak_nits=1000.,
                           source_space='rec2020',shoulder=False,input_encoding='pq_codes')
