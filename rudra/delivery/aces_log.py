"""ARRI LogC3 / LogC4 deliveries from the ACEScg master (scene-referred camera-log files).

The ACEScg master is converted with OpenColorIO's built-in ACES 2.0 *studio* config
(same OCIO version the model pins), colour space transform ACEScg -> 'ARRI LogC3 (EI800)'
(ARRI Wide Gamut 3) or 'ARRI LogC4' (ARRI Wide Gamut 4). No view transform, no peak or
knee: the log file carries the master's scene values. Verified against ARRI's references:
18% grey -> 0.391007 (LogC3 EI800) and 0.278396 (LogC4), neutrals stay neutral.

Code values above 1.0 cannot be stored and are clipped and counted (LogC3 EI800 tops out
near linear 55, LogC4 near 470). Bright masters can exceed 55 (seen: 221 on an exposed-up
render): use LogC4, or a negative exposure_stops for LogC3 (recorded in the metadata). The rudra.reconstructed mask cannot travel in these files:
ship the EXR master (or its mask) alongside.
"""
import subprocess
from pathlib import Path
import numpy as np

STUDIO_CONFIG = 'studio-config-v4.0.0_aces-v2.0_ocio-v2.5'
CURVES = {'logc3': ('ARRI LogC3 (EI800)', 'ARRI Wide Gamut 3'),
          'logc4': ('ARRI LogC4', 'ARRI Wide Gamut 4')}


class LogEncoder:
    def __init__(self, curve, exposure_stops=0.0):
        import PyOpenColorIO as ocio
        if curve not in CURVES:
            raise ValueError(f'unknown log curve {curve!r}')
        self.curve = curve
        self.colorspace, self.gamut = CURVES[curve]
        config = ocio.Config.CreateFromBuiltinConfig(STUDIO_CONFIG)
        self._proc = config.getProcessor('ACEScg', self.colorspace).getDefaultCPUProcessor()
        self.gain = float(2.0 ** exposure_stops)
        self.identity = {'curve': curve, 'colorspace': self.colorspace, 'gamut': self.gamut,
                         'config': STUDIO_CONFIG, 'ocio_version': ocio.__version__,
                         'exposure_stops': float(exposure_stops),
                         'exposure_note': 'linear master multiplied by 2**exposure_stops before encoding; undo it in the grade'}

    def encode(self, acescg):
        """HxWx3 ACEScg float -> (HxWx3 float codes clipped to [0,1], clipped pixel count)."""
        out = np.ascontiguousarray(acescg, np.float32).reshape(-1, 3) * np.float32(self.gain)
        self._proc.applyRGB(out)
        out = out.reshape(np.shape(acescg))
        clipped = int(((out > 1.0) | (out < 0.0)).any(-1).sum())
        return np.clip(out, 0.0, 1.0), clipped

    @staticmethod
    def to_uint16(codes):
        return np.round(codes * 65535.0).astype(np.uint16)

    def write_tiff(self, path, acescg):
        import cv2
        codes, clipped = self.encode(acescg)
        if not cv2.imwrite(str(path), cv2.cvtColor(self.to_uint16(codes), cv2.COLOR_RGB2BGR)):
            raise IOError(f'could not write {path}')
        return clipped


class ProResLogWriter:
    """16-bit RGB pipe -> ProRes 4444 XQ (12-bit 4:4:4, qscale 1) .mov via ffmpeg. Untagged
    transfer: set the clip to ARRI LogC3/AWG3 or LogC4/AWG4 in the grading app. ProRes is lossy
    (measured on a hard-edged frame: mean 0.76, max 11.6 ten-bit codes vs the TIFF; the
    RGB<->YUV conversion alone is under 0.5 code). The 16-bit TIFFs are the lossless log master."""
    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', f'{width}x{height}',
               '-r', str(fps), '-i', 'pipe:0', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
               '-c:v', 'prores_ks', '-profile:v', '5', '-qscale:v', '1', '-pix_fmt', 'yuv444p12le', '-vendor', 'apl0',
               '-colorspace', 'bt709', str(self.path)]
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, codes):
        self._p.stdin.write(LogEncoder.to_uint16(codes).astype('<u2').tobytes())

    def close(self):
        self._p.stdin.close()
        if self._p.wait() != 0:
            raise RuntimeError('ffmpeg ProRes encode failed')
        return self.path
