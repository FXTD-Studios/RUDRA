"""Official OCIO ACES2 reference view for research and delivery validation.

This is a CPU NumPy transform, not a differentiable training layer. Inversion
provides a view-consistent ACEScg reconstruction, not recovered scene exposure
or missing highlight detail. No nits-per-unit scaling is inferred.
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class ACES2View:
    CONFIG = 'cg-config-v4.0.0_aces-v2.0_ocio-v2.5'
    DISPLAY = 'Rec.1886 Rec.709 - Display'
    VIEW = 'ACES 2.0 - SDR 100 nits (Rec.709)'

    def __init__(self,threads=1):
        self.set_threads(threads)
        import PyOpenColorIO as ocio
        config = ocio.Config.CreateFromBuiltinConfig(self.CONFIG)
        transform = ocio.DisplayViewTransform(src='ACEScg', display=self.DISPLAY, view=self.VIEW)
        self._forward = config.getProcessor(transform).getDefaultCPUProcessor()
        transform.setDirection(ocio.TRANSFORM_DIR_INVERSE)
        self._inverse = config.getProcessor(transform).getDefaultCPUProcessor()
        self.identity = {'config': self.CONFIG, 'display': self.DISPLAY, 'view': self.VIEW,
                         'working_space': 'ACEScg', 'ocio_version': ocio.__version__,
                         'config_sha256': hashlib.sha256(config.serialize().encode()).hexdigest()}

    def set_threads(self,threads):
        if not isinstance(threads,int) or not 1<=threads<=16:
            raise ValueError('OCIO threads must be an integer from 1 to 16')
        self.threads=threads

    def _process(self,processor,out):
        # OCIO releases the GIL. Each worker writes a disjoint packed RGB span;
        # the fixed processor has no changing dynamic properties.
        if self.threads==1 or len(out)<262144:
            processor.applyRGB(out)
        else:
            with ThreadPoolExecutor(self.threads) as pool:
                list(pool.map(processor.applyRGB,np.array_split(out,self.threads)))

    @staticmethod
    def _array(rgb):
        array = np.asarray(rgb, dtype=np.float32)
        if array.ndim < 2 or array.shape[-1] != 3 or not np.isfinite(array).all():
            raise ValueError('expected finite RGB array with last dimension3')
        return array

    def inverse(self, display_rgb):
        array = self._array(display_rgb)
        if np.any((array < 0) | (array > 1)):
            raise ValueError('inverse requires display RGB codes in [0,1]')
        out = np.ascontiguousarray(array).reshape(-1,3).copy()
        self._process(self._inverse,out)
        if not np.isfinite(out).all():
            raise ValueError('nonfinite ACES inverse result')
        return out.reshape(array.shape)

    def forward(self, acescg):
        array = self._array(acescg)
        out = np.ascontiguousarray(array).reshape(-1,3).copy()
        self._process(self._forward,out)
        if not np.isfinite(out).all():
            raise ValueError('nonfinite ACES view result')
        return out.reshape(array.shape)
