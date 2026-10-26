"""Research-only complete Run05 graph; not a native v1 model package."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.export_bench_pairs import load_model
from tools.evaluate_run05_dequant import sha


class DirectHDR(torch.nn.Module):
    """Fixed BT1886 view and default reconstruction controls, full HDR output."""

    def __init__(self, model):
        super().__init__()
        if not (model.baseline_source and model.bounded_source):
            raise ValueError('requires picker baseline and bounded restoration')
        if model.gate_conditioning or model.shadow_conditioning or model.curve_head:
            raise ValueError('frame-conditioned export is not supported')
        self.model = model

    def forward(self, source):
        return self.model(source, baseline_curve=1).hdr


def probes():
    generator = torch.Generator().manual_seed(20261023)
    yield 'black', torch.zeros(1, 3, 32, 40)
    yield 'white', torch.ones(1, 3, 35, 47)
    yield 'flat', torch.full((1, 3, 64, 48), .5)
    yield 'random', torch.rand(1, 3, 65, 97, generator=generator)
    ramp = torch.linspace(0, 1, 129)[None, None, None].expand(1, 3, 33, -1).clone()
    ramp[:, 1] = 1
    ramp[:, 2] = 0
    yield 'clipped_colour_ramp', ramp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('use a new output directory')
    torch.set_num_threads(2)
    net = load_model(args.checkpoint, torch.device('cpu')).eval()
    wrapper = DirectHDR(net).eval()
    cases = list(probes())
    with torch.inference_mode():
        graph = torch.jit.trace(wrapper, cases[0][1], check_inputs=[(x,) for _, x in cases[1:]])
        args.out.mkdir(parents=True)
        graph.save(str(args.out / 'direct_hdr.pt'))
        restored = torch.jit.load(str(args.out / 'direct_hdr.pt')).eval()
        evidence = []
        for name, source in cases:
            expected, actual = wrapper(source), restored(source)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            if not torch.isfinite(actual).all():
                raise ValueError('nonfinite exported output')
            evidence.append({'case': name, 'shape': list(source.shape),
                             'max_abs_error': (actual - expected).abs().max().item()})
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
              'graph_sha256': sha(args.out / 'direct_hdr.pt'), 'parity': evidence,
              'view': 'bt1886', 'input': 'float32 1x3xHxW canonical Rec2020 SDR codes, [0,1]; batch size 1 only',
              'output': 'float32 1x3xHxW linear Rec2020, nits/10000',
              'policy': net.reconstruction_policy, 'release_authorized': False,
              'limitations': 'Research direct graph only; no ACES output-view parity, native v1 compatibility, tiling, artist controls, temporal validation or quality approval.'}
    (args.out / 'metadata.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
