"""Check existing CUDA native research masters against CPU and Python CUDA."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_native_aces_masters import read_exr
from tools.rudra_aces_export import load_model, read_still, infer
from tools.summarize_aces_generated import sha
from training.aces_reference import reference_view


def main():
    root = Path('reports/native_aces_v2_setup_20261025')
    output = root / 'cuda_master_verification.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    view = reference_view()
    view.set_threads(2)
    checkpoint = Path('G:/rudra_training/aces_guarded_projected600_20261024/step600.pt')
    model, _ = load_model(str(checkpoint), 'cuda')
    rows = []
    for index in range(2):
        source_path = root / 'frames' / f'{index}.png'
        source = read_still(source_path)
        expected, mask, _, _ = infer(model, source, 'cuda', include_fully_clipped=True,
                                     clipped_smoothing='hybrid64')
        actual, actual_mask = read_exr(root / 'native_master_cuda' / f'{index}.exr')
        cpu, cpu_mask = read_exr(root / 'native_master_cpu' / f'{index}.exr')
        comparisons = {}
        for name, reference in [('python_cuda', expected), ('native_cpu', cpu)]:
            code = float(np.abs(view.forward(actual) - view.forward(reference)).max() * 255)
            relative = float((np.abs(actual.astype(np.float64) - reference) /
                              (np.abs(reference) + 1e-3)).max())
            comparisons[name] = {'exact': bool(np.array_equal(actual, reference)),
                                 'max_display_code_difference': code,
                                 'max_relative_difference': relative,
                                 'passes': code <= .25 and relative <= .005}
        row = {'frame': index, 'source_sha256': sha(source_path),
               'mask_exact': bool(np.array_equal(actual_mask, mask) and
                                  np.array_equal(actual_mask, cpu_mask)),
               'comparisons': comparisons}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {'rows': rows, 'checkpoint_sha256': sha(checkpoint),
              'scope': 'Two 720p DEV frames, guarded600 research weights; not release qualification.',
              'passes': all(r['mask_exact'] and all(c['passes'] for c in r['comparisons'].values())
                            for r in rows)}
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    assert result['passes']


if __name__ == '__main__':
    main()
