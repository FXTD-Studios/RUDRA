"""Check saved matched-run mechanics without treating a match as quality proof."""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.resumable_training import verify_state, sha256_file


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--auxiliary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--treatment', choices=['dequant_code_weight','reconstruction_gradient_weight','signed_shadow_weight'], default='dequant_code_weight')
    parser.add_argument('--weight', type=float, default=.003)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('use a new evidence file')
    c, creason = verify_state(args.control)
    a, areason = verify_state(args.auxiliary)
    if c is None or a is None:
        raise ValueError(f'checkpoint verification failed: {creason}, {areason}')
    configs = [copy.deepcopy(state['config']) for state in (c, a)]
    for config in configs:
        config.pop('output_dir', None)
        config.pop('resumable', None)
    weight = configs[1]['experiment'].pop(args.treatment, None)
    checks = {'same_step': c['step'] == a['step'],
              'expected_auxiliary_weight': weight == args.weight and args.treatment not in configs[0]['experiment'],
              'config_except_treatment': equal(*configs),
              'code_identity': equal(c['code_sha256'], a['code_sha256']),
              'dataset_identity': equal(c['dataset'], a['dataset']),
              'parent_identity': equal(c['lineage']['warm_start_parent'], a['lineage']['warm_start_parent']),
              'sampler_position': equal(c['sampler'], a['sampler']),
              'random_state': equal(c['rng'], a['rng']),
              'augmentation_state': equal(c['augment_rng'], a['augment_rng']),
              'scheduler_state': equal(c['scheduler'], a['scheduler']),
              'amp_state': equal(c['amp'], a['amp']),
              'torch_version': c['torch_version'] == a['torch_version'],
              'finite_model_control': all(torch.isfinite(t).all().item() for t in c['model'].values()),
              'finite_model_auxiliary': all(torch.isfinite(t).all().item() for t in a['model'].values())}
    result = {'control': str(args.control), 'auxiliary': str(args.auxiliary),
              'control_sha256': sha256_file(args.control), 'auxiliary_sha256': sha256_file(args.auxiliary),
              'step': c['step'], 'treatment': args.treatment, 'weight': args.weight,
              'checks': checks, 'mechanics_pass': all(checks.values()),
              'limitation': 'CUDA kernels are not configured deterministic; matching RNG and sampler states do not prove bitwise numerical repeatability.',
              'release_authorized': False}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
