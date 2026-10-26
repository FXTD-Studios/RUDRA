"""Summarize the frozen local/coarse comparison without selecting a checkpoint."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-spatial', action='store_true')
    args = parser.parse_args()
    root = Path('G:/rudra_training')
    reports = {}
    for mode in (('local', 'coarse', 'spatial') if args.include_spatial else ('local', 'coarse')):
        path = root / ('aces_spatial600_20261024' if mode == 'spatial' else f'aces_wide_{mode}600_20261024')
        read = lambda name: json.loads((path / name).read_text())
        train = read('report.json')
        paired = read('paired_development/acceptance.json')
        grade = read('old_detail_grade.json')
        video = read('hunyuan48.json')
        reports[mode] = {
            'root': str(path), 'recipe': train['recipe'],
            'checkpoint_sha256': train['checkpoint_sha256'],
            'frozen_parent_bit_exact': train['frozen_parent_bit_exact'],
            'initial_loss': train['history'][0]['loss'],
            'paired_checks': paired['checks'],
            'H_error_by_dataset': paired['H_error_by_dataset'],
            'new_detail_gradient_rmse_ev': paired['new_detail_gradient_rmse_ev'],
            'new_aligned_residual_mse': paired['new_aligned_residual_mse'],
            'grade_summary': grade['summary'], 'grade_checks': grade['checks'],
            'video_checks': video['checks'],
            'video_max_transition_mean_ev': video['max_transition_mean_ev'],
            'all_reported_gates_pass': all(paired['checks'].values()) and all(grade['checks'].values()) and all(video['checks'].values()),
        }
    keys = ('initial_checkpoint_sha256', 'initial_context_vector_sha256',
            'sample_schedule_sha256', 'steps', 'optimizer_updates', 'new_trainable_parameters')
    matching = {k: reports['local']['recipe'][k] == reports['coarse']['recipe'][k] for k in keys}
    matching['initial_loss'] = reports['local']['initial_loss'] == reports['coarse']['initial_loss']
    if args.include_spatial:
        matching.update({'spatial_'+k: reports['local']['recipe'][k] == reports['spatial']['recipe'][k] for k in keys})
        matching['spatial_initial_loss'] = reports['local']['initial_loss'] == reports['spatial']['initial_loss']
    assert all(matching.values()), matching
    assert all(r['frozen_parent_bit_exact'] for r in reports.values())
    out = root / ('aces_spatial_comparison_20261024' if args.include_spatial else 'aces_wide_comparison_20261024')
    out.mkdir(exist_ok=True)
    result = {'matched_controls': matching, 'arms': reports, 'release_authorized': False,
              'scope': 'Development comparison only. Equal parameters and updates, different context computation and resolution. No independent holdout or release qualification.'}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
