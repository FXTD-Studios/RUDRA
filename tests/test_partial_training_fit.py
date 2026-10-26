from tools.audit_partial_training_fit import aggregate


def test_exposure_weighting_and_domain_separation():
    rows = [
        {'domain': 'still', 'pixels': 1, 'exposures': 3,
         'errors': {'model': dict.fromkeys(('photo', '-3', '0', '3'), 2)}},
        {'domain': 'motion', 'pixels': 9, 'exposures': 1,
         'errors': {'model': dict.fromkeys(('photo', '-3', '0', '3'), 10)}},
    ]
    result = aggregate(rows, 'model', 'all')
    assert result['pixel_exposure']['0'] == 8
    assert result['frame_exposure']['0'] == 4
    assert result['unique_crop']['0'] == 6
    assert aggregate(rows, 'model', 'still')['pixel_exposure']['0'] == 2
    assert aggregate(rows, 'model', 'motion')['frame_exposure']['0'] == 10
