"""Frozen full-schedule TRAIN fit; no optimization or development selection."""
import json
import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_partial_display import forward_partial
from tools.summarize_aces_generated import sha
from training.aces_model import model_from_checkpoint, reconstruction_weight
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def aggregate(records, state, domain):
    rows = [r for r in records if domain == 'all' or r['domain'] == domain]
    weights = {'pixel_exposure': [r['pixels']*r['exposures'] for r in rows],
               'frame_exposure': [r['exposures'] for r in rows],
               'unique_crop': [1 for r in rows]}
    return {kind: {term: float(np.average([r['errors'][state][term] for r in rows], weights=w))
                   for term in ('photo', '-3', '0', '3')} for kind, w in weights.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', help='Additional checkpoint on the same fixed 400-update supervision')
    parser.add_argument('--out', default='G:/rudra_training/partial_training_fit_20261025')
    args = parser.parse_args()
    base = Path('G:/rudra_training')
    root = base/'partial_temporal1_400_20261025'
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    checkpoints = {'initial': base/'aces_guarded_projected600_20261024/step600.pt',
                   'uncapped': base/'partial_pcgrad_deterministic_control400_20261025/step400.pt',
                   'capped': base/'partial_pcgrad_cap400_20261025/step400.pt',
                   'direct': base/'partial_pcgrad_direct400_20261025/step400.pt'}
    if args.candidate:
        checkpoints['candidate'] = Path(args.candidate)
    pilots = [Path('G:/datasets/corpora')/name/'train/report.json' for name in
              ('aces_consistent_exposures_v2_20261024', 'aces_temporal_context_pairs_20261024')]
    inputs = pilots + [root/n for n in ('crop_pool.json','motion_pool.json','schedule.json')]
    reg = {'inputs': {str(p): sha(p) for p in inputs},
           'checkpoints': {str(p): sha(p) for p in checkpoints.values()},
           'policy': 'All unique scheduled TRAIN crops, raw partial-route photo and display errors. Explicit pixel exposure, frame exposure and unique crop weighting. No temporal loss, final appearance constraint, updates or DEV selection.',
           'release_authorized': False}
    (out/'registration.json').write_text(json.dumps(reg, indent=2))
    (out/'auditor.py').write_bytes(Path(__file__).read_bytes())
    rows = {}; domains = {}
    for p, domain in zip(pilots, ('still','motion')):
        for r in json.loads(p.read_text())['rows']:
            assert r['asset_id'] not in rows
            rows[r['asset_id']] = r; domains[r['asset_id']] = domain
    pools = [json.loads((root/n).read_text()) for n in ('crop_pool.json','motion_pool.json')]
    schedule = json.loads((root/'schedule.json').read_text()); requests = Counter()
    for i, indices in enumerate(schedule):
        for index in indices:
            crop = pools[int(i%2 == 0)][index]
            for asset in crop['assets'] if i%2 == 0 else [crop['asset_id']]:
                requests[(asset, *crop['origin'])] += 1
    assert len(requests) == 622 and sum(requests.values()) == 1600
    torch.set_num_threads(2); view = reference_view(); view.set_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    records = []; samples = []; cache = None
    for (asset,y,x), frequency in sorted(requests.items()):
        row = rows[asset]; assert row['split'] == 'train'
        if cache != asset:
            for key in ('sdr','target'):
                assert sha(row[key+'_path']) == row[key+'_sha256']
            source = canonical_to_source(load_rgb(row['sdr_path'], hdr=False))
            target = np.load(row['target_path']); cache = asset
        s = source[y:y+192,x:x+192].copy(); t = target[y+64:y+128,x+64:x+128].copy()
        interior = s[64:-64,64:-64]
        weight = reconstruction_weight(torch.from_numpy(interior).permute(2,0,1)[None])[0,0].numpy()
        mask = (weight>0)&(interior.max(-1)>.5)&(interior.min(-1)<254.5/255)
        assert mask.any()
        records.append({'asset_id':asset,'origin':[y,x],'domain':domains[asset],
                        'exposures':frequency,'pixels':int(mask.sum()),'errors':{}})
        samples.append((s,t,mask))
    assert sum(r['pixels']*r['exposures'] for r in records) == 982607
    def errors(pred, target, mask):
        result = {'photo':float(np.square(np.arcsinh(pred.astype(np.float64))-np.arcsinh(target.astype(np.float64))).mean(-1)[mask].mean())}
        for ev in (-3,0,3):
            delta = (view.forward(pred*2.**ev)-view.forward(target*2.**ev)).astype(np.float64)*255
            result[str(ev)] = float(np.square(delta).mean(-1)[mask].mean())
        return result
    for r,(s,t,m) in zip(records,samples):
        r['errors']['analytic'] = errors(view.inverse(s[64:-64,64:-64]),t,m)
    for state,path in checkpoints.items():
        model = model_from_checkpoint(torch.load(path,map_location='cpu',weights_only=True)).eval().cuda()
        original = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            for i,(r,(s,t,m)) in enumerate(zip(records,samples)):
                tensor = torch.from_numpy(s).permute(2,0,1)[None].cuda()
                pred, mask, _ = forward_partial(model,tensor)
                assert np.array_equal(mask[0,0,64:-64,64:-64].cpu().numpy().astype(bool),m)
                pred = pred[0,:,64:-64,64:-64].permute(1,2,0).cpu().numpy()
                r['errors'][state] = errors(pred,t,m)
                if i%100 == 0: print(state,i,flush=True)
        assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
        del model, original
        (out/'partial.json').write_text(json.dumps(records,allow_nan=False))
    summary = {state:{domain:aggregate(records,state,domain) for domain in ('all','still','motion')}
               for state in ('analytic',*checkpoints)}
    calibration = json.loads((base/'partial_display_fixed_scale_20261025/report.json').read_text())
    for ev,value in calibration['denominators'].items():
        assert np.isclose(summary['analytic']['all']['pixel_exposure'][ev],value,rtol=1e-10,atol=1e-10)
    report = {'summary':summary,'rows':records,'unique_crops':len(records),'frame_exposures':1600,
              'pixel_exposures':982607,'weights_unchanged':True,'analytic_calibration_matches':True,
              'registration_sha256':sha(out/'registration.json'),'auditor_sha256':sha(__file__)}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
