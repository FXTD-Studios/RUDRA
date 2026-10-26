"""Registered multi-seed paired assessment of image-edge shadow supervision (local radius-8 branch)."""
import argparse,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_aces_shadow_edge import REPORTS,PAIRED_RECIPE_KEYS,candidate_checks
from tools.evaluate_run05_dequant import sha

DARK='original_object_motion_1'


def load(root):
    return {s:json.loads((root/p).read_text()) for s,p in REPORTS}


def dark_means(motion):
    return {c['variant']:c['mean_cell_excess_ev'] for c in motion['clips'] if c['scene_id']==DARK}


def spread(values):
    mean=statistics.fmean(values)
    return {'mean':mean,'stdev':statistics.stdev(values) if len(values)>1 else None,
            'cv':(statistics.stdev(values)/mean) if len(values)>1 and mean else None,'min':min(values),'max':max(values),'n':len(values)}


def assess(control_roots,edge_roots,baseline,min_wins):
    if len(control_roots)!=len(edge_roots) or len(control_roots)<3:raise ValueError('at least three matched seed pairs required')
    pairs=[];seen=set()
    for c_root,e_root in zip(control_roots,edge_roots):
        c=load(c_root);e=load(e_root);cr=c['training']['recipe'];er=e['training']['recipe']
        for key in PAIRED_RECIPE_KEYS:
            if cr[key]!=er[key]:raise ValueError(f'pair recipe mismatch {key}: {c_root} {e_root}')
        if cr.get('include_image_edges') or not er.get('include_image_edges'):raise ValueError('control must be unedged and treatment edged')
        if c['training']['visits']!=e['training']['visits']:raise ValueError('pair visits mismatch')
        if cr['seed'] in seen:raise ValueError('duplicate seed')
        seen.add(cr['seed'])
        rc=candidate_checks(*(c[s] for s,_ in REPORTS),baseline);re_=candidate_checks(*(e[s] for s,_ in REPORTS),baseline)
        pairs.append({'seed':cr['seed'],'control_root':str(c_root),'edge_root':str(e_root),
                      'control':{'natural_shadow_mse':rc['natural_shadow_mse'],'original_shadow_mse':rc['original_shadow_mse'],'dark_motion_mean_ev':dark_means(c['motion']),'checks':rc['checks'],'numerical_pass':rc['numerical_pass']},
                      'edge':{'natural_shadow_mse':re_['natural_shadow_mse'],'original_shadow_mse':re_['original_shadow_mse'],'dark_motion_mean_ev':dark_means(e['motion']),'checks':re_['checks'],'numerical_pass':re_['numerical_pass']}})
    arms={}
    for arm in ('control','edge'):
        arms[arm]={'natural_shadow_mse':spread([p[arm]['natural_shadow_mse'] for p in pairs]),
                   'original_shadow_mse':spread([p[arm]['original_shadow_mse'] for p in pairs]),
                   'dark_motion_mean_ev':{v:spread([p[arm]['dark_motion_mean_ev'][v] for p in pairs]) for v in pairs[0][arm]['dark_motion_mean_ev']},
                   'runs_passing_all_candidate_gates':sum(p[arm]['numerical_pass'] for p in pairs)}
    ratio=arms['edge']['natural_shadow_mse']['mean']/arms['control']['natural_shadow_mse']['mean']
    wins=sum(p['edge']['natural_shadow_mse']<p['control']['natural_shadow_mse'] for p in pairs)
    checks={'edge_mean_natural_shadow_5pct_below_control':ratio<=.95,
            f'edge_wins_natural_shadow_in_at_least_{min_wins}_of_{len(pairs)}_seeds':wins>=min_wins,
            'every_edge_run_passes_candidate_gates':all(p['edge']['numerical_pass'] for p in pairs),
            'edge_mean_original_shadow_no_regression':arms['edge']['original_shadow_mse']['mean']<=arms['control']['original_shadow_mse']['mean']}
    for v,s in arms['edge']['dark_motion_mean_ev'].items():
        checks['edge_mean_dark_motion_no_regression_'+v]=s['mean']<=arms['control']['dark_motion_mean_ev'][v]['mean']
    return {'pairs':pairs,'arms':arms,'edge_over_control_natural_shadow_ratio':ratio,'edge_wins':wins,
            'checks':checks,'numerical_pass':all(checks.values()),'release_authorized':False,
            'scope':'registered multi-seed paired shadow development study; development set has only 7 shadow-eligible natural frames from 3 scenes; no visual/detail/platform/native/release acceptance'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,nargs='+',required=True);p.add_argument('--edge',type=Path,nargs='+',required=True)
    for key in ('baseline','registration','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--min-wins',type=int,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    baseline_path=a.baseline/'development24/report.json'
    out=assess(a.control,a.edge,json.loads(baseline_path.read_text()),a.min_wins)
    out['registration']={'path':str(a.registration),'sha256':sha(a.registration)};out['baseline_report_sha256']=sha(baseline_path)
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({k:out[k] for k in ('arms','edge_over_control_natural_shadow_ratio','edge_wins','checks','numerical_pass')},indent=2))


if __name__=='__main__':main()
