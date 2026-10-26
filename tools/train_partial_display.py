"""Fine-tune ONLY the partial-highlight route (v8 highlight_context) of a v9 model.

Loss on partially clipped highlight pixels (reconstruction_weight>0, max>.5,
min<254.5/255), crop interior only:
  photo   = mean asinh-RGB MSE vs target (the H metric)
  display = mean squared 0 EV ACES display error vs target, in 8-bit code units^2
            (exact OCIO forward, finite-difference gradient)
total = photo + display_weight * display.
Everything else (restorer, shadow, full_context, frozen parent fields) is fixed;
fully clipped and non-highlight outputs are unchanged by construction.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import load_model
from training.aces_reference import reference_view
from training.aces_full_context_model import fully_clipped_support
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from training.aces_model import reconstruction_weight
from training.aces_partial_geometry import aligned_partial_origin

D = os.path.expanduser('~/data/')
PILOTS = ['datasets/corpora/aces_consistent_exposures_v2_20261024/train/report.json',
          'datasets/corpora/aces_temporal_context_pairs_20261024/train/report.json']


def loc(p):
    return D + p.replace('\\', '/').split('G:/')[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_fixed_display_scale(path,evs):
    values={float(k):float(v) for k,v in json.loads(Path(path).read_text())['denominators'].items()}
    if any(ev not in values or not np.isfinite(values[ev]) or values[ev]<1e-4 for ev in evs):
        raise ValueError('finite registered display scales >=1e-4 required for every EV')
    return values


class _Forward(torch.autograd.Function):
    """Exact ACES view forward on BCHW ACEScg; finite-difference backward."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        a = x.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return torch.from_numpy(np.ascontiguousarray(reference_view().forward(a))).permute(0, 3, 1, 2).to(x.device)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        a = x.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
        up = g.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        grad = np.zeros_like(a)
        for c in range(3):
            h = np.maximum(np.abs(a[..., c]) * 1e-3, 1e-6).astype(np.float32)
            p, m = a.copy(), a.copy()
            p[..., c] += h
            m[..., c] -= h
            jac = (reference_view().forward(p) - reference_view().forward(m)) / (2 * h[..., None])
            grad[..., c] = np.sum(jac * up, -1)
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(grad).permute(0, 3, 1, 2).to(x.device)


def build_pool(crops_per_frame, crop, seed, aligned=False, border=48, matched=False, metadata=None, pilots=None):
    rng = np.random.default_rng(seed)
    pool, used = [], []
    for pil in (PILOTS if pilots is None else pilots):
        rows = json.loads(Path(D + pil).read_text())['rows']
        for r in rows:
            assert r['split'] == 'train'
            for key in ('sdr','target'):
                if sha(loc(r[key+'_path']))!=r[key+'_sha256']:raise ValueError('training content identity')
            src = canonical_to_source(load_rgb(loc(r['sdr_path']), hdr=False))
            tgt = np.load(loc(r['target_path'])).astype(np.float32)
            w = reconstruction_weight(torch.from_numpy(src).permute(2, 0, 1)[None])[0, 0].numpy()
            part = (w > 0) & (src.max(-1) > .5) & (src.min(-1) < 254.5 / 255)
            h, wd = src.shape[:2]
            ys, xs = np.nonzero(part[crop // 4:h - crop // 4, crop // 4:wd - crop // 4])
            if len(ys) < 64 or h < crop or wd < crop:
                continue
            count_before=len(pool)
            for k in rng.choice(len(ys), size=min(crops_per_frame, len(ys)), replace=False):
                cy, cx = ys[k] + crop // 4, xs[k] + crop // 4
                y0 = int(np.clip(cy - crop // 2, 0, h - crop)); x0 = int(np.clip(cx - crop // 2, 0, wd - crop))
                if matched:
                    ya,xa=aligned_partial_origin(cy,cx,h,wd,crop,border)
                    if any(not part[yy+border:yy+crop-border,xx+border:xx+crop-border].any() for yy,xx in ((y0,x0),(ya,xa))):continue
                if aligned:
                    y0,x0=aligned_partial_origin(cy,cx,h,wd,crop,border)
                    if not part[y0+border:y0+crop-border,x0+border:x0+crop-border].any():
                        continue
                pool.append((src[y0:y0 + crop, x0:x0 + crop].copy(), tgt[y0:y0 + crop, x0:x0 + crop].copy()))
                if metadata is not None:metadata.append({'asset_id':r['asset_id'],'centre':[int(cy),int(cx)],'origin':[y0,x0]})
            if len(pool)>count_before:used.append(r['asset_id'])
    return pool, used


def forward_partial(model, x):
    with torch.no_grad():
        base = super(type(model).__mro__[1], model).forward(x)  # ACESModel.forward (frozen v7 path)
        full_fields = model.full_context(x, base.baseline)[:, 3:]
    w = base.reconstruction_weight
    fields = model.highlight_context(x, base.baseline)[:, 3:]
    rest = base.restored_acescg
    scale = rest.abs().amax(1, keepdim=True) + .001
    partial_pred = rest + w * (scale * fields)
    full_pred = rest + w * (scale * full_fields)
    high = (w > 0) & (x.amax(1, keepdim=True) > .5)
    full = fully_clipped_support(x, w)
    out = torch.where(full, full_pred, torch.where(high, partial_pred, base.acescg))
    return out, high & ~full, base.baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--display-weight', type=float, default=1.0)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=192)
    ap.add_argument('--border', type=int, default=48)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--crops-per-frame', type=int, default=4)
    ap.add_argument('--display-evs', type=float, nargs='+', default=[0.0])
    ap.add_argument('--relative', action='store_true', help='normalise each EV display error by the analytic inverse error on the same batch')
    ap.add_argument('--aligned-crops', action='store_true', help='phase-align crop origins; requires --border 64 or larger and dimensions divisible by4')
    ap.add_argument('--matched-crops', action='store_true', help='same eligible source centres for legacy/aligned comparison')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--data-root',default=D)
    ap.add_argument('--paired-motion',action='store_true',help='alternate real adjacent motion pairs and still replay; requires aligned crops')
    ap.add_argument('--temporal-weight',type=float,default=0.0)
    ap.add_argument('--fixed-display-scale',help='TRAIN-calibrated denominator report; requires --relative')
    ap.add_argument('--registered-schedule',help='directory with hash-bound paired schedule and manifest')
    ap.add_argument('--pcgrad',action='store_true',help='project conflicting weighted objective gradients before existing clipping/Adam')
    ap.add_argument('--pcgrad-probe-steps',nargs='+',type=int,default=[],help='diagnostics only at specified one-based updates; save state and restore counterfactuals')
    ap.add_argument('--deterministic',action='store_true',help='require deterministic torch algorithms and CUDA workspace for reproducible diagnostics')
    ap.add_argument('--max-update-l2',type=float,help='opt-in cap on applied Adam parameter step; requires PCGrad; moments unchanged')
    ap.add_argument('--display-ev-weights',type=float,nargs='+',help='per-EV multipliers for the display terms, same order as --display-evs (default all 1)')
    ap.add_argument('--direct-step-report',help='replace Adam with normalized PCGrad using capped TRAIN report step lengths')
    a = ap.parse_args()
    if a.direct_step_report and (not a.pcgrad or not a.deterministic or a.max_update_l2 is not None or a.pcgrad_probe_steps):
        raise ValueError('direct projected steps require deterministic PCGrad and exclude Adam cap/probes')
    if a.max_update_l2 is not None and (not a.pcgrad or not np.isfinite(a.max_update_l2) or a.max_update_l2<=0):
        raise ValueError('positive finite cap and PCGrad required')
    if a.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    if a.pcgrad_probe_steps and (not a.pcgrad or a.crop!=192 or a.border!=64 or a.batch!=4 or not a.relative or a.display_evs!=[-3,0,3] or any(s<1 or s>a.steps for s in a.pcgrad_probe_steps)):
        raise ValueError('step probes require registered PCGrad geometry/EV recipe and valid steps')
    ev_weights=a.display_ev_weights or [1.0]*len(a.display_evs)
    if len(ev_weights)!=len(a.display_evs) or any(not np.isfinite(v) or v<=0 for v in ev_weights):raise ValueError('one positive finite weight per display EV')
    fixed_scale=load_fixed_display_scale(a.fixed_display_scale,a.display_evs) if a.fixed_display_scale else None
    if fixed_scale and not a.relative:raise ValueError('fixed display scales require relative loss')
    globals()['D']=a.data_root.rstrip('/\\')+'/'
    if a.temporal_weight<0 or (a.temporal_weight and not a.paired_motion):raise ValueError('temporal objective requires paired motion')
    if a.paired_motion and (not a.aligned_crops or a.batch%2):raise ValueError('aligned crops and even frame batch required')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())  # exact code, for provenance
    (out / 'geometry_snapshot.py').write_bytes(Path('training/aces_partial_geometry.py').read_bytes())
    if fixed_scale:(out/'fixed_display_scale.json').write_bytes(Path(a.fixed_display_scale).read_bytes())
    torch.manual_seed(a.seed); torch.set_num_threads(2);reference_view().set_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model, state = load_model(a.checkpoint, a.device)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith('highlight_context.'))
    t0 = time.time()
    metadata=[]
    pool, used = build_pool(a.crops_per_frame, a.crop, a.seed,a.aligned_crops,a.border,a.matched_crops,metadata,PILOTS[:1] if a.paired_motion else None)
    motion_pool=[]
    if a.paired_motion:
        from training.aces_partial_pairs import build_motion_pool
        motion_pool,motion_metadata=build_motion_pool(D+PILOTS[1],loc,sha,a.crop,a.border,a.seed,a.crops_per_frame)
        if len(motion_pool)<a.batch//2:raise ValueError('insufficient motion pairs')
        (out/'motion_pool.json').write_text(json.dumps(motion_metadata,indent=2))
        used=sorted(set(used)|{asset for row in motion_metadata for asset in row['assets']})
    (out/'crop_pool.json').write_text(json.dumps(metadata,indent=2))
    if len(pool)<a.batch:raise ValueError('insufficient eligible crops')
    print('pool', len(pool), 'frames', len(used), 'prep_s', round(time.time() - t0), flush=True)
    opt = None if a.direct_step_report else torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    rng = np.random.default_rng(a.seed + 1)
    schedule=[rng.choice(len(motion_pool) if a.paired_motion and step%2==0 else len(pool),a.batch//2 if a.paired_motion and step%2==0 else a.batch,replace=False).tolist() for step in range(a.steps)]
    if a.registered_schedule:
        from training.aces_partial_schedule import load_registered_schedule
        schedule=load_registered_schedule(a.registered_schedule,out,steps=a.steps,batch=a.batch,crop=a.crop,border=a.border,paired_motion=a.paired_motion)
        (out/'schedule_manifest.json').write_bytes((Path(a.registered_schedule)/'manifest.json').read_bytes())
    (out/'schedule.json').write_text(json.dumps(schedule))
    movement_lengths=None
    if a.direct_step_report:
        from training.aces_direct_projected import load_movement_schedule
        movement_lengths=load_movement_schedule(a.direct_step_report,out,vars(a))
        (out/'direct_step_source.json').write_bytes(Path(a.direct_step_report).read_bytes())
    b = a.border
    hist = []
    ema = {}
    projection_rng=np.random.default_rng(a.seed+2)
    trainable=[p for p in model.parameters() if p.requires_grad]
    for step in range(1, a.steps + 1):
        idx = schedule[step-1]
        is_motion=a.paired_motion and step%2==1
        selected=motion_pool if is_motion else pool;combine=np.concatenate if is_motion else np.stack
        x = torch.from_numpy(combine([selected[i][0] for i in idx])).permute(0, 3, 1, 2).contiguous().to(a.device)
        t = torch.from_numpy(combine([selected[i][1] for i in idx])).permute(0, 3, 1, 2).contiguous().to(a.device)
        full_target=t
        pred, mask, ana = forward_partial(model, x)
        temporal=pred.new_zeros(());active_count=0
        if is_motion:
            from training.aces_active_temporal import active_temporal_loss
            high=(reconstruction_weight(x)>0)&(x.amax(1,keepdim=True)>.5)
            terms=[]
            for pair in range(0,a.batch,2):
                value,count,_=active_temporal_loss(pred[pair:pair+2],t[pair:pair+2],high[pair:pair+2],mask[pair:pair+2],border=b)
                if not count:raise ValueError('empty temporal support')
                terms.append(value);active_count+=count
            temporal=torch.stack(terms).mean()
        pred, t, mask, ana = pred[..., b:-b, b:-b], t[..., b:-b, b:-b], mask[..., b:-b, b:-b].float(), ana[..., b:-b, b:-b]
        n = mask.sum().clamp_min(1)
        photo = (((torch.asinh(pred) - torch.asinh(t)) ** 2).mean(1, keepdim=True) * mask).sum() / n
        loss = photo
        objectives={'photo':photo}
        disp = torch.zeros((),device=a.device)
        if a.display_weight > 0:
            terms = []
            for ev,ev_weight in zip(a.display_evs,ev_weights):
                k = 2.0 ** ev
                with torch.no_grad():
                    td = _Forward.apply(t * k)
                    ref = torch.ones((),device=a.device)
                    if a.relative:
                        r = float(((((_Forward.apply(ana * k) - td) * 255) ** 2).mean(1, keepdim=True) * mask).sum() / n)
                        ema[ev] = fixed_scale[ev] if fixed_scale is not None else (r if ev not in ema else .95 * ema[ev] + .05 * r)
                        ref = torch.tensor(max(ema[ev], 1e-4),device=a.device)
                pd = _Forward.apply(pred * k)
                e = ((((pd - td) * 255) ** 2).mean(1, keepdim=True) * mask).sum() / n
                terms.append(ev_weight * e / ref.clamp_min(1e-6))
                objectives[f'display_ev_{ev:g}']=a.display_weight/len(a.display_evs)*terms[-1]
            disp = torch.stack(terms).mean()
            loss = photo + a.display_weight * disp
        loss=loss+a.temporal_weight*temporal
        if is_motion and a.temporal_weight:objectives['temporal']=a.temporal_weight*temporal
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        model.zero_grad(set_to_none=True) if opt is None else opt.zero_grad()
        projection_info=None
        if a.pcgrad:
            from training.aces_pcgrad import project_conflicting
            task_gradients=[]
            for objective in objectives.values():
                grads=torch.autograd.grad(objective,trainable,retain_graph=True,allow_unused=True)
                task_gradients.append(torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,p in zip(grads,trainable)]))
            task_gradients=torch.stack(task_gradients)
            projected,projection_info=project_conflicting(task_gradients,projection_rng)
            offset=0
            for p in trainable:
                p.grad=projected[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
            previous=torch.cat([p.detach().flatten() for p in trainable]).clone()
            projection_info['task_names']=list(objectives)
            projection_info['weighted_gradient_norms']=task_gradients.norm(dim=1).tolist()
            projection_info['unit_projected_descent_derivatives']=(-(task_gradients.double()@projected.double())/projected.double().norm().clamp_min(1e-30)).tolist()
        else:loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite gradient')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if step in a.pcgrad_probe_steps:
            snapshot=out/f'step_probe_{step:04d}.pt'
            torch.save({'checkpoint':{**state,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},
                        'optimizer':opt.state_dict(),'clipped_gradients':[p.grad.detach().cpu().clone() for p in trainable],
                        'projected':projected.detach().cpu(),'source':x.detach().cpu(),'target':full_target.detach().cpu(),
                        'step':step,'motion':is_motion,'crop_indices':idx,'ema':dict(ema),
                        'projection_rng_state':projection_rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all() if a.device=='cuda' else [],'recipe':vars(a)},snapshot)
        if movement_lengths is not None:
            from training.aces_direct_projected import direct_projected_step
            projection_info['direct_step']=direct_projected_step(trainable,previous,projected,movement_lengths[step-1])
        else:opt.step()
        if a.max_update_l2 is not None:
            from training.aces_update_cap import cap_update
            projection_info['update_cap']=cap_update(trainable,previous,a.max_update_l2)
        if a.pcgrad:
            delta=torch.cat([p.detach().flatten() for p in trainable])-previous
            projection_info['actual_projected_step_l2' if movement_lengths is not None else 'actual_adam_step_l2']=float(delta.double().norm())
            projection_info['actual_step_first_order_changes']=(task_gradients.double()@delta.double()).tolist()
            if step in a.pcgrad_probe_steps:
                from training.aces_step_probe import measure_step
                from tools.audit_partial_objective_conflict import objective_terms
                def evaluate_probe():
                    terms,_=objective_terms(model,x,full_target,is_motion)
                    values={'photo':float(terms['photo'])}
                    for ev in a.display_evs:
                        values[f'display_ev_{ev:g}']=float(terms[f'ev{int(ev)}']/max(ema[ev],1e-4)*(a.display_weight/len(a.display_evs)))
                    if is_motion:values['temporal']=float(terms['temporal']*a.temporal_weight)
                    values['weighted_total']=sum(values.values())
                    return values
                probe=measure_step(trainable,previous,projected,evaluate_probe)
                probe.update(step=step,domain='motion' if is_motion else 'still',snapshot_sha256=sha(snapshot))
                (out/f'step_probe_{step:04d}.json').write_text(json.dumps(probe,indent=2,allow_nan=False))
        hist.append({'step': step, 'photo': float(photo.detach()), 'display': float(disp.detach()), 'pixels': int(n), 'analytic_ema': dict(ema),'domain':'motion' if is_motion else 'still','temporal':float(temporal.detach()),'active_pixels':active_count})
        if projection_info is not None:hist[-1]['pcgrad']=projection_info
        if step % 25 == 0:
            r = hist[-25:]
            print(step, 'photo', round(np.mean([h['photo'] for h in r]), 5), 'display', round(np.mean([h['display'] for h in r]), 5),
                  round(time.time() - t0), flush=True)
    after = model.state_dict()
    frozen_ok = all(torch.equal(before[k], after[k]) for k in before if not k.startswith('highlight_context.'))
    if not frozen_ok or any(not torch.isfinite(v).all() for v in after.values()):raise ValueError('invalid final state')
    repo=Path(__file__).resolve().parents[1];sources={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.suffix!='.py' or not path.is_relative_to(repo):continue
        rel=path.relative_to(repo)
        if rel.parts[0] not in ('training','tools','rudra','src'):continue
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());sources[str(rel)]=sha(path)
    (out/'source_hashes.json').write_text(json.dumps(sources,indent=2))
    ck = out / f'step{a.steps}.pt'
    torch.save({**state, 'model': after,
                'recipe': {**vars(a), 'parent_checkpoint': a.checkpoint, 'parent_sha256': sha(a.checkpoint), 'trainer_sha256': sha(out / 'trainer_snapshot.py'), 'geometry_sha256':sha(out / 'geometry_snapshot.py'),
                           'trained': 'highlight_context only (partial highlights)', 'train_frames': used,
                           'pilots': PILOTS,'pilot_hashes':{pil:sha(D+pil) for pil in PILOTS},'source_hashes':sources,
                           'schedule_sha256':sha(out/'schedule.json'),'crop_pool_sha256':sha(out/'crop_pool.json'),
                           'schedule_manifest_sha256':sha(out/'schedule_manifest.json') if a.registered_schedule else None,
                           'direct_step_source_sha256':sha(out/'direct_step_source.json') if a.direct_step_report else None,
                           'motion_pool_sha256':sha(out/'motion_pool.json') if a.paired_motion else None,
                           'fixed_display_scale_sha256':sha(out/'fixed_display_scale.json') if fixed_scale else None}}, ck)
    (out / 'report.json').write_text(json.dumps({'frozen_parent_bit_exact': frozen_ok, 'checkpoint_sha256': sha(ck),
                                                 'recipe': {**vars(a)}, 'history': hist, 'train_frames': len(used),
                                                 'pool': len(pool)}, indent=1))
    print('done frozen_ok', frozen_ok, flush=True)


if __name__ == '__main__':
    main()
