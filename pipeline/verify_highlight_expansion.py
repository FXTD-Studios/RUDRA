"""Review a prepared expansion and write hash-bound internal-pilot evidence.

No training is launched. Visual review is recorded separately, and a numerical
preflight is not a commercial release or calibrated HDR quality certification.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from pipeline import prepare_highlight_expansion as prep
from pipeline import commercial_preflight as commercial
import cv2
import numpy as np
from PIL import Image,ImageDraw


def feature(path):
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im is None:raise ValueError('Decode failed '+str(path))
    gray=cv2.cvtColor(cv2.resize(im,(32,32)),cv2.COLOR_BGR2GRAY).astype(np.float32)/65535
    d=cv2.dct(gray)[:8,:8].ravel()[1:]
    phash=sum(int(x>np.median(d))<<j for j,x in enumerate(d))
    flat=gray.ravel()-gray.mean()
    return phash,flat/max(float(np.linalg.norm(flat)),1e-9),float(gray.std())


def contact(data,path,columns=4):
    cellw,cellh=320,210
    canvas=Image.new('RGB',(columns*cellw,((len(data)+columns-1)//columns)*cellh),(22,22,22));draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(data):
        code=cv2.imread(r['sdr_path'],cv2.IMREAD_UNCHANGED)[...,::-1].astype(np.float32)/65535
        # Stored canonical SDR is sRGB encoded with Rec.2020 primaries. Preview in sRGB/709.
        linear=np.where(code<=.04045,code/12.92,((code+.055)/1.055)**2.4)
        rgb=np.maximum(linear@np.linalg.inv(prep.render.M_709_2020).T,0)
        rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
        image=Image.fromarray(np.uint8(np.rint(np.clip(rgb,0,1)*255))).resize((320,180))
        x,y=i%columns*cellw,i//columns*cellh;canvas.paste(image,(x,y));draw.text((x+3,y+181),r['asset_id'][:45],fill='white')
        draw.text((x+3,y+195),f"Peak {r['peak_nits']:.0f} nit | >203 nit {100*r['highlight_fraction_above_203_nits']:.1f}%",fill='white')
    canvas.save(path)


def main():
    cv2.setNumThreads(1);out=prep.OUT
    plan=json.loads((out/'plan.json').read_text())
    # Verify original files used for the batch, not just their receipt JSON.
    original=prep.rows(prep.PH/'files_manifest.jsonl')
    wanted={a['path'] for a in plan['panorama_sources']}
    for j in plan['cg_jobs']:
        wanted.update(j.get('material_maps',{}).values())
        if j.get('hdri_path'):wanted.add(j['hdri_path'])
        for p,h in zip(j.get('vdb_frames',[]),j.get('vdb_sha256',[])):
            if prep.sha(p)!=h:raise ValueError('Volume source changed '+p)
    for r in original:
        if r['path'] in wanted and prep.sha(r['path'])!=r['sha256']:raise ValueError('Original source changed '+r['path'])
    print('[sources] selected source hashes verified',flush=True)
    new=prep.rows(out/'sdr_hdr_manifest.jsonl');base=prep.rows(prep.BASE/'combined_manifest.jsonl');combined=prep.rows(out/'combined_candidate_manifest.jsonl')
    if combined[:len(base)]!=base or combined[len(base):]!=new:raise ValueError('Base or new rows changed')
    paired=defaultdict(list);sequences=defaultdict(list)
    for r in new:
        paired[(r['sequence_id'],r['frame_index'])].append(r)
        if r['variant']=='clean':sequences[r['sequence_id']].append(r)
    for key,variants in paired.items():
        if len(variants)!=2 or {r['variant'] for r in variants}!={'clean','hard'}:raise ValueError('Missing SDR variant')
        if len({r['hdr_sha256'] for r in variants})!=1:raise ValueError('Degradation changed HDR target')
        if len({r['sdr_sha256'] for r in variants})!=2:raise ValueError('Degradation produced identical SDR')
    for key,seq in sequences.items():
        expected=3 if seq[0]['source_id']=='polyhaven_highlight_pipeline' else 4
        if sorted(r['frame_index'] for r in seq)!=list(range(expected)):raise ValueError('Missing sequence frame')
        if len({r['hdr_sha256'] for r in seq})<2:raise ValueError('Static duplicate sequence')
    prep.dump(out/'variant_sequence_checks.json',dict(paired_variant_groups=len(paired),sequences=len(sequences),
        hdr_target_identical_between_sdr_variants=True,complete_short_sequences=True,distinct_frames=True,held_out_rows_unchanged=True))
    held=[r for r in base if r['split']!='train']
    flags=[]
    for kind in ['sdr','hdr']:
        hf=[feature(r[kind+'_path']) for r in held]
        # Only clean variants for HDR, because clean/hard targets are identical.
        test=new if kind=='sdr' else [r for r in new if r['variant']=='clean']
        for a in test:
            ha,fa,sa=feature(a[kind+'_path'])
            for b,(hb,fb,sb) in zip(held,hf):
                if min(sa,sb)>.01 and (ha^hb).bit_count()<=6 and float(np.dot(fa,fb))>=.98:
                    flags.append(dict(new=a['asset_id'],held_out=b['asset_id'],kind=kind))
        print('[similarity]',kind,'complete',flush=True)
    prep.dump(out/'similarity_review.json',dict(new_pairs=len(new),held_out_pairs=len(held),cross_split_flags=flags,
        limits='Whole-frame pHash plus normalized 32x32 correlation. Does not exhaustively detect crops, retimes or semantically related scenes. Existing base/base audit retained.'))
    if flags:raise ValueError('Cross-split similarity needs review')
    clean=[r for r in new if r['variant']=='clean']
    panoramas=[r for r in clean if r['source_id']=='polyhaven_highlight_pipeline' and r['frame_index']==1]
    cg=[r for r in clean if r['source_id']!='polyhaven_highlight_pipeline' and r['frame_index'] in [0,3]]
    contact(panoramas,out/'panorama_contact_sheet.jpg')
    contact(cg,out/'synthetic_contact_sheet.jpg')
    degraded=[r for r in new if r['variant']=='hard' and r['source_id']!='polyhaven_highlight_pipeline' and r['frame_index']==0]
    contact(degraded,out/'degraded_contact_sheet.jpg')

    # Retain existing source approvals verbatim, resolving evidence paths to their originals.
    old_review=json.loads((prep.BASE/'commercial_review.json').read_text())
    approvals={sid:dict(v,evidence_path=str((prep.BASE/v['evidence_path']).resolve())) for sid,v in old_review['sources'].items()}
    evidence_dir=out/'source_evidence';evidence_dir.mkdir(exist_ok=True)
    for label,source in [('polyhaven_license',prep.PH/'licence_evidence/asset_licence.html'),
        ('polyhaven_ai_faq',prep.PH/'licence_evidence/ai_permission_faq.html'),
        ('jangafx_free_vdb_license',prep.BJ/'licence_evidence/jangafx_licence.html')]:
        shutil.copy2(source,evidence_dir/(label+'.html'))
    for name in ['campfire','simple_fire']:
        shutil.copy2(prep.BJ/'unpacked'/name/'LICENSE.txt',evidence_dir/(name+'_LICENSE.txt'))
    dependencies={str(p):prep.sha(p) for p in evidence_dir.iterdir()}
    dependencies.update({str(p):prep.sha(p) for p in [out/'plan.json',prep.PH/'files_manifest.jsonl',prep.BJ/'sources.json',
        REPO/'pipeline/prepare_highlight_expansion.py',REPO/'pipeline/render_highlight_scenes.py',REPO/'pipeline/verify_highlight_expansion.py',REPO/'pipeline/render_pipeline_pairs.py']})
    evidence=dict(owner_request='Collect the missing coverage and prepare it with the newly collected data for the next training; 18 October 2026.',
        scope='Preparation and internal commercial-development candidate only; no model release or automatic training launch.',
        sources={'Poly Haven':'CC0; provider explicitly permits AI training','JangaFX':'Published free VDB sets CC0; embedded licences retained',
                 'Authored geometry':'Simple scenes procedurally authored for FXTD in the recorded renderer; no external geometry or downloaded scripts'},
        unused='Charge CC BY materials/effects remain source-only; not incorporated in this batch.',
        changes='Perspective projection, scene-relative exposure normalization, authored rendering, colour conversion, quantization and JPEG degradation.',
        rights_limits='Existing owned-footage scope retained. This is automated provenance review, not certification of client contracts, privacy rights or weight distribution.',
        dependencies=dependencies,release_authorized=False)
    prep.dump(out/'source_evidence.json',evidence)
    for sid in sorted({r['source_id'] for r in new}):
        approvals[sid]=dict(commercial_training=True,weight_distribution=False,evidence_path='source_evidence.json',evidence_sha256=prep.sha(out/'source_evidence.json'))
    manifest=out/'combined_candidate_manifest.jsonl'
    review=dict(schema_version=2,scope='internal-pilot',release_authorized=False,manifest_sha256=prep.sha(manifest),
        reviewed_at='2026-10-18',reviewed_by='Automated provenance, split and pixel review under owner preparation instruction',sources=approvals,
        decoder_dependencies={str(p):prep.sha(p) for p in commercial.decoder_dependencies(combined)},preparation_dependencies=dependencies)
    prep.dump(out/'commercial_review.json',review)
    result=commercial.validate(manifest,out/'commercial_review.json',progress=lambda s:print(s,flush=True))
    prep.dump(out/'preflight_result.json',result)
    # Exercise the real loader once per category/variant without CUDA or training.
    from training.sdr2hdr_dataset import SDRHDRDataset
    import torch
    torch.set_num_threads(1)
    dataset=SDRHDRDataset(out/'sdr_hdr_manifest.jsonl',split='train',crop_size=256,augment=False,degradation_probability=0)
    chosen={}
    for i,r in enumerate(dataset.records):chosen.setdefault((r['category'],r['variant']),i)
    checks=[]
    for (category,variant),i in chosen.items():
        sample=dataset[i]
        for key in ('sdr','hdr'):
            if not torch.isfinite(sample[key]).all():raise ValueError('Loader produced nonfinite '+key)
            if tuple(sample[key].shape)!=(3,256,256):raise ValueError('Unexpected loader crop')
        checks.append(dict(category=category,variant=variant,asset_id=dataset.records[i]['asset_id'],
            sdr_shape=list(sample['sdr'].shape),hdr_shape=list(sample['hdr'].shape),hdr_max_network_units=float(sample['hdr'].max())))
    prep.dump(out/'loader_smoke.json',dict(samples=checks,count=len(checks),device='CPU',training_performed=False))
    summary=json.loads((out/'preparation_summary.json').read_text())
    summary.update(stage='numeric_preflight_passed_visual_review_pending',perceptual_cross_split_flags=0,paired_files_verified=len(combined)*2,
        new_pair_fraction=len(new)/len(combined),new_train_fraction=len(new)/sum(r['split']=='train' for r in combined),
        next='Inspect rendered contact sheets, record limitations; no automatic training launch.')
    prep.dump(out/'preparation_summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
