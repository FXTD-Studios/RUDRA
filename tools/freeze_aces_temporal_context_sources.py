"""Freeze scene/content-disjoint CC0 sources before new context/temporal training."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha


def main():
    regpath=Path('configs/experiments/aces_temporal_context_sources_20261024.json');reg=json.loads(regpath.read_text())
    out=Path('G:/datasets/corpora/aces_temporal_context_sources_20261024');out.mkdir(exist_ok=False)
    (out/'selector_snapshot.py').write_bytes(Path(__file__).read_bytes())
    excluded_rows=[]
    for key in ('ancestry','development'):excluded_rows.extend(json.loads(Path(reg[key]).read_text())['rows'])
    excluded_scenes={r['scene_id'] for r in excluded_rows}
    excluded_hashes={r.get('source_cc0_parent_sha256',r.get('origin_sha256')) for r in excluded_rows}-{None}
    sources=[json.loads(line) for line in Path(reg['source_manifest']).read_text().splitlines()]
    scenes={}
    for row in sorted(sources,key=lambda r:r['asset_id']):
        if row['commercial_ok'] and row['licence_licence']=='CC0' and row['scene_id'] not in excluded_scenes:
            scenes.setdefault(row['scene_id'],row)
    ordered=sorted(scenes,key=lambda s:hashlib.sha256((reg['selection_salt']+s).encode()).hexdigest())
    selected=[];seen=set(excluded_hashes)
    for split,count in reg['scenes'].items():
        candidates=[scenes[s] for s in ordered if scenes[s]['split']==split];picked=[]
        for source in candidates:
            identity=sha(source['hdr_path'])
            if identity in seen:continue
            seen.add(identity);picked.append({**source,'origin_sha256':identity})
            if len(picked)==count:break
        if len(picked)!=count:raise ValueError('insufficient disjoint CC0 sources')
        selected.extend(picked)
    result={'rows':selected,'registration_sha256':sha(regpath),'source_manifest_sha256':sha(reg['source_manifest']),
        'excluded_ancestry_sha256':sha(reg['ancestry']),'excluded_development_sha256':sha(reg['development']),
        'excluded_scenes':sorted(excluded_scenes),'selector_sha256':sha(__file__),
        'used_for_optimization':False,'scope':reg['scope'],'release_authorized':False}
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'counts':{k:sum(r['split']==k for r in selected) for k in reg['scenes']},'excluded_scenes':len(excluded_scenes),'used_for_optimization':False},indent=2))


if __name__=='__main__':main()
