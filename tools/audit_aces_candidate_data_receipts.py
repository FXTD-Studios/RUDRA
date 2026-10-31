"""Check declared candidate data ancestry against retained local rights receipts."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('G:/rudra_training/aces_consistent_head_equal600_v2_20261024'))
    p.add_argument('--checkpoint',default='step600.pt');p.add_argument('--out',type=Path)
    args=p.parse_args();root=args.root
    out=args.out or root/'declared_data_receipt_audit_v2.json'
    if out.exists():raise ValueError('new output required')
    ancestry=root/'training_ancestry.json';state=torch.load(root/args.checkpoint,map_location='cpu',weights_only=True)
    if state['pilot_sha256']!=sha(ancestry):raise ValueError('checkpoint ancestry identity')
    rows=json.loads(ancestry.read_text())['rows'];sourcepath=Path('G:/datasets/corpora/rudra_run05_sources_20261022/sources.jsonl')
    sources=[json.loads(line) for line in sourcepath.read_text().splitlines()]
    byasset={};actual_hashes={}
    for source in sources:byasset.setdefault(source['asset_id'],[]).append(source)
    checks=[];counts={};unresolved=[]
    for r in rows:
        if r['split']!='train':raise ValueError('non-TRAIN ancestry')
        content=all(sha(r[k+'_path'])==r[k+'_sha256'] for k in ('sdr','target'))
        source_hash=r.get('source_cc0_parent_sha256',r.get('origin_sha256'))
        if source_hash:
            asset=r.get('parent_asset_id',r['asset_id']).split('__r05')[0]
            matches=byasset.get(asset,[])
            for m in matches:
                if m['hdr_path'] not in actual_hashes:actual_hashes[m['hdr_path']]=sha(m['hdr_path'])
            accepted=bool(matches) and all(m['commercial_ok'] and m['licence_licence']=='CC0' and
                m['scene_id']==r['scene_id'] and actual_hashes[m['hdr_path']]==source_hash for m in matches)
            kind='CC0 receipt match'
        else:
            accepted=r.get('commercial_ok') is True and r.get('provenance')=='original procedural generation, no third-party media; no public licence dedication'
            kind='original procedural declaration'
        counts[kind]=counts.get(kind,0)+1
        if not content or not accepted:unresolved.append(r['asset_id'])
        checks.append({'asset_id':r['asset_id'],'content_hashes_match':content,'receipt_match':accepted,'kind':kind,'source_hash':source_hash})
    result={'checkpoint_sha256':sha(root/args.checkpoint),'ancestry_sha256':sha(ancestry),'source_receipts_sha256':sha(sourcepath),'auditor_sha256':sha(__file__),
            'rows':checks,'counts':counts,'unresolved':unresolved,'declared_data_checks_pass':not unresolved,
            'receipt_rows_without_hdr_hash':sum(not r.get('hdr_sha256') for r in sources),
            'actual_origin_hashes':actual_hashes,
            'supersedes':'Initial audit only looked for precomputed hashes absent in receipt records. This audit resolves exact asset/scene identity and hashes each original file against declared ancestry.',
            'scope':'Declared data ancestry only; relies on retained receipt records. Not a complete base-checkpoint initialization-chain, dependency, distribution-rights or release audit.',
            'release_authorized':False}
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('counts','unresolved','declared_data_checks_pass')},indent=2))


if __name__=='__main__':main()
