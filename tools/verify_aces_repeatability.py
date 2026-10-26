"""Exact checkpoint/model/optimizer/RNG comparison for repeated training runs."""
import argparse,json
from pathlib import Path
import torch


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('first','second','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    first=torch.load(a.first,map_location='cpu',weights_only=True)
    second=torch.load(a.second,map_location='cpu',weights_only=True)
    checks={key:key in second and equal(first[key],second[key]) for key in first}
    if first.keys()!=second.keys():checks['matching_keys']=False
    result={'checks':checks,'bit_exact':all(checks.values()),
            'scope':'same machine/software/data/recipe repeatability; not portability or quality proof'}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['bit_exact']:raise SystemExit(1)


if __name__=='__main__':main()
