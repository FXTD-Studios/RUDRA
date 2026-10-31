"""Inventory defective publisher EXRs without repairing or admitting them."""
import json
from pathlib import Path
from pipeline.review_next_hdr_sources import ROOT, read_exr, np
from pipeline.collect_next_hdr_sources import dump, sha

out = ROOT / 'tears_of_steel'
manifest = out/'initial_sequence_manifest.json'
if not manifest.exists():
    manifest = out/'sequence_manifest.json'
rows = json.loads(manifest.read_text())
records = []
for i, row in enumerate(rows):
    p = Path(row['path'])
    a, chroma = read_exr(p)
    invalid = ~np.isfinite(a)
    records.append(dict(path=str(p),scene=row['scene'],shape=list(a.shape),
        frame=int(p.stem.rsplit('_',1)[-1]),invalid_channels=int(invalid.sum()),
        invalid_pixels=int(invalid.any(axis=-1).sum()),nan_channels=int(np.isnan(a).sum()),
        infinite_channels=int(np.isinf(a).sum()),
        receipt_hash_matches=sha(p)==row['receipt']['sha256']))
    if (i+1)%24 == 0: print('[audit]',i+1,flush=True)
dump(out/'source_pixel_audit.json',dict(frames=records,training_enabled=False))
print(json.dumps([r for r in records if r['invalid_pixels'] or not r['receipt_hash_matches']],indent=2))
