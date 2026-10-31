"""Decode and hash every run05 pair, and produce source previews for visual QA."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.sdr2hdr_dataset import load_rgb
from training.contract_checks import canonical_to_source


def audit(manifest, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    scenes, digests, samples = {}, {}, []
    for row in rows:
        scene, split = row['scene_id'], row['split']
        if scenes.setdefault(scene, split) != split:
            raise ValueError('scene overlaps splits')
        for kind in ('sdr', 'hdr', 'mask'):
            p = Path(row[kind + '_path'])
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest != row[kind + '_sha256']:
                raise ValueError(f'hash mismatch: {p}')
            key = (kind, digest)
            if kind != 'mask' and digests.setdefault(key, split) != split:
                raise ValueError(f'identical {kind} media across splits: {p}')
        sdr, hdr = load_rgb(row['sdr_path'], False), load_rgb(row['hdr_path'], True, ceiling=1e6)
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        if (sdr.shape != hdr.shape or not np.isfinite(hdr).all()
                or not np.isfinite(sdr).all() or (hdr < 0).any()
                or (sdr < 0).any() or (sdr > 1).any()
                or mask is None or mask.shape != sdr.shape[:2] or mask.dtype != np.uint8):
            raise ValueError('invalid pair')
        src = np.uint8(np.clip(canonical_to_source(sdr) * 255, 0, 255))
        preview = Image.fromarray(src)
        preview.thumbnail((256, 144))
        samples.append((preview, row['asset_id'], row['sdr_curve'], row['sdr_codec']))
    for start in range(0, len(samples), 24):
        sheet = Image.new('RGB', (1024, 6 * 178), '#202020')
        draw = ImageDraw.Draw(sheet)
        for i, (pic, asset, curve, codec) in enumerate(samples[start:start + 24]):
            x, y = (i % 4) * 256, (i // 4) * 178
            sheet.paste(pic, (x, y))
            draw.text((x + 3, y + 145), asset[-34:], fill='white')
            draw.text((x + 3, y + 160), f'{curve} / {codec}', fill='white')
        sheet.save(out / f'inputs_{start // 24:02d}.jpg')
    report = {'pairs': len(rows), 'verified_media': 3 * len(rows), 'scenes': len(scenes),
              'splits': {s: sum(r['split'] == s for r in rows) for s in ('train', 'val')},
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'scope': 'decode, hashes, exact duplicates and scene IDs only; not historical near-duplicate clearance'}
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.manifest, a.out)
