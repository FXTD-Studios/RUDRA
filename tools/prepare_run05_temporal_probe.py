"""Freeze two CC0 camera-move sequences for development-only temporal probes.

One look and exposure per short9-frame clip; no framewise auto-exposure. No
object-motion or generator coverage is implied. Timing is synthetic24fps.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import load_plate, render_look, canonical, NETWORK_PER_SCENE
from tools.evaluate_run05_dequant import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('new output required')
    torch.set_num_threads(2)
    sources = {r['asset_id']: r for r in map(json.loads, args.sources.read_text().splitlines())}
    rows = [r for r in map(json.loads, args.manifest.read_text().splitlines()) if r['split']=='val']
    chosen = {}
    for row in sorted(rows, key=lambda r: (r['scene_id'],r['asset_id'])):
        chosen.setdefault(row['scene_id'], row)
    chosen = list(chosen.values())[:2]
    if len(chosen) != 2:
        raise ValueError('two development scenes required')
    args.out.mkdir(parents=True)
    clips = []
    for index, row in enumerate(chosen):
        origin = sources[row['parent_asset_id']]
        if origin['licence_licence'] != 'CC0' or not origin['commercial_ok'] or origin['split']!='val':
            raise ValueError('expected CC0 validation source')
        first = Path(origin['hdr_path'])
        prefix = first.stem.rsplit('_',1)[0]
        files = [first.parent / f'{prefix}_{i:05d}.png' for i in range(9)]
        if not all(f.is_file() for f in files):
            raise ValueError('need9 consecutive source frames')
        if sha(files[0]) != row['parent_hdr_sha256']:
            raise ValueError('first source frame identity mismatch')
        directory = args.out / f'clip{index:02d}'
        directory.mkdir()
        frames = []
        for i, path in enumerate(files):
            plate = load_plate({**origin, 'hdr_path': str(path)})
            clean = render_look(plate / NETWORK_PER_SCENE, row['look'], row['render_ev'])
            code8 = np.rint(clean*255).astype(np.uint8)
            canon = canonical(code8.astype(np.float32)/255)
            output = directory / f'{i:05d}.png'
            encoded = np.rint(canon*65535).astype(np.uint16)
            if not cv2.imwrite(str(output), encoded[...,::-1]):
                raise ValueError('failed to write frame')
            frames.append({'frame': i, 'sdr_path': str(output), 'sdr_sha256': sha(output),
                           'source_path': str(path), 'source_sha256': sha(path)})
        clips.append({'scene_id': row['scene_id'], 'look': row['look'], 'fixed_render_ev': row['render_ev'],
                      'codec_applied': 'none', 'vae_applied': False,
                      'licence': 'CC0', 'licence_url': 'https://polyhaven.com/license',
                      'fps': 24, 'frames': frames})
        print(f'[temporal fixture] {index+1}/2', flush=True)
    report = {'sources_sha256': sha(args.sources), 'development_manifest_sha256': sha(args.manifest),
              'clips': clips, 'storage': '16bit canonical Rec2020 sRGB PNG from quantized8bit Rec709 source',
              'scope': 'short9-frame development camera-motion smoke only; longer sequences, object motion, codec/VAE and independent release evidence remain required',
              'release_authorized': False}
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
