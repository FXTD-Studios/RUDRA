"""Create source-only contact sheets for a completed panorama overlap screen."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_holdout_panorama_similarity import load_rgb


def main():
    root = Path('reports/ambientcg_holdout_candidates_20261025')
    scan = root / 'panorama_similarity'
    result = json.loads((scan / 'report.json').read_text())
    registration = json.loads((scan / 'registration.json').read_text())
    assets = {a['asset_id']: a for a in json.loads((root / 'downloads.json').read_text())}
    out = scan / 'review'
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(2)
    for candidate, nearest in result['nearest'].items():
        source = next(f['path'] for f in assets[candidate]['files'] if f['path'].endswith('.exr'))
        items = [(f'Candidate: {candidate}', source)]
        items += [(f"{r['known']} | score {r['score']:.6f} | flag {r['flagged']}",
                   registration['known_panorama_paths'][r['known']]) for r in nearest]
        sheet = Image.new('RGB', (1536, 1224), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for i, (label, path) in enumerate(items):
            rgb = cv2.resize(load_rgb(path), (768, 384), interpolation=cv2.INTER_AREA)
            scale = max(float(np.percentile(rgb, 90)), 1e-6)
            rgb = np.maximum(rgb / scale, 0)
            display = np.power(rgb / (1 + rgb), 1 / 2.2)
            image = Image.fromarray(np.uint8(np.clip(display * 255 + .5, 0, 255)))
            x, y = (i % 2) * 768, (i // 2) * 408
            sheet.paste(image, (x, y + 24)); draw.text((x + 8, y + 5), label, fill='white')
        sheet.save(out / (candidate + '.png'))
    (out / 'scope.txt').write_text('Source identity review only. Per-image90th-percentile scaling, RGB/(1+RGB), gamma1/2.2. No model output or grading-quality evaluation. Original similarity flags retained.')


if __name__ == '__main__':
    main()
