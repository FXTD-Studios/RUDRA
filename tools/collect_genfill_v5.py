"""Training material for the AI fill v5 (22 Nov 2026): CC0 HDR panoramas of the cases the fill gets wrong.

    .venv\\Scripts\\python.exe tools\\collect_genfill_v5.py --dry-run          # what it would take, by theme, and the GB
    .venv\\Scripts\\python.exe tools\\collect_genfill_v5.py [--max-gb 30] [--res 4k]

The fill learns from scene-linear truth with its SDR made from it (tools/build_genfill_clips.py), so the
material must be real HDR: normal stock video is clipped SDR already and cannot show what is inside a
blown-out window. Two CC0 libraries (no attribution, commercial use, the licence of today's training set):

  Poly Haven   https://polyhaven.com/license   api.polyhaven.com (assets, files: md5 checked)
  ambientCG    https://docs.ambientcg.com/license/   ambientcg.com/api/v2 (a zip per size, the EXR kept)

Each panorama is sorted into the themes the 09V / testwindow QC asked for, from its tags, categories and
attributes (themes_of): practical lights, windows (interiors looking out), doors and openings (a dark
inside, a bright outside), sun and glare, thin things against the sky (cables, antennas, masts, railings,
bare branches) and pure skies (backgrounds for the Blender scenes, tools/blender/genfill_v5_scenes.py).
The picks are taken theme by theme in turn, the rarest themes first, until --max-gb.

Written to --dest (default G:/datasets/sources/genfill_v5):
  polyhaven/<id>_<res>.exr, ambientcg/<id>_<res>.exr
  manifest.jsonl   one row per file: ids, scene_id (ph:<id> / acg:<id>), themes, licence, page and download
                   URLs, authors, sha256, size, and the file's light (median, p99.99 and peak luminance, and
                   whether the brightest pixels sit on one flat value: a clipped sun, `plateau`)
  receipts/        the API's own record of each asset as fetched (the licence receipt)
Scenes the run05 registry holds out (any split but train) are never fetched; the builder checks again.
A re-run continues: files already in the manifest are kept. Nothing is ever deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DEST = 'G:/datasets/sources/genfill_v5'
REGISTRY = 'G:/datasets/corpora/rudra_run05_sources_20261022/sources.jsonl'
PH_API = 'https://api.polyhaven.com'
ACG_API = 'https://ambientcg.com/api/v2/full_json'
LICENCE = {'polyhaven': ('CC0', 'https://polyhaven.com/license'),
           'ambientcg': ('CC0', 'https://docs.ambientcg.com/license/')}
UA = 'RUDRA-genfill-collector/1.0 (FXTD Studios; dataset build)'
THEME_ORDER = ('thin', 'door', 'practical', 'window', 'sun', 'sky')     # rarest first

PRACTICAL = {'lamp', 'lamps', 'street light', 'streetlight', 'chandelier', 'fluorescent', 'spotlight', 'led panel',
             'neon', 'lantern', 'bulb', 'candle', 'fireplace', 'stage', 'softbox', 'lights', 'reflector',
             'theater', 'concert', 'tube light', 'christmas'}
WINDOW = {'window', 'windows', 'arched windows', 'stained glass', 'curtain', 'curtains', 'glass', 'skylight',
          'conservatory', 'greenhouse'}
DOOR = {'door', 'doors', 'doorway', 'tunnel', 'underpass', 'garage', 'hangar', 'warehouse', 'barn', 'cave',
        'overpass', 'arch', 'alley', 'entrance', 'gate', 'shed', 'loading', 'portal', 'passage'}
SUN = {'sun', 'sunny', 'sunset', 'sunrise', 'sunlight', 'direct', 'golden', 'rays', 'glare', 'backlit',
       'golden hour', 'dusk', 'dawn'}
THIN = {'power line', 'power lines', 'powerline', 'cable', 'cables', 'wire', 'wires', 'antenna', 'radio', 'tower',
        'mast', 'masts', 'crane', 'fence', 'railing', 'railings', 'rail', 'bare trees', 'bare', 'branches', 'tree',
        'trees', 'forest', 'canopy', 'leaves', 'flag', 'rigging', 'harbour', 'harbor', 'boat', 'ship', 'pier',
        'dock', 'jetty', 'scaffold', 'scaffolding', 'construction', 'pole', 'poles', 'bridge', 'street light',
        'pylon', 'windmill', 'wind turbine', 'lamp post', 'satellite'}
SKY = {'pure skies', 'skyonly', 'sky only'}
SKY_CATEGORIES = ('day sky hdri', 'evening sky hdri', 'morning sky hdri')


def words(meta: dict) -> set:
    """Everything an asset says about itself, lower case: tags, categories, attribute values, its
    category path (Poly Haven) and display category (ambientCG)."""
    w = {str(x).strip().lower() for x in list(meta.get('tags') or []) + list(meta.get('categories') or [])}
    w |= {str(v).strip().lower() for v in (meta.get('attributes') or {}).values()}
    for k in ('category', 'displayCategory'):
        if meta.get(k):
            w |= {p.strip().lower() for p in str(meta[k]).split('/')}
    return {x for x in w if x}


def themes_of(meta: dict) -> set:
    """The themes an asset serves: practical, window, door, sun, thin, sky."""
    w = words(meta)
    attrs = {k: str(v).lower() for k, v in (meta.get('attributes') or {}).items()}
    disp = str(meta.get('displayCategory') or '').lower()
    indoor = 'indoor' in w or attrs.get('environment') == 'indoor' or disp.startswith('indoor')
    outdoor = not indoor and ('outdoor' in w or attrs.get('environment') == 'outdoor' or 'sky' in disp
                              or disp.endswith('environment hdri'))
    night = 'night' in w or attrs.get('time_of_day') == 'night' or disp.startswith('night')
    out = set()
    if 'artificial light' in w or attrs.get('light_type') in ('artificial', 'mixed') or w & PRACTICAL:
        out.add('practical')
    if indoor and not night and (w & WINDOW or 'natural light' in w or attrs.get('light_type') in ('natural', 'mixed')):
        out.add('window')
    if w & DOOR and not night:
        out.add('door')
    if outdoor and not night and (w & SUN or 'sunrise-sunset' in w or attrs.get('weather') == 'clear'
                                  or attrs.get('time_of_day') in ('sunrise', 'sunset')):
        out.add('sun')
    if outdoor and not night and attrs.get('sky_view') != 'none' and w & THIN:
        out.add('thin')
    if not night and (w & SKY or disp in SKY_CATEGORIES):
        out.add('sky')
    return out


# --------------------------------------------------------------------------------- network
def fetch(url: str, tries: int = 4, timeout: int = 60) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:                                   # noqa: BLE001 - retried, then raised
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def download(url: str, dst: Path, tries: int = 4) -> None:
    """To dst via dst.part (a stopped download never looks finished)."""
    part = dst.with_name(dst.name + '.part')
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(part, dst)
            return
        except Exception:                                   # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def file_hash(p: Path, algo: str = 'sha256') -> str:
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def get_json(url):
    return json.loads(fetch(url))


# --------------------------------------------------------------------------------- catalogues
def polyhaven_catalogue(get=get_json) -> list[dict]:
    assets = get(f'{PH_API}/assets?type=hdris')
    return [{'library': 'polyhaven', 'id': aid, 'scene_id': 'ph:' + aid, 'name': m.get('name', aid),
             'themes': sorted(themes_of(m)), 'authors': sorted((m.get('authors') or {}).keys()),
             'page': f'https://polyhaven.com/a/{aid}', 'meta': m} for aid, m in sorted(assets.items())]


def acg_catalogue(get=get_json) -> list[dict]:
    out, offset = [], 0
    while True:
        page = get(f'{ACG_API}?type=HDRI&limit=100&offset={offset}&include=tagData,downloadData,displayData')
        found = page.get('foundAssets') or []
        for m in found:
            aid = m['assetId']
            out.append({'library': 'ambientcg', 'id': aid, 'scene_id': 'acg:' + aid, 'name': m.get('displayName', aid),
                        'themes': sorted(themes_of(m)), 'authors': [],
                        'page': m.get('shortLink') or f'https://ambientcg.com/a/{aid}', 'meta': m})
        offset += len(found)
        if not found or offset >= int(page.get('numberOfResults') or 0):
            return out


def acg_zip(meta: dict, res: str) -> dict | None:
    """The ambientCG download of this size (a zip): {'url', 'size'}."""
    want = res.upper()
    for folder in (meta.get('downloadFolders') or {}).values():
        for cat in (folder.get('downloadFiletypeCategories') or {}).values():
            for d in cat.get('downloads') or []:
                if str(d.get('attribute', '')).upper() == want and d.get('filetype') == 'zip':
                    return {'url': d.get('fullDownloadPath') or d['downloadLink'], 'size': int(d.get('size') or 0)}
    return None


def polyhaven_file(aid: str, res: str, get=get_json) -> dict | None:
    """Poly Haven's EXR of this size: {'url', 'size', 'md5'}."""
    try:
        return get(f'{PH_API}/files/{aid}')['hdri'][res]['exr']
    except Exception:                                       # noqa: BLE001 - no such size, or the call failed
        return None


# --------------------------------------------------------------------------------- choosing
def held_out_scenes(registry: Path) -> set:
    """Scene ids the run05 registry keeps out of training (any split but train), and the evaluation
    scenes (tools/build_aces_scale_crops.evaluation_scenes) when the corpora folder is there."""
    out = set()
    if registry.exists():
        for line in registry.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get('split') != 'train':
                    out.add(r['scene_id'])
    try:
        from tools.build_aces_scale_crops import evaluation_scenes
        from tools.build_genfill_clips import FRESH
        ex, _, fresh = evaluation_scenes(registry.parents[1], Path(FRESH))
        out |= set(ex) | set(fresh)
    except Exception:                                       # noqa: BLE001 - not on this machine
        pass
    return out


def choose(cands: list[dict], sizes: dict, budget_bytes: float, held: set, per_theme: int = 0,
           have: frozenset | set = frozenset()) -> list[dict]:
    """Theme by theme in turn (THEME_ORDER, rarest first) the next asset of the theme not yet taken, until
    the budget: every theme gets its share before any gets a second helping. Within a theme, assets that
    serve more themes come first, then the smaller file. Assets with no theme, held-out scenes and assets
    without a file of the size are left out; files already collected (`have`, scene ids) are kept and
    counted against the budget."""
    ok = lambda c: c['themes'] and c['scene_id'] not in held and sizes.get(c['scene_id'])   # noqa: E731
    pools = {t: sorted((c for c in cands if t in c['themes'] and ok(c)),
                       key=lambda c: (-len(c['themes']), sizes[c['scene_id']], c['scene_id'])) for t in THEME_ORDER}
    taken, seen, used = [], set(), 0.0
    count = {t: 0 for t in THEME_ORDER}

    def take(c):
        nonlocal used
        taken.append(c); seen.add(c['scene_id']); used += sizes[c['scene_id']]
        for x in c['themes']:
            if x in count:
                count[x] += 1
    for c in cands:
        if c['scene_id'] in have and ok(c):
            take(c)
    moved = True
    while moved:
        moved = False
        for t in THEME_ORDER:
            if per_theme and count[t] >= per_theme:
                continue
            while pools[t] and (pools[t][0]['scene_id'] in seen or used + sizes[pools[t][0]['scene_id']] > budget_bytes):
                pools[t].pop(0)
            if pools[t]:
                take(pools[t].pop(0))
                moved = True
    return taken


# --------------------------------------------------------------------------------- the light
def light_stats(path: Path) -> dict:
    """Median, p99.99 and peak luminance (Rec.709 Y), and whether the top 0.005 % sit on one flat value
    (a sun clipped in the capture: no use for glare, still fine for the other themes)."""
    import numpy as np
    from tools.build_genfill_clips import read_exr_rgb
    a = read_exr_rgb(path)
    y = (a @ np.array([.2126, .7152, .0722], np.float32)).ravel()
    y = y[np.isfinite(y)]
    peak = float(y.max())
    top = np.sort(y)[-max(8, int(len(y) * 5e-5)):]
    med = float(np.median(y))
    return {'median': round(med, 6), 'p9999': round(float(np.percentile(y, 99.99)), 4), 'peak': round(peak, 3),
            'dynamic_range_stops': round(float(np.log2(max(peak, 1e-9) / max(med, 1e-9))), 2),
            'plateau': bool(peak > 0 and float(top[0]) >= 0.999 * peak)}


def extract_exr(zpath: Path, dst: Path) -> None:
    """The largest .exr in an ambientCG zip -> dst."""
    with zipfile.ZipFile(zpath) as z:
        exrs = [i for i in z.infolist() if i.filename.lower().endswith('.exr')]
        if not exrs:
            raise ValueError(f'{zpath.name}: no EXR inside ({[i.filename for i in z.infolist()]})')
        best = max(exrs, key=lambda i: i.file_size)
        part = dst.with_name(dst.name + '.part')
        with z.open(best) as src, open(part, 'wb') as f:
            shutil.copyfileobj(src, f, 1 << 20)
        os.replace(part, dst)


# --------------------------------------------------------------------------------- main
def main(argv=None, get=get_json, fetch_file=download):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', default=DEST)
    ap.add_argument('--registry', default=REGISTRY)
    ap.add_argument('--res', default='4k', choices=('2k', '4k', '8k'),
                    help='4k (default): thin cables and antennas need it (2k is ~6 px a degree, the clips want ~15)')
    ap.add_argument('--max-gb', type=float, default=30.0)
    ap.add_argument('--per-theme', type=int, default=0, help='at most this many per theme (0: no limit)')
    ap.add_argument('--library', choices=('both', 'polyhaven', 'ambientcg'), default='both')
    ap.add_argument('--keep-free-gb', type=float, default=10.0, help='stop when the drive would have less free')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    held = held_out_scenes(Path(a.registry))
    cands = []
    if a.library in ('both', 'polyhaven'):
        cands += polyhaven_catalogue(get)
    if a.library in ('both', 'ambientcg'):
        cands += acg_catalogue(get)
    print(json.dumps({'catalogue': len(cands), 'held_out': len(held),
                      'by_theme': {t: sum(t in c['themes'] for c in cands) for t in THEME_ORDER}}), flush=True)
    manifest = dest / 'manifest.jsonl'
    rows = [json.loads(l) for l in manifest.read_text(encoding='utf-8').splitlines() if l.strip()] if manifest.exists() else []
    have = {r['scene_id'] for r in rows if r.get('res') == a.res}
    sizes, urls = {}, {}
    want = [c for c in cands if c['themes'] and c['scene_id'] not in held]

    def file_of(c):                                          # Poly Haven's sizes: one call per asset
        return c, (acg_zip(c['meta'], a.res) if c['library'] == 'ambientcg' else polyhaven_file(c['id'], a.res, get))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        for c, f in pool.map(file_of, want):
            if f:
                sizes[c['scene_id']], urls[c['scene_id']] = int(f.get('size') or 0) or 1, f
    picks = choose(cands, sizes, a.max_gb * 2 ** 30, held, a.per_theme, have)
    plan = {'picked': len(picks), 'gb': round(sum(sizes[c['scene_id']] for c in picks) / 2 ** 30, 2),
            'by_theme': {t: sum(t in c['themes'] for c in picks) for t in THEME_ORDER},
            'by_library': {lib: sum(c['library'] == lib for c in picks) for lib in ('polyhaven', 'ambientcg')},
            'already_here': sum(c['scene_id'] in have for c in picks)}
    print(json.dumps({'plan': plan}), flush=True)
    if a.dry_run:
        for c in picks:
            print(f"  {c['scene_id']:55s} {sizes[c['scene_id']] / 2 ** 20:7.1f} MB  {','.join(c['themes'])}")
        return plan
    for sub in ('polyhaven', 'ambientcg', 'receipts'):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    got = 0
    for k, c in enumerate(picks):
        sid = c['scene_id']
        if sid in have:
            continue
        free = shutil.disk_usage(dest).free
        if free - sizes[sid] * (2.2 if c['library'] == 'ambientcg' else 1.0) < a.keep_free_gb * 2 ** 30:
            print(json.dumps({'stop': 'disk', 'free_gb': round(free / 2 ** 30, 1)}), flush=True)
            break
        u = urls[sid]
        dst = dest / c['library'] / f"{c['id']}_{a.res}.exr"
        t0 = time.time()
        try:
            if c['library'] == 'polyhaven':
                fetch_file(u['url'], dst)
                if u.get('md5') and file_hash(dst, 'md5') != u['md5']:
                    os.replace(dst, dst.with_name(dst.name + '.bad'))
                    raise ValueError('md5 differs from the API')
            else:
                zp = dest / 'ambientcg' / f"{c['id']}_{a.res}.zip"
                fetch_file(u['url'], zp)
                extract_exr(zp, dst)
                zp.unlink()                                  # the zip only carried the EXR
            stats = light_stats(dst)
        except Exception as e:                              # noqa: BLE001 - one bad asset does not stop the run
            print(json.dumps({'failed': sid, 'error': f'{type(e).__name__}: {e}'}), flush=True)
            continue
        (dest / 'receipts' / f"{c['library']}_{c['id']}.json").write_text(json.dumps(c['meta'], indent=1), encoding='utf-8')
        lic, lic_url = LICENCE[c['library']]
        row = {'asset_id': f"{c['library']}{a.res}:{c['id']}", 'scene_id': sid, 'library': c['library'], 'id': c['id'],
               'name': c['name'], 'themes': c['themes'], 'res': a.res, 'path': str(dst).replace('\\', '/'),
               'bytes': dst.stat().st_size, 'sha256': file_hash(dst), 'licence': lic, 'licence_url': lic_url,
               'page': c['page'], 'download_url': u['url'], 'authors': c['authors'], 'space': 'rec709',
               'fetched': time.strftime('%Y-%m-%dT%H:%M:%S'), 'light': stats}
        with open(manifest, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        have.add(sid)
        got += 1
        print(json.dumps({'got': sid, 'n': k + 1, 'of': len(picks), 'mb': round(row['bytes'] / 2 ** 20, 1),
                          'seconds': round(time.time() - t0, 1), 'themes': c['themes'],
                          'peak': stats['peak'], 'plateau': stats['plateau']}), flush=True)
    plan['fetched'] = got
    return plan


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
