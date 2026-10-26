"""Model provenance: walk a checkpoint's parent chain and archive the exact code it recorded.

  collect  (no torch needed):  python tools/provenance_archive.py collect <checkpoint.pt> --search <dir> [<dir>...] --out chain.json
      Follows recipe.parent_sha256 (or parent_checkpoint) through .pt files under the search
      roots, recording every *_sha256 the recipes declare.
  resolve  (git only):     python tools/provenance_archive.py resolve chain.json --repo <repo> --out <archive_dir>
      Indexes sha256 of every blob in the working tree and the full git history
      (training/ tools/ rudra/ pipeline/ configs/), maps each recorded hash to the
      file/commits that produced it, copies those exact bytes into <archive_dir>/code/,
      and writes <archive_dir>/provenance.json. Unresolved hashes are listed, not guessed.

Recorded hashes that name data (pilot, registration, replay, parent) are resolved
against files under the search roots in collect when possible; code hashes in resolve.
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

PARENT_KEYS = ('parent_sha256', 'base_checkpoint_sha256', 'parent_checkpoint_sha256', 'init_checkpoint_sha256')
DATA_WORDS = ('pilot', 'registration', 'replay', 'parent', 'checkpoint', 'dataset', 'corpus')
CODE_DIRS = ['training', 'tools', 'rudra', 'pipeline', 'configs', 'scripts']


def sha_file(p, cache):
    p = Path(p)
    st = p.stat()
    key = f'{p}|{st.st_size}|{int(st.st_mtime)}'
    if key not in cache:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        cache[key] = h.hexdigest()
    return cache[key]


def load_meta(path):
    """Checkpoint metadata without torch: unpickle data.pkl with tensors stubbed out."""
    import pickle, zipfile

    class Stub:
        def __init__(self, *args, **kw): pass
        def __call__(self, *args, **kw): return Stub()
        def __setstate__(self, state): pass

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(('torch', 'numpy', 'collections')) and not (module == 'collections' and name == 'OrderedDict'):
                return Stub
            return super().find_class(module, name)

        def persistent_load(self, pid):
            return None

    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('data.pkl'))
        with z.open(name) as f:
            obj = U(f).load()
    return obj if isinstance(obj, dict) else {}


def collect(a):
    cache_path = Path(a.out).with_suffix('.hashcache.json')
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    index = {}
    for root in a.search:
        for p in Path(root).rglob('*.pt'):
            try:
                index.setdefault(sha_file(p, cache), str(p))
            except OSError:
                pass
    cache_path.write_text(json.dumps(cache))
    chain, cur, seen = [], Path(a.checkpoint), set()
    while cur is not None:
        s = sha_file(cur, cache)
        if s in seen:
            break
        seen.add(s)
        st = load_meta(cur)
        recipe = st.get('recipe', {}) or {}
        hashes = {k: v for k, v in recipe.items() if k.endswith('sha256') and isinstance(v, str)}
        node = {'checkpoint': str(cur), 'sha256': s, 'contract_id': (st.get('contract') or {}).get('id'),
                'research_only': st.get('research_only'), 'recorded_sha256': hashes,
                'recipe_scalars': {k: v for k, v in recipe.items() if isinstance(v, (int, float, str, bool)) and not k.endswith('sha256')}}
        parent_sha = next((recipe[k] for k in PARENT_KEYS if isinstance(recipe.get(k), str)), None)
        parent_path = recipe.get('parent_checkpoint') or recipe.get('parent')
        nxt = None
        if parent_sha and parent_sha in index:
            nxt = Path(index[parent_sha])
        elif isinstance(parent_path, str) and Path(parent_path).exists():
            nxt = Path(parent_path)
        node['parent'] = {'sha256': parent_sha, 'declared_path': parent_path, 'found': str(nxt) if nxt else None}
        if parent_sha is None and parent_path is None:
            node['parent']['note'] = 'no parent recorded (declares fresh initialization or predates parent recording)'
        chain.append(node)
        print(len(chain), cur, 'parent', 'found' if nxt else ('none recorded' if not parent_sha else 'NOT FOUND'), flush=True)
        cur = nxt
    Path(a.out).write_text(json.dumps({'root': str(a.checkpoint), 'chain': chain}, indent=1))


def git(repo, *args, inp=None):
    return subprocess.run(['git', '-C', str(repo), '--no-optional-locks', *args], input=inp, capture_output=True, check=True).stdout


def resolve(a):
    repo = Path(a.repo)
    chain = json.loads(Path(a.chain).read_text())['chain']
    wanted = {}
    for i, n in enumerate(chain):
        for k, v in n['recorded_sha256'].items():
            wanted.setdefault(v, []).append({'node': i, 'checkpoint': n['checkpoint'], 'key': k})
    found = {}
    # working tree
    for d in CODE_DIRS:
        for p in (repo / d).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.yml', '.ps1', '.txt', '.cfg'):
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                if h in wanted:
                    found.setdefault(h, {'working_tree': [], 'history': []})['working_tree'].append(str(p.relative_to(repo)))
    # code snapshots kept beside runs (e.g. <run>/trainer_snapshot.py)
    for root in a.extra or []:
        for p in Path(root).rglob('*.py'):
            try:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                continue
            if h in wanted:
                found.setdefault(h, {'working_tree': [], 'history': []}).setdefault('snapshots', []).append(str(p))
    # full history: every blob under the code dirs
    objs = git(repo, 'rev-list', '--all', '--objects', '--', *CODE_DIRS).decode(errors='replace').splitlines()
    blobs = {}
    for line in objs:
        parts = line.split(' ', 1)
        if len(parts) == 2:
            blobs.setdefault(parts[0], parts[1])
    shas = list(blobs)
    contents = {}
    batch = git(repo, 'cat-file', '--batch', inp=('\n'.join(shas) + '\n').encode())
    pos = 0
    while pos < len(batch):
        nl = batch.index(b'\n', pos)
        hdr = batch[pos:nl].split()
        pos = nl + 1
        if len(hdr) < 3 or hdr[1] != b'blob':
            if len(hdr) >= 3:
                pos += int(hdr[2]) + 1
            continue
        size = int(hdr[2])
        data = batch[pos:pos + size]
        pos += size + 1
        h = hashlib.sha256(data).hexdigest()
        if h in wanted:
            found.setdefault(h, {'working_tree': [], 'history': []})
            path = blobs[hdr[0].decode()]
            if path not in found[h]['history']:
                found[h]['history'].append(path)
            contents[h] = (path, data)
    out = Path(a.out)
    (out / 'code').mkdir(parents=True, exist_ok=True)
    for h, info in found.items():
        if h in contents:
            path, data = contents[h]
        elif info['working_tree']:
            path = info['working_tree'][0]
            data = (repo / path).read_bytes()
        else:
            path = 'snapshot/' + Path(info['snapshots'][0]).parent.name + '/' + Path(info['snapshots'][0]).name
            data = Path(info['snapshots'][0]).read_bytes()
        dst = out / 'code' / h[:12] / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        if info['history']:
            first = git(repo, 'log', '--all', '--format=%H %cI', '--find-object=' + git(repo, 'hash-object', '--stdin', inp=data).decode().strip()).decode().split('\n')
            info['commits'] = [c for c in first if c][:5]
        info['archived'] = str(dst.relative_to(out))
    rows = []
    for h, uses in wanted.items():
        rows.append({'sha256': h, 'uses': uses, 'resolved': h in found, **found.get(h, {})})
    code_keys = [r for r in rows if any(not any(w in u['key'] for w in DATA_WORDS) for u in r['uses'])]
    report = {'repo': str(repo), 'chain_length': len(chain), 'recorded_hashes': len(rows),
              'code_hashes': len(code_keys), 'code_resolved': sum(r['resolved'] for r in code_keys),
              'unresolved_code': [{'sha256': r['sha256'], 'uses': r['uses']} for r in code_keys if not r['resolved']],
              'rows': rows, 'chain': chain}
    (out / 'provenance.json').write_text(json.dumps(report, indent=1))
    print('code hashes', report['code_hashes'], 'resolved', report['code_resolved'], '-> ', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect'); c.add_argument('checkpoint'); c.add_argument('--search', nargs='+', required=True); c.add_argument('--out', required=True)
    r = sub.add_parser('resolve'); r.add_argument('chain'); r.add_argument('--repo', required=True); r.add_argument('--out', required=True); r.add_argument('--extra', nargs='*', help='folders with code snapshots (run dirs)')
    a = ap.parse_args()
    collect(a) if a.cmd == 'collect' else resolve(a)


if __name__ == '__main__':
    main()
