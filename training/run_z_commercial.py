"""Use the established bounded scratch runner with a verified expanded corpus.

The data adapter is local to this new Python process. The original runner and
the current training process are never edited or reconfigured.
"""
from pathlib import Path
import json
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.commercial_preflight import sha256
from pipeline.prepare_z_footage import DEFAULT_OUT
from training import run_fxtd_commercial as runner


def prepare_combined(run):
    ready_path = DEFAULT_OUT / 'ready.json'
    if not ready_path.is_file():
        raise ValueError('Expanded corpus is not verified yet; finish prepare_z_commercial first')
    ready = json.loads(ready_path.read_text())
    manifest = Path(ready['manifest'])
    receipt_path = Path(ready['receipt'])
    if sha256(manifest) != ready['manifest_sha256'] or sha256(receipt_path) != ready['receipt_sha256']:
        raise ValueError('Expanded corpus receipt or manifest changed')
    receipt = json.loads(receipt_path.read_text())
    for path, expected in receipt['preparation_dependencies'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('Preparation evidence changed: ' + path)
    data = run / 'data'
    data.mkdir(exist_ok=True)
    mapping = {manifest: data/'manifest.jsonl', receipt_path:data/'review.json',
               DEFAULT_OUT/'owner_training_scope.txt':data/'owner_training_scope.txt'}
    for source,target in mapping.items():
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError('Run-local data changed; use a new run name')
        if not target.exists():
            shutil.copyfile(source,target)
    # The trainer independently validates receipt, dependencies and all media.
    return data


if __name__ == '__main__':
    runner.prepare = prepare_combined
    runner.main()
