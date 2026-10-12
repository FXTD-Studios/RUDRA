"""rudra batch runs the app's queues (12 Oct 2026): a job with `grade` (the app's
export settings) goes to rudra-native video, every other job stays in Python."""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from rudra import batch

STUB = """#!{python}
import json, sys
from pathlib import Path
argv = sys.argv[1:]
Path({log!r}).write_text(json.dumps(argv))
out = Path(argv[argv.index('--output') + 1])
out.write_bytes(b'master')
Path(str(out) + '.json').write_text(json.dumps({{'qc': {{'passed': True}}}}))
"""


def stub(tmp_path):
    log = tmp_path / 'argv.json'
    exe = tmp_path / 'rudra-native'
    exe.write_text(STUB.format(python=sys.executable, log=str(log)), encoding='utf-8')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe, log


def queue(tmp_path, checkpoint, grade, options=None):
    (tmp_path / 'clip.mov').write_bytes(b'source')
    q = tmp_path / 'queue.json'
    q.write_text(json.dumps({'version': 1, 'defaults': {'checkpoint': checkpoint},
                             'jobs': [{'input': 'clip.mov', 'output': 'out/clip_hdr.mp4',
                                       'options': {'format': 'hdr10', 'grade': grade,
                                                   'temporal-smoothing': 0.25, **(options or {})}}]}))
    (tmp_path / 'out').mkdir()
    return q


@pytest.mark.skipif(os.name == 'nt', reason='the stub is a POSIX script')
def test_an_analytic_grade_job_runs_in_rudra_native_with_no_package(tmp_path, monkeypatch):
    exe, log = stub(tmp_path)
    monkeypatch.setenv('RUDRA_NATIVE', str(exe))
    (tmp_path / 'clip.settings.json').write_text(json.dumps({'analytic': True}))
    q = queue(tmp_path, 'clip.settings.json', 'clip.settings.json')
    assert batch.run_queue(q) == 0
    argv = json.loads(log.read_text())
    assert argv[:3] == ['video', '-', str((tmp_path / 'clip.mov').resolve())]
    assert argv[argv.index('--grade') + 1] == str((tmp_path / 'clip.settings.json').resolve())
    assert argv[argv.index('--format') + 1] == 'hdr10'
    assert argv[argv.index('--temporal-smoothing') + 1] == '0.25'
    assert '--checkpoint' not in argv
    state = json.loads((tmp_path / 'queue.json.state.json').read_text())
    assert state['jobs'][0]['status'] == 'complete'


@pytest.mark.skipif(os.name == 'nt', reason='the stub is a POSIX script')
def test_a_model_grade_job_names_the_package_folder(tmp_path, monkeypatch):
    exe, log = stub(tmp_path)
    monkeypatch.setenv('RUDRA_NATIVE', str(exe))
    pkg = tmp_path / 'models' / 'sdr2hdr_image_v8'
    pkg.mkdir(parents=True)
    (pkg / 'manifest.json').write_text('{}')
    (tmp_path / 'clip.settings.json').write_text(json.dumps({'peak_nits': 1000}))
    q = queue(tmp_path, 'models/sdr2hdr_image_v8/manifest.json', 'clip.settings.json')
    assert batch.run_queue(q) == 0
    assert json.loads(log.read_text())[1] == str(pkg.resolve())


def test_without_rudra_native_the_job_fails_and_says_why(tmp_path, monkeypatch):
    monkeypatch.delenv('RUDRA_NATIVE', raising=False)
    monkeypatch.setattr(batch.shutil, 'which', lambda name: None)
    (tmp_path / 'clip.settings.json').write_text(json.dumps({'analytic': True}))
    q = queue(tmp_path, 'clip.settings.json', 'clip.settings.json')
    assert batch.run_queue(q) == 1
    state = json.loads((tmp_path / 'queue.json.state.json').read_text())
    assert state['jobs'][0]['status'] == 'failed'
    assert 'rudra-native' in state['jobs'][0]['error']


def test_a_job_without_grade_still_rejects_unknown_options(tmp_path):
    (tmp_path / 'clip.mov').write_bytes(b'source')
    q = tmp_path / 'queue.json'
    q.write_text(json.dumps({'version': 1, 'defaults': {'checkpoint': 'm.pt'},
                             'jobs': [{'input': 'clip.mov', 'output': 'o.mp4', 'options': {'no-such': 1}}]}))
    with pytest.raises(ValueError):
        batch.load_jobs(q)
