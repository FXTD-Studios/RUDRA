"""Fresh-process tests: strict backend flags must not leak into other tests."""
import json
import subprocess
import sys

import pytest
import torch


def child(code):
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_default_is_noop_and_strict_mode_records_settings():
    result = child('''
import json, torch
from training.train_sdr2hdr import configure_determinism, runtime_record
before = runtime_record(torch.device('cpu'))
configure_determinism(False)
assert before == runtime_record(torch.device('cpu'))
configure_determinism(True)
print(json.dumps(runtime_record(torch.device('cpu'))))
''')
    assert result['deterministic_algorithms']
    assert result['cudnn_deterministic']
    assert not any(result[k] for k in ('deterministic_warn_only', 'cudnn_benchmark',
                                      'cudnn_tf32', 'matmul_tf32'))
    assert result['cublas_workspace_config'] in (':4096:8', ':16:8')


@pytest.mark.parametrize('condition', ['workspace', 'initialized'])
def test_unsafe_setup_is_rejected(condition):
    code = '''
import json, os, torch
from training.train_sdr2hdr import configure_determinism
'''
    code += ('os.environ["CUBLAS_WORKSPACE_CONFIG"] = "invalid"\n' if condition == 'workspace'
             else 'torch.cuda.is_initialized = lambda: True\n')
    code += '''
try:
    configure_determinism(True)
except (ValueError, RuntimeError):
    print('true')
else:
    print('false')
'''
    assert child(code)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA repeatability probe')
def test_cuda_bfloat16_model_updates_repeat_across_processes():
    code = '''
import hashlib, json, torch
from training.train_sdr2hdr import configure_determinism, seed_everything, runtime_record
from rudra.sdr2hdr import SDR2HDRNet, sdr2hdr_loss
configure_determinism(True)
seed_everything(20261013)
torch.set_num_threads(2)
model = SDR2HDRNet(base_channels=8).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
trace = []
for step in range(60):
    sdr = torch.rand(2,3,16,16).cuda()
    sdr[:,:,0:4,0:4] = 1/255
    target = torch.rand_like(sdr) * .2
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(sdr, record_preclamp=True)
        loss = sdr2hdr_loss(output, sdr, target, crushed_hold=1.,
                           crushed_hold_domain='preclamp')['total']
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    assert torch.isfinite(loss)
    trace.append([loss.item(), norm.item()])
digest = hashlib.sha256()
for name, value in model.state_dict().items():
    assert torch.isfinite(value).all()
    digest.update(name.encode())
    digest.update(value.detach().cpu().numpy().tobytes())
print(json.dumps({'trace': trace, 'weights': digest.hexdigest(),
                  'runtime': runtime_record(torch.device('cuda'))}))
'''
    first, second = child(code), child(code)
    assert first == second
    assert len(first['trace']) == 60
