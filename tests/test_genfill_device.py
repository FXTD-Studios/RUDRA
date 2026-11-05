"""AI fill device choice (31 Oct 2026): CUDA in bfloat16, Apple MPS on a Mac, else the CPU."""
import torch

from rudra.genfill.run import pick_device


def _machine(monkeypatch, cuda, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_cuda_first_in_bfloat16(monkeypatch):
    _machine(monkeypatch, cuda=True, mps=False)
    assert pick_device("auto") == ("cuda", torch.bfloat16)
    assert pick_device("cuda") == ("cuda", torch.bfloat16)


def test_a_mac_uses_mps_in_float32(monkeypatch):
    _machine(monkeypatch, cuda=False, mps=True)
    assert pick_device("auto") == ("mps", torch.float32)
    assert pick_device("cuda") == ("mps", torch.float32)   # the app's old default still finds the Mac GPU
    assert pick_device("mps") == ("mps", torch.float32)


def test_cpu_otherwise_and_when_asked(monkeypatch):
    _machine(monkeypatch, cuda=False, mps=False)
    assert pick_device("auto") == ("cpu", torch.float32)
    assert pick_device("mps") == ("cpu", torch.float32)
    _machine(monkeypatch, cuda=True, mps=True)
    assert pick_device("cpu") == ("cpu", torch.float32)
