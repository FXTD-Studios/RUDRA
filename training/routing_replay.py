"""CPU augmentation replay for the isolated, zero-worker routing experiment."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import random

import numpy as np
import torch

from rudra.source_baseline import baseline_curve_index


def fingerprint(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode() + b"\0")
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode() + b"\0")
            digest.update(value.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
        digest.update(b"\0")
    return digest.hexdigest()


@contextmanager
def augmentation_rng(seed):
    """Restore global streams; never seed or otherwise touch CUDA RNG state."""
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class ReplayDataset:
    """Both arms generate corrected-routing samples from identical local RNGs.

    Control then restores the old real-record BT.1886 selection. Only that
    baseline ID may differ; all returned tensors and metadata are fingerprinted.
    matched_baselines=True retains corrected routing in both arms and requires
    baseline IDs to match too, for the isolated preservation comparison.
    allowed_baseline_change=(control_id, candidate_id) relaxes that for one
    declared direction only (the push-baseline comparison: BT.1886 -> ACES);
    every other difference still stops the run, and the changes are counted.
    Works only with num_workers=0, enforced by the experiment runner.
    """
    def __init__(self, dataset, arm, seed, journal, control_journal=None, *, matched_baselines=False,
                 allowed_baseline_change=None):
        if arm not in ("control", "corrected"):
            raise ValueError("unknown routing arm")
        self.dataset, self.records = dataset, dataset.records
        self.arm, self.seed, self.count = arm, seed, 0
        self.changed_baselines = 0
        self.matched_baselines = matched_baselines
        if allowed_baseline_change is not None:
            if not matched_baselines or len(allowed_baseline_change) != 2:
                raise ValueError("a declared baseline change needs matched baselines and (control, candidate)")
            allowed_baseline_change = tuple(int(v) for v in allowed_baseline_change)
        self.allowed_baseline_change = allowed_baseline_change
        self.control = open(control_journal, encoding="utf-8") if arm == "corrected" else None
        try:
            self.journal = open(journal, "x", encoding="utf-8")
        except BaseException:
            if self.control:
                self.control.close()
            raise

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("routing replay requires zero loader workers")
        seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.count}:{index}".encode()).digest()[:8], "big")
        with augmentation_rng(seed):
            sample = dict(self.dataset[index])
        if not self.matched_baselines and self.arm == "control" and self.records[index].get("sdr_kind") == "real":
            sample["baseline_curve"] = torch.tensor(baseline_curve_index("bt1886"), dtype=torch.long)
        record = {"draw": self.count, "record_index": int(index), "augmentation_seed": seed,
                  "sample_sha256": fingerprint({k: v for k, v in sample.items() if k != "baseline_curve"}),
                  "baseline_curve": int(sample["baseline_curve"])}
        if self.control:
            line = self.control.readline()
            if not line:
                raise ValueError("control replay ended before corrected arm")
            expected = json.loads(line)
            if any(record[k] != expected[k] for k in record if k != "baseline_curve"):
                raise ValueError(f"paired input mismatch at draw {self.count}")
            if (self.matched_baselines and record["baseline_curve"] != expected["baseline_curve"]
                    and (expected["baseline_curve"], record["baseline_curve"]) != self.allowed_baseline_change):
                raise ValueError(f"paired baseline mismatch at draw {self.count}")
            self.changed_baselines += record["baseline_curve"] != expected["baseline_curve"]
        self.journal.write(json.dumps(record) + "\n")
        self.journal.flush()
        self.count += 1
        return sample

    def finish(self, expected_count):
        if self.count != expected_count:
            raise ValueError(f"replay coverage {self.count} != {expected_count}")
        if self.control and self.control.readline():
            raise ValueError("corrected replay ended before control arm")

    def close(self):
        self.journal.close()
        if self.control:
            self.control.close()
