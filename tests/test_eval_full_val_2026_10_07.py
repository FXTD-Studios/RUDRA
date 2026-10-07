"""evaluate_image: ``max_batches=0`` scores the whole loader, and gains are
split by sdr_kind.

Every best.pt from v4 to v7 was selected on ``--eval-batches 8`` x batch 4 =
32 records of a 500+ record val split, and on the mixed manifest (7 Oct 2026)
the pooled gain read +1.6 dB while the PU21/CVVDP benches put the same
checkpoint 19 to 24 dB under the analytic inverse on rendered frames. The
per-kind split is what makes that visible during training instead of after
a 10 h run.
"""
import torch
from torch.utils.data import DataLoader, Dataset

from rudra.sdr2hdr import SDR2HDRNet
from training.train_sdr2hdr import evaluate_image


class _Pairs(Dataset):
    def __init__(self, n: int, with_kind: bool = True):
        g = torch.Generator().manual_seed(7)
        self.sdr = torch.rand(n, 3, 64, 96, generator=g)
        self.hdr = torch.rand(n, 3, 64, 96, generator=g) * 2.0
        self.with_kind = with_kind

    def __len__(self):
        return len(self.sdr)

    def __getitem__(self, i):
        item = {"sdr": self.sdr[i], "hdr": self.hdr[i], "ceiling": torch.tensor(float("inf"))}
        if self.with_kind:
            item["real"] = torch.tensor(1 if i % 3 == 0 else 0)
        return item


def _model():
    torch.manual_seed(0)
    return SDR2HDRNet(base_channels=8, corpus_ev=0.0)


def test_zero_means_every_batch():
    loader = DataLoader(_Pairs(10), batch_size=4, shuffle=False)
    full = evaluate_image(_model(), loader, torch.device("cpu"), max_batches=0)
    capped = evaluate_image(_model(), loader, torch.device("cpu"), max_batches=1)
    assert full["eval_batches"] == 3
    assert capped["eval_batches"] == 1
    assert full["real_frames"] + full["rendered_frames"] == 10


def test_gains_are_split_by_kind():
    loader = DataLoader(_Pairs(9), batch_size=3, shuffle=False)
    m = evaluate_image(_model(), loader, torch.device("cpu"), max_batches=0)
    assert m["real_frames"] == 3 and m["rendered_frames"] == 6
    for kind in ("real", "rendered"):
        assert m[f"{kind}_gain_db"] == m[f"{kind}_psnr_log"] - m[f"{kind}_baseline_psnr_log"]
    # The pooled gain is still the batch-mean definition every earlier log used.
    assert m["gain_db"] == m["psnr_log"] - m["baseline_psnr_log"]


def test_loaders_without_the_flag_still_score():
    """The video dataset and older corpora carry no sdr_kind; nothing breaks."""
    loader = DataLoader(_Pairs(4, with_kind=False), batch_size=2, shuffle=False)
    m = evaluate_image(_model(), loader, torch.device("cpu"), max_batches=0)
    assert m["eval_batches"] == 2
    assert m["real_frames"] == 0 and m["rendered_frames"] == 0
    assert "real_gain_db" not in m
