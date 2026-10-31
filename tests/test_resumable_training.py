"""Resume workflow: exact continuation, corruption fallback, refusals, locks, lineage."""
import json
import socket
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from pipeline.hdr_io import HDRStorage, encode_hdr_u16
from training import resumable_training as rt


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def corpus(root: Path, n_train=8, n_val=2) -> Path:
    storage = HDRStorage(mode="log2_extended")
    (root / "media").mkdir(parents=True)
    (root / "_ingest_config.json").write_text(json.dumps({"storage": storage.as_dict()}))
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_train + n_val):
        lin = rng.uniform(0.0, 4.0, size=(40, 48, 3)).astype(np.float32)
        sdr = (np.clip(lin / (1.0 + lin), 0, 1) ** (1 / 2.2) * 65535).astype(np.uint16)
        hdr, _ = encode_hdr_u16(lin, storage)
        sp, hp = root / "media" / f"s{i}.png", root / "media" / f"h{i}.png"
        assert cv2.imwrite(str(sp), sdr) and cv2.imwrite(str(hp), hdr)
        rows.append(dict(asset_id=f"a{i}", scene_id=f"scene{i}", source_id="src",
                         split="train" if i < n_train else "val", sdr_path=str(sp), hdr_path=str(hp),
                         sdr_kind="rendered", tonemap_ev=None, render_family="pipeline",
                         pipeline_recipe={"family": "test"},
                         sdr_sha256=rt.sha256_file(sp), hdr_sha256=rt.sha256_file(hp)))
    manifest = root / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def argv(manifest, out, *extra):
    return ["--manifest", str(manifest), "--output-dir", str(out), "--device", "cpu", "--workers", "0",
            "--steps", "9", "--batch-size", "2", "--crop-size", "32", "--base-channels", "8",
            "--eval-every", "3", "--save-every", "3", "--eval-batches", "0", "--log-every", "1",
            "--seed", "11", "--lr", "0.001", "--baseline-source", "--baseline-dropout", "0",
            "--pipeline-corpus-ev", "0", "--crushed-hold", "1", "--crushed-hold-domain", "preclamp",
            "--best-smoothing", "1", "--degradation-probability", "0.5", *extra]


def run(manifest, out, *extra, **kw):
    _, args = rt.parse(argv(manifest, out, *extra))
    return rt.train(args, **kw)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_dataset_policy_must_match_model_before_training(tmp_path):
    manifest = corpus(tmp_path / 'data')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]['reconstruction_policy'] = 'observed_clip_v1'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='dataset reconstruction policies'):
        run(manifest, tmp_path / 'run')


def test_observed_clip_training_records_and_loads_actual_policy(tmp_path):
    from rudra.sdr2hdr import SDR2HDRNet
    manifest = corpus(tmp_path / 'data')
    rows = [dict(json.loads(line), reconstruction_policy='observed_clip_v1') for line in manifest.read_text().splitlines()]
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    out = tmp_path / 'run'
    run(manifest, out, '--bounded-source', '--reconstruction-policy', 'observed_clip_v1',
        '--crushed-hold', '0', '--crushed-hold-domain', 'postclamp', '--steps', '3')
    checkpoint = load(out / 'best.pt')
    model = SDR2HDRNet.from_config(checkpoint['config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    assert model.reconstruction_policy == 'observed_clip_v1'


def same_tensors(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def same_optimizer(a, b):
    assert a["param_groups"] == b["param_groups"]
    for k in a["state"]:
        for name, value in a["state"][k].items():
            assert torch.equal(value, b["state"][k][name]) if torch.is_tensor(value) else value == b["state"][k][name]


def evals(out):
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    return [r for r in rows if "eval" in r or "total" in r]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("ref")
    m = corpus(root / "c")
    out = root / "uninterrupted"
    result = run(m, out)
    assert result["state"] == "completed" and result["step"] == 9
    return m, out


def test_uninterrupted_run_matches_the_legacy_scratch_trainer(reference, tmp_path, monkeypatch):
    m, out = reference
    from training import train_sdr2hdr as legacy
    monkeypatch.setattr(sys, "argv", ["train_sdr2hdr.py", *argv(m, tmp_path / "legacy")])
    legacy.train(legacy.parse_args())
    same_tensors(load(tmp_path / "legacy" / "step_0000009.pt")["model"], load(out / "step_0000009.pt")["model"])
    old = [r for r in map(json.loads, (tmp_path / "legacy" / "train.jsonl").read_text().splitlines())
           if "eval" in r or "total" in r]
    assert old == evals(out)


@pytest.mark.parametrize("stop_at", [5, 6])  # 5: mid-interval slot save; 6: numbered save; both mid-epoch
def test_interrupted_then_resumed_is_bitwise_identical(reference, tmp_path, stop_at):
    m, ref = reference
    out = tmp_path / "run"
    first = run(m, out, control=rt.Control(out / "STOP", None, stop_at_step=stop_at))
    assert first == {**first, "state": "stopped_resumable", "step": stop_at}
    assert not (out / "step_0000009.pt").exists()
    second = run(m, out, resume=True)
    assert second["state"] == "completed"
    a, b = load(ref / "step_0000009.pt"), load(out / "step_0000009.pt")
    same_tensors(a["model"], b["model"])
    same_optimizer(a["optimizer"], b["optimizer"])
    assert a["scheduler"] == b["scheduler"] and a["best_step"] == b["best_step"]
    same_tensors(load(ref / "best.pt")["model"], load(out / "best.pt")["model"])
    assert evals(ref) == evals(out)
    lineage = json.loads((out / "lineage.json").read_text())
    assert [s["start_step"] for s in lineage["segments"]] == [0, stop_at]
    assert lineage["segments"][1]["resumed_from"]["step"] == stop_at


def test_corrupt_newest_state_falls_back_and_orphans_lost_records(reference, tmp_path):
    m, ref = reference
    out = tmp_path / "run"
    run(m, out, control=rt.Control(out / "STOP", None, stop_at_step=7))
    slot = next(p for p in (out / "resume").glob("slot_*.pt") if load(p)["step"] == 7)
    slot.write_bytes(slot.read_bytes()[:1000])  # torn write after the sidecar
    assert rt.verify_state(slot)[0] is None
    second = run(m, out, resume=True)
    lineage = json.loads((out / "lineage.json").read_text())
    resumed = lineage["segments"][1]["resumed_from"]
    assert resumed["step"] == 6 and resumed["path"].endswith("step_0000006.pt")
    assert any(c["status"].startswith("sha256 mismatch") for c in resumed["candidates_checked"])
    orphan = Path(resumed["orphaned_log"])
    assert all(json.loads(l)["step"] == 7 for l in orphan.read_text().splitlines())
    assert second["state"] == "completed"
    same_tensors(load(ref / "step_0000009.pt")["model"], load(out / "step_0000009.pt")["model"])


def test_incompatible_settings_and_code_changes_are_refused(reference, tmp_path, monkeypatch):
    m, _ = reference
    out = tmp_path / "run"
    run(m, out, control=rt.Control(out / "STOP", None, stop_at_step=4))
    with pytest.raises(rt.ResumeError, match="configuration differs.*lr"):
        run(m, out, "--lr", "0.002", resume=True)
    with pytest.raises(rt.ResumeError, match="configuration differs.*steps"):
        _, args = rt.parse(argv(m, out))
        args.steps = 12
        rt.train(args, resume=True)
    real = rt.code_hashes
    monkeypatch.setattr(rt, "code_hashes", lambda: {**real(), "rudra/sdr2hdr.py": "0" * 64})
    with pytest.raises(rt.ResumeError, match="training code changed"):
        run(m, out, resume=True)
    monkeypatch.setattr(rt, "code_hashes", real)
    # the refusals left the run intact and resumable
    assert run(m, out, resume=True)["state"] == "completed"


def test_stale_and_live_locks(reference, tmp_path):
    m, _ = reference
    out = tmp_path / "run"
    run(m, out, control=rt.Control(out / "STOP", None, stop_at_step=4))
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock = out / "train.lock"
    lock.write_text(json.dumps(dict(pid=dead.pid, host=socket.gethostname(), process_create_time=None)))
    with pytest.raises(rt.ResumeError, match="stale lock"):
        run(m, out, resume=True)
    assert lock.exists()
    import psutil
    lock.write_text(json.dumps(dict(pid=psutil.Process().pid, host=socket.gethostname(),
                                    process_create_time=psutil.Process().create_time())))
    with pytest.raises(rt.ResumeError, match="live process"):
        run(m, out, resume=True)
    lock.write_text(json.dumps(dict(pid=dead.pid, host=socket.gethostname(), process_create_time=None)))
    assert run(m, out, resume=True, recover_stale_lock=True)["state"] == "completed"
    assert len(list((out / "_stale_locks").iterdir())) == 1 and not lock.exists()


def test_fresh_start_refuses_existing_run_and_resume_needs_identity(reference, tmp_path):
    m, ref = reference
    with pytest.raises(rt.ResumeError, match="not empty"):
        run(m, ref)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "step_0001000.pt").write_bytes(b"")
    with pytest.raises(rt.ResumeError, match="run_identity"):
        run(m, legacy_dir, resume=True)


def test_legacy_checkpoints_are_classified_as_warm_start_only(tmp_path, reference):
    _, ref = reference
    legacy = tmp_path / "legacy_step.pt"
    payload = load(ref / "step_0000009.pt")
    torch.save({k: payload[k] for k in ("model", "optimizer", "scheduler", "step", "best", "config")}, legacy)
    info = rt.classify_checkpoint(legacy)
    assert info["kind"] == "legacy_optimizer_state" and "RNG states" in info["missing_for_exact_resume"]
    assert rt.classify_checkpoint(ref / "step_0000009.pt")["kind"] == "exact_resume_state"
    assert rt.classify_checkpoint(ref / "best.pt")["kind"] == "weights_only"


def test_warm_start_is_a_new_run_with_recorded_parent(reference, tmp_path):
    m, ref = reference
    parent = ref / "best.pt"
    out = tmp_path / "warm"
    result = run(m, out, warm_start_from=str(parent))
    lineage = json.loads((out / "lineage.json").read_text())
    assert result["state"] == "completed" and lineage["initialization"] == "warm_start"
    assert lineage["warm_start_parent"]["sha256"] == rt.sha256_file(parent)
    assert lineage["run_id"] != json.loads((ref / "lineage.json").read_text())["run_id"]
    with pytest.raises(rt.ResumeError, match="choose one"):
        run(m, tmp_path / "x", resume=True, warm_start_from=str(parent))


def commercial_receipt(manifest: Path) -> Path:
    from pipeline.commercial_preflight import decoder_dependencies
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    evidence = manifest.parent / "licence.txt"
    evidence.write_text("test licence evidence\n")
    receipt = dict(schema_version=2, scope="internal-pilot", release_authorized=False,
                   manifest_sha256=rt.sha256_file(manifest), reviewed_by="test", reviewed_at="2026-10-21",
                   sources={"src": dict(commercial_training=True, weight_distribution=False,
                                        evidence_path=evidence.name, evidence_sha256=rt.sha256_file(evidence))},
                   decoder_dependencies={str(p): rt.sha256_file(p) for p in decoder_dependencies(rows)})
    path = manifest.parent / "review.json"
    path.write_text(json.dumps(receipt))
    return path


def test_commercial_resume_revalidates_evidence_and_media(tmp_path):
    m = corpus(tmp_path / "c")
    receipt = commercial_receipt(m)
    out = tmp_path / "run"
    first = run(m, out, "--commercial-evidence", str(receipt),
                control=rt.Control(out / "STOP", None, stop_at_step=4))
    assert first["state"] == "stopped_resumable"
    assert load(out / "resume" / "slot_a.pt")["config"]["training_use"] == "internal-pilot"
    media = Path(json.loads(m.read_text().splitlines()[0])["sdr_path"])
    original = media.read_bytes()
    media.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="content hash mismatch"):
        run(m, out, "--commercial-evidence", str(receipt), resume=True)
    media.write_bytes(original)
    with pytest.raises(rt.ResumeError, match="commercial warm start refused"):
        run(m, tmp_path / "w", "--commercial-evidence", str(receipt), warm_start_from=str(out / "best.pt"))
    done = run(m, out, "--commercial-evidence", str(receipt), resume=True)
    assert done["state"] == "completed" and done["release_authorized"] is False


def test_unsupported_modes_refused(reference, tmp_path):
    m, _ = reference
    with pytest.raises(rt.ResumeError, match="workers 0"):
        run(m, tmp_path / "w", "--workers", "2")
    with pytest.raises(rt.ResumeError, match="legacy flags"):
        run(m, tmp_path / "w", "--init-checkpoint", "x.pt")
