"""tools/holdout_kit.py: register, seal and verify a confirmatory holdout."""
import json

from tools.holdout_kit import main, stratified_pick


def _setup(tmp_path, n_scenes=12):
    data = tmp_path / "footage"
    data.mkdir()
    rows = []
    for i in range(n_scenes):
        for f in range(2):
            p = data / f"shot{i:02d}_f{f}.exr"
            p.write_bytes(f"pixels {i} {f}".encode())
            rows.append({"id": f"s{i}f{f}", "scene_id": f"scene{i:02d}", "path": str(p),
                         "project": "A" if i < 6 else "B", "camera": "RED" if i % 2 else "ARRI"})
    cand = tmp_path / "rows.jsonl"
    cand.write_text("\n".join(json.dumps(r) for r in rows))
    train = tmp_path / "training"
    train.mkdir()
    # Training already used scene00 by id, shot01 by file name, and shot02's exact file by hash.
    import hashlib
    h = hashlib.sha256((data / "shot02_f0.exr").read_bytes()).hexdigest()
    (train / "manifest.jsonl").write_text(json.dumps({"scene_id": "scene00"}) + "\n" +
                                          json.dumps({"frame": "G:/x/shot01_f1.exr"}) + "\n")
    (train / "report.json").write_text(json.dumps({"rows": [{"sha256": h}]}))
    return cand, train, data


def test_register_excludes_everything_training_saw(tmp_path):
    cand, train, _ = _setup(tmp_path)
    out = tmp_path / "holdout.json"
    rc = main(["register", "--candidates", str(cand), "--exclude-roots", str(train), "--n", "8",
               "--strata", "project", "camera", "--out", str(out)])
    assert rc == 0
    rec = json.loads(out.read_text())
    ids = {r["id"] for r in rec["rows"]}
    reasons = {e["id"]: e["reason"] for e in rec["excluded"]}
    assert reasons["s0f0"] == "scene_id used" and reasons["s0f1"] == "scene_id used"
    assert reasons["s1f1"] == "file name used"
    assert reasons["s2f0"] == "file hash used"
    assert not ids & set(reasons)
    assert rec["counts"]["picked"] == 8
    assert rec["counts"]["scenes"] == 8                    # one row per scene first
    assert {r["project"] for r in rec["rows"]} == {"A", "B"}
    assert all(len(r["sha256"]) == 64 for r in rec["rows"])


def test_registration_is_never_rewritten_and_is_deterministic(tmp_path):
    cand, train, _ = _setup(tmp_path)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    args = ["register", "--candidates", str(cand), "--exclude-roots", str(train), "--n", "6", "--seed", "7"]
    assert main(args + ["--out", str(a)]) == 0
    assert main(args + ["--out", str(a)]) == 1
    assert main(args + ["--out", str(b)]) == 0
    pick = lambda p: [r["id"] for r in json.loads(p.read_text())["rows"]]
    assert pick(a) == pick(b)


def test_verify_catches_edits_changed_files_and_leaks(tmp_path):
    cand, train, data = _setup(tmp_path)
    out = tmp_path / "h.json"
    main(["register", "--candidates", str(cand), "--exclude-roots", str(train), "--n", "6", "--out", str(out)])
    assert main(["verify", str(out)]) == 0
    rec = json.loads(out.read_text())
    # a frame changed on disk
    first = rec["rows"][0]
    from pathlib import Path
    Path(first["path"]).write_bytes(b"regraded")
    assert main(["verify", str(out)]) == 1
    Path(first["path"]).write_bytes(f"pixels {int(first['scene_id'][5:])} {first['id'][-1]}".encode())
    assert main(["verify", str(out)]) == 0
    # the registration edited after sealing
    rec["selection"]["n"] = 5
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(rec))
    assert main(["verify", str(edited)]) == 1
    # a registered scene later shows up in training
    (train / "later.jsonl").write_text(json.dumps({"scene_id": first["scene_id"]}))
    assert main(["verify", str(out), "--exclude-roots", str(train)]) == 1


def test_stratified_pick_balances_and_prefers_new_scenes():
    rows = [{"id": f"{p}{i}", "scene_id": f"{p}{i // 2}", "project": p} for p in "AB" for i in range(10)]
    pick = stratified_pick(rows, 6, ["project"], 1)
    assert sum(r["project"] == "A" for r in pick) == 3
    assert len({r["scene_id"] for r in pick}) == 6
