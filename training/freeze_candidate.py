"""Freeze ONE selected checkpoint for final testing: hash, settings, software, lineage.

Run only after shortlist selection (production_shortlist_eval summary.json). The
freeze record is what the reserved test set, the independent/temporal kits and
the display review must all cite; any later byte change to the checkpoint or
pinned inference code invalidates it. It never copies, promotes or publishes
weights, and release_authorized is always false.

    python training/freeze_candidate.py --checkpoint <ckpt> --selection <summary.json> \
        --label <label> --out <new freeze.json>
    python training/freeze_candidate.py verify --freeze <freeze.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CODE = ("rudra/sdr2hdr.py", "rudra/inference.py", "rudra/source_baseline.py", "rudra/radiometry.py",
                  "rudra/video.py", "training/infer_sdr2hdr.py", "training/export_bench_pairs.py")
INFERENCE_SETTINGS = {"precision": "fp32 for evaluation (bf16 allowed in production only after a matched check)",
                      "tile_size": 512, "tile_overlap": 64, "preserve_outside": True,
                      "baseline_curve": "bt1886 (app default Source picker)",
                      "input": "8-bit SDR -> canonical sRGB code in BT.2020 (rudra.video.Predictor)",
                      "output": "linear BT.2020, network units = nits/10000, clamp 4.0 (40,000 nits)"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def software() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "numpy", "cv2", "OpenEXR"):
        try:
            out[mod] = str(__import__(mod).__version__)
        except Exception:  # noqa: BLE001
            out[mod] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    return out


def freeze(checkpoint: Path, selection: Path, label: str, out: Path, owner_decision: str | None = None) -> dict:
    import torch
    if out.exists():
        raise ValueError(f"{out} exists; a freeze record is never rewritten")
    summary = json.loads(selection.read_text(encoding="utf-8"))
    entry = summary.get("candidates", {}).get(label)
    if entry is None:
        raise ValueError(f"{label} is not a shortlisted candidate in {selection}")
    proposed = summary.get("selection", {}).get("proposed_for_next_stage")
    if proposed != label:
        if not owner_decision or len(owner_decision.strip()) < 20:
            raise ValueError(f"{label} is not the frozen rule's proposal ({proposed}); pass --owner-decision "
                             "with the reason to record an explicit, visible override")
        if label not in summary.get("selection", {}).get("ranking", []):
            raise ValueError(f"{label} did not pass the frozen gates; an override cannot select it")
    digest = sha256_file(checkpoint)
    if digest != entry["sha256"]:
        raise ValueError("checkpoint bytes differ from the evaluated candidate")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    record = {
        "label": label, "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "step": payload.get("step"),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_summary": str(selection), "selection_summary_sha256": sha256_file(selection),
        "validation": {"clean": entry["clean"], "hard": entry["hard"]},
        "selection": {"rule_proposal": proposed, "frozen": label,
                      "owner_override": owner_decision if proposed != label else None},
        "architecture": {k: config.get(k) for k in ("architecture", "base_channels", "baseline_source", "curve_head",
                                                    "gate_conditioning", "source_curve", "corpus_ev")},
        "parameters": int(sum(v.numel() for v in payload["model"].values())),
        "training": {"manifest_sha256": config.get("manifest_sha256"), "training_use": config.get("training_use"),
                     "initialization": (config.get("commercial_provenance") or {}).get("initialization", "unknown"),
                     "provenance_evidence_sha256": (config.get("commercial_provenance") or {}).get("evidence_sha256"),
                     "runtime": config.get("runtime")},
        "inference_settings": INFERENCE_SETTINGS,
        "inference_code_sha256": {p: code_sha(ROOT / p) for p in INFERENCE_CODE if (ROOT / p).is_file()},
        "software": software(),
        "next_required": ["reserved test set (Mataha, 260 frames) on this exact hash",
                          "SI-HDR and Tears of Steel kits on this exact hash",
                          "calibrated HDR display + motion review", "rights decisions for owner footage and weights"],
        "release_authorized": False, "promotion": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def verify(freeze_path: Path) -> dict:
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    problems = []
    if sha256_file(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
        problems.append("checkpoint bytes changed")
    for p, h in record["inference_code_sha256"].items():
        if not (ROOT / p).is_file() or code_sha(ROOT / p) != h:
            problems.append(f"inference code changed: {p}")
    return {"freeze": str(freeze_path), "ok": not problems, "problems": problems, "release_authorized": False}


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["verify"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("--freeze", type=Path, required=True)
        result = verify(ap.parse_args(argv[1:]).freeze)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--owner-decision", help="required when freezing a gate-passing candidate other than the rule's proposal")
    a = ap.parse_args(argv)
    print(json.dumps(freeze(a.checkpoint, a.selection, a.label, a.out, a.owner_decision), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
