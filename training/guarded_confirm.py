"""Confirm a validation-selected dark guard on data that did NOT select it.

No unused production footage exists (owner, 22 Oct 2026), so confirmation is
pre-registered on the independent sets already frozen, with criteria fixed in
this file before any guarded score exists:

  register  freezes the guard setting chosen by training/dark_guard_eval.py on
            VALIDATION (summary.json `selected`) plus these criteria.
  sihdr     SI-HDR protocol v1.1, unchanged code (training/sihdr_eval.py), with
            three predictions per image: guarded model, unguarded model, baseline.
  tos       Tears of Steel kit v1 (unchanged kit and code), guarded model vs baseline.
  mataha    reserved-test exports re-scored with the guard: DIAGNOSTIC ONLY. That
            set revealed the defect, so it cannot confirm the fix.

Pre-registered criteria
  C1 SI-HDR, each clip level: guarded A1 and A3 hold (lower CI >= -0.25 dB full / unsaturated).
  C2 SI-HDR night groups (g09_city_night, g13_winter_night_lights, g15_home_interior_night),
     each clip level: guarded mean full-frame gain vs baseline >= -0.1 dB.
  C3 SI-HDR, each clip level: guarded saturated-region gain is no more than 0.1 dB below the
     unguarded model's (the guard must not cost highlight expansion).
  C4 ToS 03_2a (dim interior), clean: mean PU gain vs baseline >= -0.1 dB; T1 and T3 still pass
     in every section and condition.
A pass is "validation-selected guard confirmed on independent simulated/synthetic-SDR sets".
It is not a real-SDR, display-review or release result.

    python training/guarded_confirm.py register --selection <dark_guard_v1\\summary.json> --out <prereg.json>
    python training/guarded_confirm.py sihdr --prereg <prereg.json> --protocol <sihdr_protocol_v1_1.json> \
        --root <SI-HDR root> --checkpoint <ckpt> --out <new dir>
    python training/guarded_confirm.py tos --prereg <prereg.json> --kit <tos_temporal_kit_v1> --checkpoint <ckpt> --out <new dir>
    python training/guarded_confirm.py mataha --prereg <prereg.json> --run <test_run02_best_step16200> --out <new json>
    python training/guarded_confirm.py verdict --prereg <prereg.json> --sihdr <dir> --tos <dir> --out <new json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NIGHT_GROUPS = ("g09_city_night", "g13_winter_night_lights", "g15_home_interior_night")
NETWORK_NITS = 10000.0
CRITERIA = {
    "C1": "SI-HDR each clip level: guarded A1 and A3 hold (lower CI >= -0.25 dB full / unsaturated)",
    "C2": "SI-HDR night groups g09/g13/g15, each clip level: guarded mean full-frame gain vs baseline >= -0.1 dB",
    "C3": "SI-HDR each clip level: guarded saturated-region gain no more than 0.1 dB below the unguarded model",
    "C4": "ToS 03_2a clean: mean PU gain vs baseline >= -0.1 dB; T1 and T3 pass in every section and condition",
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(selection: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("pre-registration exists; never rewritten")
    summary = json.loads(selection.read_text(encoding="utf-8"))
    chosen = summary.get("selected")
    if not chosen:
        raise ValueError("dark_guard_eval selected no setting: the guard failed on validation; go to the training fix")
    setting = next(s for s in summary["grid"] if f"lo{s['lo']}_hi{s['hi']}_f{s['feather_px']}" == chosen)
    prereg = {"registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "guard": setting, "label": chosen,
              "selected_on": "validation kit only", "selection_summary": str(selection),
              "selection_summary_sha256": sha(selection), "checkpoint_sha256": summary["checkpoint_sha256"],
              "criteria": CRITERIA,
              "code_sha256": {"training/guarded_confirm.py": sha(Path(__file__)), "rudra/dark_guard.py": sha(ROOT / "rudra/dark_guard.py")},
              "release_authorized": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    return prereg


def load_prereg(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    for name, h in p["code_sha256"].items():
        if sha(ROOT / name) != h:
            raise ValueError(f"{name} changed after pre-registration")
    return p


class GuardedRunner:
    """Wraps sihdr_eval.Runner: adds the guarded prediction, keeps the unguarded one."""

    def __init__(self, inner, guard: dict):
        self.inner, self.guard = inner, guard

    def predict(self, input_u8):
        from rudra.dark_guard import apply_dark_guard
        out = self.inner.predict(input_u8)
        raw, base = out["model"], out["baseline_bt1886"]
        guarded = apply_dark_guard(np.maximum(raw, 0) * NETWORK_NITS, np.maximum(base, 0) * NETWORK_NITS,
                                   **self.guard) / NETWORK_NITS
        return {"model": guarded.astype(np.float32), "unguarded": raw, "baseline_bt1886": base}


def run_sihdr(prereg: Path, protocol: Path, root: Path, checkpoint: Path, out: Path, runner=None) -> dict:
    from training import sihdr_eval
    p = load_prereg(prereg)
    if sha(checkpoint) != p["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the one the guard was selected for")
    runner = GuardedRunner(runner or sihdr_eval.Runner(checkpoint), p["guard"])
    return sihdr_eval.evaluate(protocol, root, checkpoint, out, runner=runner)


def tos_predictor(checkpoint: Path, guard: dict | None):
    """Mirror of tos_temporal_eval.evaluate's default predictor, plus the guard."""
    import torch
    from rudra.dark_guard import apply_dark_guard
    from rudra.sdr2hdr import sdr_to_baseline_hdr
    from rudra.source_baseline import baseline_curve_index, correction_log2, knots_table
    from training.export_bench_pairs import degrade_like_eval, load_model
    from training.infer_sdr2hdr import predict_image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)

    def predictor(sdr01, condition, index):
        x = torch.from_numpy(sdr01).permute(2, 0, 1)[None]
        if condition == "hard":
            x = degrade_like_eval(x[0], index)[None]
        x = x.to(device)
        with torch.inference_mode():
            curve = baseline_curve_index("bt1886") if getattr(model, "baseline_source", False) else None
            y = predict_image(model, x, preserve_outside=True, tile_size=512, overlap=64, bf16=False, baseline_curve=curve)
            ids = torch.tensor([baseline_curve_index("bt1886")], device=device)
            b = sdr_to_baseline_hdr(x.float(), model.corpus_ev) * torch.exp2(correction_log2(x.float(), ids, knots_table(device)))
        n = lambda t: t[0].permute(1, 2, 0).float().cpu().numpy() * NETWORK_NITS
        y, b = n(y), n(b)
        if guard:
            y = apply_dark_guard(np.maximum(y, 0), np.maximum(b, 0), **guard).astype(np.float32)
        return y, b
    return predictor


def run_tos(prereg: Path, kit: Path, checkpoint: Path, out: Path, predictor=None) -> dict:
    from training import tos_temporal_eval
    p = load_prereg(prereg)
    if sha(checkpoint) != p["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the one the guard was selected for")
    return tos_temporal_eval.evaluate(kit, checkpoint, out, predictor=predictor or tos_predictor(checkpoint, p["guard"]))


def run_mataha(prereg: Path, run: Path, out: Path) -> dict:
    from training.dark_guard_eval import label, score_condition
    p = load_prereg(prereg)
    if out.exists():
        raise ValueError("output exists")
    res = {"DIAGNOSTIC_ONLY": "this reserved-test set revealed the defect; it cannot confirm the fix",
           "guard": p["guard"], "conditions": {}, "release_authorized": False}
    for cond in ("clean", "hard"):
        rows = score_condition(run / cond, [p["guard"]])
        g = [r["guarded"][label(p["guard"])] - r["baseline"] for r in rows]
        u = [r["model"] - r["baseline"] for r in rows]
        res["conditions"][cond] = {"frames": len(rows), "unguarded_mean_gain_db": float(np.mean(u)),
                                   "guarded_mean_gain_db": float(np.mean(g))}
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def verdict(prereg: Path, sihdr_dir: Path, tos_dir: Path) -> dict:
    from training.sihdr_eval import CLIP_LEVELS
    p = load_prereg(prereg)
    summ = json.loads((sihdr_dir / "summary.json").read_text())
    rows = [json.loads(l) for l in (sihdr_dir / "per_image.jsonl").read_text().splitlines()]
    proto = json.loads(Path(summ["protocol"]).read_text()) if Path(summ["protocol"]).is_file() else None
    groups = proto["groups"] if proto else {}
    night = {i for g in NIGHT_GROUPS for i in groups.get(g, [])}
    d = lambda r, a, b, reg: r["methods"][a]["pu21_psnr_db"][reg] - r["methods"][b]["pu21_psnr_db"][reg]
    res = {"C1": {}, "C2": {}, "C3": {}}
    for clip in CLIP_LEVELS:
        v = summ["verdict"][clip]
        res["C1"][clip] = bool(v["A1_full_frame_no_regression"] and v["A3_unsaturated_preservation"])
        nr = [d(r, "model", "baseline_bt1886", "full") for r in rows if r["clip"] == clip and r["image"] in night]
        res["C2"][clip] = {"images": len(nr), "mean_gain_db": float(np.mean(nr)) if nr else None,
                           "pass": bool(nr) and float(np.mean(nr)) >= -0.1}
        sat = [d(r, "model", "unguarded", "saturated") for r in rows if r["clip"] == clip
               and math.isfinite(r["methods"]["model"]["pu21_psnr_db"]["saturated"])]
        res["C3"][clip] = {"images": len(sat), "guarded_minus_unguarded_db": float(np.mean(sat)) if sat else None,
                           "pass": bool(sat) and float(np.mean(sat)) >= -0.1}
    tos = json.loads((tos_dir / "temporal_report.json").read_text())
    c4 = tos["sections"]["03_2a"]["clean"]["pu21_psnr_gain_db_mean"]
    t13 = all(v["verdict"]["T1"] and v["verdict"]["T3"] for s in tos["sections"].values() for v in s.values())
    res["C4"] = {"tos_03_2a_clean_gain_db": c4, "T1_T3_all": t13, "pass": c4 >= -0.1 and t13}
    passed = (all(res["C1"].values()) and all(x["pass"] for x in res["C2"].values())
              and all(x["pass"] for x in res["C3"].values()) and res["C4"]["pass"])
    return {"criteria": res, "confirmed": passed, "guard": p["guard"],
            "meaning": "independent simulated/synthetic-SDR confirmation only; not real-SDR, display or release",
            "release_authorized": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register"); r.add_argument("--selection", type=Path, required=True); r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sihdr")
    for k in ("prereg", "protocol", "root", "checkpoint", "out"):
        s.add_argument(f"--{k}", type=Path, required=True)
    t = sub.add_parser("tos")
    for k in ("prereg", "kit", "checkpoint", "out"):
        t.add_argument(f"--{k}", type=Path, required=True)
    m = sub.add_parser("mataha")
    for k in ("prereg", "run", "out"):
        m.add_argument(f"--{k}", type=Path, required=True)
    v = sub.add_parser("verdict")
    for k in ("prereg", "sihdr", "tos", "out"):
        v.add_argument(f"--{k}", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "register":
        res = register(a.selection, a.out)
    elif a.cmd == "sihdr":
        res = run_sihdr(a.prereg, a.protocol, a.root, a.checkpoint, a.out)["verdict"]
    elif a.cmd == "tos":
        res = {s: {c: v["verdict"] for c, v in d.items()} for s, d in run_tos(a.prereg, a.kit, a.checkpoint, a.out)["sections"].items()}
    elif a.cmd == "mataha":
        res = run_mataha(a.prereg, a.run, a.out)
    else:
        if a.out.exists():
            raise SystemExit("output exists")
        res = verdict(a.prereg, a.sihdr, a.tos)
        a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
