"""RUDRA 1.1 validation (the gap audit's priority 6, 5 Nov 2026).

Runs the RUDRA 1.1 features on real footage through `rudra-native` and checks, per shot,
what docs/RUDRA_1_1_SWITCHHDR_GAP_AUDIT.md promises. Every check is PASS / FAIL against a
stated threshold, INFO (measured, no threshold), or MANUAL (a person looks at the pictures).

    python tools/validate_rudra_1_1.py --manifest shots.json --out validation/
    python tools/validate_rudra_1_1.py --selftest --out validation_selftest/     # synthetic shots, no footage
    ... [--cli native/build/.../rudra-native] [--frames 24] [--peak 1000]
        [--genfill --python .venv/Scripts/python.exe --base <Wan2.1-VACE-1.3B> --lora <rudra_genfill.safetensors>
         --prompt "warm sunset sun with a soft halo"]

shots.json: {"shots": [{"name": "SKY01", "path": "D:/plates/sky01.mov", "category": "sky"}, ...]}
  path: a still, a movie (the first --frames frames are used) or a folder of stills.
  category: sky | sunset | window | reflection | fire | other (chooses the region tested).

Checks (per shot; masters are analytic, float32 EXR, so nothing is lost to half rounding):
  C1 preserved   every pixel outside the luminance region (its rudra.region matte is 0) is bit
                 for bit the master with the reconstruction switched off (both with the anchor,
                 chroma carry and grain settle off: those stages read neighbours / the frame)
  C2 mattes      every pixel where the region master differs from that baseline has rudra.changed 1
                 (FAIL otherwise); pixels marked changed whose difference the EXR's float32 rounds
                 away are counted (INFO: changed is exact at the reconstruction's own precision)
  C3 brackets    the -2/0/+2 EV brackets of the EXR as written match the preview's: 0 codes
  C4 chain       INFO: with the default master chain on, how far pixels outside the region move
                 (stops): the anchor's frame-wide hold and the spatial stages reach them
  C5 temporal    moving shots: frame-to-frame change the region adds beyond the SDR's own (EV),
                 PASS under 0.02 EV mean (visible pumping starts about there); and the region's
                 coverage from frame to frame (INFO)
  C6 PQ / HLG    the masters delivered as HDR10 and HLG (rudra-native deliver), tags read back
                 with ffprobe, frame 0 decoded and turned back into light:
                   fidelity  delivered at a peak above the master's own (no shoulder): luminance
                             error in stops on near-neutral pixels (max/min channel < 2), the
                             transfer and codec themselves (PASS: tags right, median < 0.02, p95 <
                             0.1); by saturation band as INFO: 4:2:0 Y'CbCr is not constant
                             luminance, so strongly saturated light loses luminance in any HDR10/HLG
                             file, more the more saturated
                   preview   delivered at --peak (the deliverable): how far the SDR range (0.5 to
                             203 nits) moved from what the viewer previewed (median stops) and the
                             0 EV bracket's code difference (INFO: the delivery shoulder rolls the
                             whole frame into the peak, which the viewer's preview does not show)
  C7 AI fill     with --genfill: the fill run with the region and a prompt, and without the
                 prompt: no 8-bit code of the SDR view changed, nothing generated outside the
                 region, pixels outside the generated area bit-identical; prompt adherence is
                 MANUAL (contact sheet), with a CLIP score when open_clip is installed
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WHITE = 203.0
LUMA2020 = np.array([0.2627, 0.6780, 0.0593])
M2020_TO_709 = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]])
REGIONS = {   # what each category's test region isolates
    "sky": {"highlight_from": 0.80, "highlight_softness": 0.08, "shadow_strength": 0.0},
    "sunset": {"highlight_from": 0.85, "highlight_softness": 0.06, "shadow_strength": 0.0},
    "window": {"highlight_from": 0.88, "highlight_softness": 0.04, "shadow_strength": 0.0},
    "reflection": {"band": True, "band_low": 0.70, "band_high": 1.0, "band_softness": 0.04, "shadow_strength": 0.0},
    "fire": {"highlight_from": 0.75, "highlight_softness": 0.10, "shadow_strength": 0.0},
    "other": {"highlight_from": 0.85, "highlight_softness": 0.06, "shadow_below": 0.05, "shadow_softness": 0.03},
}
OFF = {"highlight_strength": 0.0, "shadow_strength": 0.0}   # a region that admits nothing: the baseline
STRICT = {"analytic": True, "anchor": False, "carry_chroma": False, "settle_grain": False, "exr_half": False,
          "container": "linear"}


# ---------------------------------------------------------------- io
def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-800:]}")
    return r.stdout


def read_exr(path) -> dict:
    import OpenEXR
    with OpenEXR.File(str(path), separate_channels=True) as f:
        return {k: np.array(c.pixels, dtype=np.float32) for k, c in f.channels().items()}


def rgb(ch) -> np.ndarray:
    return np.stack([ch["R"], ch["G"], ch["B"]], -1)


def master(cli, image, out, params):
    run([cli, "master", "analytic", image, "--out", out, "--params", json.dumps(params)])
    return json.loads(Path(out).with_suffix(".json").read_text())


def srgb_oetf(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def bracket_from_nits2020(nits, ev=0.0):
    """The 0 EV SDR bracket (core exposure_bracket) of Rec.2020 nits, as 8-bit codes."""
    lin709 = np.einsum("ij,hwj->hwi", M2020_TO_709, nits) / WHITE * 2.0 ** ev
    return np.rint(srgb_oetf(lin709) * 255.0).astype(np.int16)


def pq_eotf(code):
    m1, m2, c1, c2, c3 = 2610 / 16384, 2523 / 4096 * 128, 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    p = np.power(np.clip(code, 0, 1), 1 / m2)
    return 10000.0 * np.power(np.maximum(p - c1, 0) / (c2 - c3 * p), 1 / m1)


def hlg_display(code, peak):
    a, b, c = 0.17883277, 1 - 4 * 0.17883277, 0.5 - 0.17883277 * math.log(4 * 0.17883277)
    code = np.clip(code, 0, 1)
    s = np.where(code <= 0.5, code * code / 3.0, (np.exp((code - c) / a) + b) / 12.0)
    gamma = 1.2 + 0.42 * math.log10(peak / 1000.0)
    ys = np.maximum(s @ LUMA2020, 1e-12)
    return peak * (ys ** (gamma - 1.0))[..., None] * s


def decode_movie_frame(movie, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(movie), "-frames:v", "1", "-vf",
                          "scale=in_color_matrix=bt2020:in_range=tv:out_range=pc,format=rgb48le",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<u2").reshape(h, w, 3).astype(np.float64) / 65535.0


def probe_tags(movie):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=color_transfer,color_primaries,color_space", "-of", "json", movie])
    return json.loads(out)["streams"][0]


# ---------------------------------------------------------------- shots
def frames_of(shot, work: Path, n: int) -> list[Path]:
    p = Path(shot["path"])
    if p.is_dir():
        ims = sorted(x for x in p.iterdir() if x.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dpx"))
        return ims[:n]
    if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dpx"):
        return [p]
    d = work / "frames"
    d.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-i", p, "-frames:v", n, "-pix_fmt", "rgb48be", d / "f_%04d.png"])
    return sorted(d.glob("f_*.png"))


def selftest_shots(root: Path) -> list[dict]:
    """Synthetic shots for checking the harness itself (not RUDRA's quality)."""
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    h, w = 136, 240   # even: 4:2:0 HEVC needs it
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(1)
    shots = []

    def save(name, img):
        p = root / f"{name}.png"
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(p)
        return p
    sky = np.stack([140 + 115 * (1 - yy / h), 170 + 85 * (1 - yy / h), 255 + 0 * yy], -1)
    sky[: h // 4] = 255
    shots.append({"name": "SELF_SKY", "path": str(save("sky", sky + rng.normal(0, 2, sky.shape))), "category": "sky"})
    sun = np.stack([255 - 60 * yy / h, 120 + 60 * (1 - yy / h), 40 + 0 * yy], -1)
    sun[(yy - h * 0.45) ** 2 + (xx - w * 0.6) ** 2 < 18 ** 2] = 255
    shots.append({"name": "SELF_SUNSET", "path": str(save("sunset", sun)), "category": "sunset"})
    room = np.full((h, w, 3), 35.0) + 10 * xx[..., None] / w
    room[20:80, 140:210] = 255
    shots.append({"name": "SELF_WINDOW", "path": str(save("window", room)), "category": "window"})
    refl = np.full((h, w, 3), 60.0)
    for cy, cx in ((40, 50), (90, 120), (60, 200)):
        refl[(yy - cy) ** 2 + (xx - cx) ** 2 < 25] = 255
    shots.append({"name": "SELF_REFLECTION", "path": str(save("reflection", refl)), "category": "reflection"})
    fire_dir = root / "fire"
    fire_dir.mkdir(exist_ok=True)
    for t in range(12):   # a moving, flickering flame: the temporal checks
        f = np.full((h, w, 3), 20.0)
        cx, r = 60 + 8 * t, 22 + 4 * math.sin(t * 1.7)
        d2 = (yy - 80) ** 2 + (xx - cx) ** 2
        glow = np.exp(-d2 / (2 * (r * 1.6) ** 2))[..., None] * np.array([255, 150, 40])
        f = f + glow
        f[d2 < r * r] = 255
        Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).save(fire_dir / f"fire_{t:04d}.png")
    shots.append({"name": "SELF_FIRE_MOVING", "path": str(fire_dir), "category": "fire"})
    return shots


# ---------------------------------------------------------------- checks
def verdict(ok, **kw):
    return {"result": "PASS" if ok else "FAIL", **kw}


def validate_shot(shot, args, out: Path) -> dict:
    work = out / shot["name"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    frames = frames_of(shot, work, args.frames)
    if not frames:
        return {"shot": shot["name"], "error": "no frames"}
    region = {**REGIONS.get(shot.get("category", "other"), REGIONS["other"])}
    rep = {"shot": shot["name"], "category": shot.get("category", "other"), "frames": len(frames), "region": region,
           "checks": {}}
    c1_bad = c2_bad = c2_rounded = 0
    c3_worst = 0
    c4 = []
    cover, sdr_mean, hdr_mean = [], [], []
    regdir, defdir = work / "region", work / "defaults"
    for i, f in enumerate(frames):
        base_p, reg_p = work / "base" / f"f_{i:04d}.exr", regdir / f"f_{i:04d}.exr"
        for p in (base_p, reg_p):
            p.parent.mkdir(exist_ok=True)
        master(args.cli, f, base_p, {**STRICT, "reconstruction_region": OFF})
        side = master(args.cli, f, reg_p, {**STRICT, "reconstruction_region": region, "mattes": True, "brackets": True})
        a, b = rgb(read_exr(base_p)), rgb(read_exr(reg_p))
        m = read_exr(reg_p.with_suffix(".mattes.exr"))
        outside = m["rudra.region"] <= 0
        c1_bad += int((a[outside] != b[outside]).any(-1).sum())
        differs = (a != b).any(-1)
        marked = m["rudra.changed"] > 0.5
        c2_bad += int((differs & ~marked).sum())
        c2_rounded += int((marked & ~differs).sum())
        c3_worst = max(c3_worst, int(side["brackets"]["preview_vs_export"]["max_code_delta"]))
        # C4: the default chain, region vs off, outside the region
        dp, dr = defdir / f"off_{i:04d}.exr", defdir / f"reg_{i:04d}.exr"
        dp.parent.mkdir(exist_ok=True)
        master(args.cli, f, dp, {"analytic": True, "exr_half": False, "container": "linear", "reconstruction_region": OFF})
        master(args.cli, f, dr, {"analytic": True, "exr_half": False, "container": "linear", "reconstruction_region": region})
        da, db = rgb(read_exr(dp)), rgb(read_exr(dr))
        ya, yb = np.maximum(da @ LUMA2020, 1e-6), np.maximum(db @ LUMA2020, 1e-6)
        st = np.abs(np.log2(yb / ya))[outside]
        if st.size:
            c4.append((float(np.max(st)), float(np.percentile(st, 99))))
        cover.append(float((~outside).mean()))
        sdr = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(f).convert("RGB"), np.float64) / 255.0
        sdr_mean.append(float(np.log2(max(np.mean(sdr ** 2.2 @ LUMA2020), 1e-6))))
        hdr_mean.append((float(np.log2(max(np.mean(a @ LUMA2020), 1e-9))), float(np.log2(max(np.mean(b @ LUMA2020), 1e-9)))))
    ck = rep["checks"]
    ck["C1_preserved"] = verdict(c1_bad == 0, differing_pixels_outside_region=c1_bad)
    ck["C2_mattes_match"] = verdict(c2_bad == 0, differing_but_unmarked=c2_bad,
                                    marked_but_rounded_equal_in_exr=c2_rounded)
    ck["C3_brackets_preview_vs_export"] = verdict(c3_worst == 0, max_code_delta=c3_worst)
    ck["C4_default_chain_outside_region"] = {"result": "INFO",
                                             "max_stops": max((x[0] for x in c4), default=0.0),
                                             "p99_stops": max((x[1] for x in c4), default=0.0),
                                             "why": "anchor hold / chroma / grain read neighbours or the whole frame"}
    if len(frames) > 1:
        d_sdr = np.diff(sdr_mean)
        d_off = np.diff([x[0] for x in hdr_mean])
        d_reg = np.diff([x[1] for x in hdr_mean])
        added = np.abs(d_reg - d_sdr) - np.abs(d_off - d_sdr)   # what the region adds over the plain reconstruction
        ck["C5_temporal"] = verdict(float(np.mean(np.maximum(added, 0))) < 0.02,
                                    region_added_flicker_ev_mean=float(np.mean(np.maximum(added, 0))),
                                    region_added_flicker_ev_max=float(np.max(np.maximum(added, 0))),
                                    region_coverage_step_mean=float(np.mean(np.abs(np.diff(cover)))))
    # C6: PQ and HLG deliveries of the region masters
    h, w = b.shape[:2]
    ref = rgb(read_exr(regdir / "f_0000.exr")) * WHITE                     # Rec.2020 nits, frame 0
    preview0 = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(regdir / "f_0000.ev+0.png").convert("RGB"), np.int16)
    master_peak = float(max((rgb(read_exr(p)) * WHITE).max() for p in sorted(regdir.glob("f_*.exr"))
                            if not p.name.endswith(".mattes.exr")))
    fid_peak = float(min(10000.0, max(1000.0, 1.25 * master_peak)))
    ys_ref = np.maximum(ref @ LUMA2020, 1e-4)
    for target, transfer in (("hdr10", "smpte2084"), ("hlg", "arib-std-b67")):
        res = {}
        try:
            for kind, peak in (("fidelity", fid_peak), ("preview", float(args.peak))):
                stem = work / f"deliver_{target}_{kind}"
                run([args.cli, "deliver", regdir, "--output", stem, "--target", target, "--peak-nits", peak,
                     "--source-space", "rec2020", "--fps", 24])
                movie = next(work.glob(f"deliver_{target}_{kind}.*"))
                light = decode_movie_frame(movie, w, h)
                light = pq_eotf(light) if target == "hdr10" else hlg_display(light, peak)
                ys_got = np.maximum(light @ LUMA2020, 1e-4)
                if kind == "fidelity":
                    tags = probe_tags(movie)
                    e_all = np.abs(np.log2(ys_got / ys_ref))
                    sat = ref.max(-1) / np.maximum(ref.min(-1), 1e-3)
                    lit = ys_ref > 0.5
                    err = e_all[lit & (sat < 2)]
                    med, p95 = (float(np.median(err)), float(np.percentile(err, 95))) if err.size else (0.0, 0.0)
                    bands = {}
                    for lo, hi in ((2, 4), (4, 8), (8, np.inf)):
                        s = lit & (sat >= lo) & (sat < hi)
                        if s.any():
                            bands[f"sat_{lo}_{hi}"] = {"pixels_pct": round(100 * float(s.mean()), 2),
                                                       "median_stops": float(np.median(e_all[s]))}
                    ok_tags = tags.get("color_transfer") == transfer and tags.get("color_primaries") == "bt2020"
                    res["fidelity"] = verdict(ok_tags and err.size > 0 and med < 0.02 and p95 < 0.1, peak_nits=peak,
                                              tags=tags, neutral_pixels=int(err.size), median_stops=med, p95_stops=p95,
                                              saturated_info=bands)
                else:
                    sdr_range = (ys_ref > 0.5) & (ys_ref < WHITE)
                    shift = np.log2(ys_got / ys_ref)[sdr_range]
                    dcode = np.abs(bracket_from_nits2020(light) - preview0).max(-1)
                    res["preview"] = {"result": "INFO", "peak_nits": peak, "master_peak_nits": master_peak,
                                      "sdr_range_shift_stops_median": float(np.median(shift)) if shift.size else 0.0,
                                      "bracket0_codes_p99": float(np.percentile(dcode, 99)), "bracket0_codes_max": int(dcode.max())}
            res["result"] = res["fidelity"]["result"]
        except Exception as e:                                             # noqa: BLE001
            res = {"result": "FAIL", "error": str(e)}
        ck[f"C6_{target}"] = res
    if args.genfill:
        ck["C7_ai_fill"] = check_genfill(frames, region, args, work)
    return rep


def check_genfill(frames, region, args, work: Path) -> dict:
    """The AI fill on ACES masters of the shot, with the region, with and without the prompt."""
    try:
        from rudra.aces_view import ACES2View
    except Exception as e:                                                  # noqa: BLE001
        return {"result": "FAIL", "error": f"rudra.aces_view needs PyOpenColorIO: {e}"}
    src = work / "aces"
    src.mkdir()
    for i, f in enumerate(frames):
        master(args.cli, f, src / f"f_{i:04d}.exr", {"analytic": True, "exr_half": False, "container": "aces"})
    for p in src.glob("*.json"):
        p.unlink()
    runs = {}
    for tag, prompt in (("prompt", args.prompt), ("noprompt", "")):
        out = work / f"aifill_{tag}"
        cmd = [args.python, "-m", "rudra.genfill.run", "--master", src, "--out", out, "--base", args.base, "--lora", args.lora,
               "--region", json.dumps(region), "--fallback", "spikes", "--json"]
        if prompt:
            cmd += ["--prompt-highlight", prompt]
        run(cmd, cwd=REPO)
        runs[tag] = out
    view = ACES2View()
    codes_moved = outside_gen = outside_region = 0
    for f in sorted(src.glob("f_*.exr")):
        m = read_exr(f)
        o = read_exr(runs["prompt"] / f.name)
        from rudra.genfill.run import read_frame
        a, b = read_frame(f)["ap1"], read_frame(runs["prompt"] / f.name)["ap1"]
        ca = np.rint(np.clip(view.forward(a), 0, 1) * 255)
        cb = np.rint(np.clip(view.forward(b), 0, 1) * 255)
        codes_moved += int((ca != cb).any(-1).sum())
        gen = o["rudra.generated"] > 0
        outside_gen += int((rgb(m)[~gen] != rgb(o)[~gen]).any(-1).sum())
        from rudra.delivery.region import fill_region_weight, region_from_dict
        w = fill_region_weight(ca.astype(np.uint8), region_from_dict(region))
        outside_region += int((gen & (w <= 0)).sum())
    sheet = contact_sheet(src, runs, view, work / "aifill_contact_sheet.png")
    res = verdict(codes_moved == 0 and outside_gen == 0 and outside_region == 0, sdr_codes_moved=codes_moved,
                  changed_outside_generated=outside_gen, generated_outside_region=outside_region,
                  prompt_adherence={"result": "MANUAL", "contact_sheet": str(sheet), "prompt": args.prompt})
    try:
        res["prompt_adherence"]["clip_score"] = clip_scores(sheet.parent, runs, view, args.prompt)
    except Exception as e:                                                  # noqa: BLE001
        res["prompt_adherence"]["clip_score"] = f"not measured ({type(e).__name__}: open_clip not installed?)"
    return res


def contact_sheet(src, runs, view, path: Path) -> Path:
    from PIL import Image
    from rudra.genfill.run import read_frame
    rows = []
    for f in sorted(src.glob("f_*.exr"))[:6]:
        row = [np.rint(np.clip(view.forward(read_frame(p)["ap1"] * 2.0 ** -3), 0, 1) * 255).astype(np.uint8)
               for p in (f, runs["noprompt"] / f.name, runs["prompt"] / f.name)]
        rows.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    return path


def clip_scores(_dir, runs, view, prompt):
    import open_clip
    import torch
    from PIL import Image
    from rudra.genfill.run import read_frame
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok([prompt]))
        t = t / t.norm(dim=-1, keepdim=True)
        out = {}
        for tag, d in runs.items():
            s = []
            for f in sorted(d.glob("f_*.exr"))[:6]:
                im = np.rint(np.clip(view.forward(read_frame(f)["ap1"] * 2.0 ** -3), 0, 1) * 255).astype(np.uint8)
                e = model.encode_image(pre(Image.fromarray(im))[None])
                s.append(float((e / e.norm(dim=-1, keepdim=True) @ t.T)[0, 0]))
            out[tag] = float(np.mean(s))
    out["prompt_minus_noprompt"] = out["prompt"] - out["noprompt"]
    return out


# ---------------------------------------------------------------- report
def write_report(reports, out: Path):
    (out / "report.json").write_text(json.dumps(reports, indent=1))
    lines = ["# RUDRA 1.1 validation", "", "| Shot | Category | Frames | " +
             " | ".join(["C1", "C2", "C3", "C4", "C5", "C6 PQ", "C6 HLG", "C7"]) + " |", "|" + "---|" * 11]
    keys = ["C1_preserved", "C2_mattes_match", "C3_brackets_preview_vs_export", "C4_default_chain_outside_region",
            "C5_temporal", "C6_hdr10", "C6_hlg", "C7_ai_fill"]
    fails = 0
    for r in reports:
        if "error" in r:
            lines.append(f"| {r['shot']} | | | {r['error']} |")
            fails += 1
            continue
        cells = []
        for k in keys:
            c = r["checks"].get(k)
            cells.append("-" if c is None else c["result"])
            fails += c is not None and c["result"] == "FAIL"
        lines.append(f"| {r['shot']} | {r['category']} | {r['frames']} | " + " | ".join(cells) + " |")
    lines += ["", f"FAIL count: {fails}. Details per check in report.json; thresholds in the script's docstring."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="shots.json")
    ap.add_argument("--selftest", action="store_true", help="synthetic shots: checks the harness, not the footage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cli", default=os.environ.get("RUDRA_NATIVE", "rudra-native"))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--peak", type=float, default=1000.0)
    ap.add_argument("--genfill", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--base", default="")
    ap.add_argument("--lora", default="")
    ap.add_argument("--prompt", default="warm sunset sun with a soft halo")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.selftest:
        shots = selftest_shots(out / "selftest_shots")
    elif a.manifest:
        shots = json.loads(Path(a.manifest).read_text())["shots"]
    else:
        ap.error("--manifest or --selftest")
    reports = []
    for s in shots:
        print(f"{s['name']} ...", flush=True)
        try:
            reports.append(validate_shot(s, a, out))
        except Exception as e:                                              # noqa: BLE001
            reports.append({"shot": s["name"], "error": f"{type(e).__name__}: {e}"})
        print(json.dumps({k: v.get("result") for k, v in reports[-1].get("checks", {}).items()} or reports[-1]), flush=True)
    fails = write_report(reports, out)
    print((out / "report.md").read_text())
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
