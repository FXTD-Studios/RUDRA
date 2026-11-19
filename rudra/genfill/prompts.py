"""Prompt presets and a suggestion for the AI fill (RUDRA 1.1, 5 Nov 2026).

The fill takes an optional highlight and shadow prompt. Writing one from scratch is guesswork,
and a prompt that is not in the prompt cache loads umT5 (about 11 GB, on the CPU) once to embed
it. So RUDRA offers a short list of PRESETS (their embeddings made once with `precache`, then
instant), and SUGGESTS one by looking at the shot's fully clipped areas, with no model:

    size and place     a large area at the top of the frame is a sky; a small round one a source
    shape              a filled rectangle with a dark surround is a window; many specks are glints
    the colour around  the ring of pixels just outside the area: blue, warm or neutral
    motion (several    a warm, irregular area whose size jumps from frame to frame is a flame
    frames given)

A suggestion is a starting point the artist can edit; every one says why it was picked. Nothing
here changes the fill: it only fills the prompt boxes.

1.2.6 (18 Nov 2026, user feedback: "the prompt suggestion should be related to the shot"): with
`--describe auto` (the app's Suggest button) a small captioning model, Florence-2 base
(florence-community/Florence-2-base, about 0.5 GB, transformers' own Florence2ForConditionalGeneration),
describes the frame and a crop round its brightest area. `shot_prompt` turns the two captions into
a short prompt about the bright area only (what it is, how it looks, what it is seen through, what
lies beyond): "bright hazy sky, soft even white light, seen through an open doorway, over the sea".
People and objects in the captions are left out, so the fill is never asked to draw them. It is the
first suggestion ("From this shot"); without the model (not installed, no network the first time)
the presets are suggested as before and the reason says why.

    python -m rudra.genfill.prompts suggest <image|EXR> [more frames] [--tau-high 250] [--tau-low -1] [--cache DIR] [--json]
                                            [--describe off|auto] [--describe-model florence-community/Florence-2-base]
    python -m rudra.genfill.prompts list [--cache DIR] [--json]
    python -m rudra.genfill.prompts precache --base <Wan2.1-VACE-1.3B-diffusers> --cache <LoRA folder>/prompt_cache

Torch-free except `precache`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    text: str
    region: str          # 'highlight' or 'shadow': which prompt box it fills


PRESETS: tuple[Preset, ...] = (
    Preset("sky_blue", "Clear blue sky", "clear blue sky, smooth gradient toward the horizon, soft high clouds", "highlight"),
    Preset("sky_overcast", "Overcast sky", "bright overcast sky, soft even white light, subtle cloud texture", "highlight"),
    Preset("sky_sunset", "Sunset sky", "glowing sunset sky, warm orange to pink gradient, soft clouds", "highlight"),
    Preset("sun_disc", "Sun", "warm sun disc with a soft halo and a smooth glow", "highlight"),
    Preset("light_source", "Lamp / practical", "practical lamp, bright core with a soft falloff", "highlight"),
    Preset("window_daylight", "Window daylight", "window daylight, soft bright exterior, smooth falloff at the frame",
           "highlight"),
    Preset("specular_glints", "Specular glints", "small specular glints on glossy surfaces, sharp bright cores",
           "highlight"),
    Preset("flame", "Flame", "flame core, bright yellow center with an orange glow", "highlight"),
    Preset("shadow_detail", "Shadow detail", "dark shadow detail, soft texture, low light", "shadow"),
)
BY_ID = {p.id: p for p in PRESETS}

LARGE = 0.03          # a component covering at least this share of the frame is large
SOURCE = 0.05         # a round component below this share can be a single source (sun, lamp)
TINY = 0.003          # below this a component is a speck
GLINTS = 3            # this many specks make glints


def cache_key(text: str) -> str:
    """model.PromptBank.key, without importing torch: the prompt cache's file name."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:24]


def cached(text: str, cache: str | Path | None) -> bool:
    return bool(cache) and (Path(cache) / f"{cache_key(text)}.pt").is_file()


# ------------------------------------------------------------------ reading frames
def codes_of(path: str | Path) -> np.ndarray:
    """A frame as (H, W, 3) uint8 SDR codes: an SDR image as stored, an EXR master through its
    ACES 2.0 SDR view (what the fill's regions are made from)."""
    import cv2
    p = Path(path)
    if p.suffix.lower() == ".exr":
        from rudra.aces_view import ACES2View
        from rudra.genfill.run import read_frame
        return np.rint(np.clip(ACES2View().forward(read_frame(p)["ap1"]), 0, 1) * 255).astype(np.uint8)
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"cannot read {p}")
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, -1)
    im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB)
    if im.dtype == np.uint16:
        im = np.rint(im / 257.0)
    return np.clip(im, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ analysis
def components(codes: np.ndarray, tau: int = 254) -> list[dict]:
    """Fully clipped areas (max code >= tau) and what is around them."""
    import cv2
    h, w = codes.shape[:2]
    mask = (codes.max(-1) >= tau).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    rgb = codes.astype(np.float32) / 255.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        a = area / float(h * w)
        comp = (lab == i).astype(np.uint8)
        ring = (cv2.dilate(comp, k) > 0) & (comp == 0)
        rc = rgb[ring].mean(0) if ring.any() else np.array([1.0, 1.0, 1.0], np.float32)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = sum(cv2.arcLength(c, True) for c in contours)
        out.append({
            "area": a, "fill": area / float(max(1, bw * bh)), "aspect": bw / float(max(1, bh)),
            "top": y == 0, "cy": (cent[i][1]) / h,
            "circularity": float(4 * math.pi * area / (per * per)) if per > 0 else 1.0,
            "ring_rgb": [float(v) for v in rc], "ring_luma": float(rc @ np.array([0.2126, 0.7152, 0.0722])),
            "warmth": float(rc[0] - rc[2]), "blueness": float(rc[2] - rc[0]),
        })
    return out


def classify(c: dict) -> tuple[str, str]:
    """(preset id, why) for one clipped area of at least TINY."""
    pct = 100 * c["area"]
    # a filled rectangle in a dark surround, not cut by the frame's top: a window, before the sky test
    if (c["fill"] >= 0.8 and c["area"] >= 0.005 and 0.25 <= c["aspect"] <= 4 and c["ring_luma"] < 0.35
            and not c["top"]):
        return "window_daylight", f"a rectangular clipped area ({pct:.1f}%) with a dark surround"
    if c["area"] >= LARGE and (c["top"] or c["cy"] < 0.45):
        if c["blueness"] > 0.06:
            return "sky_blue", f"a large clipped area ({pct:.0f}% of the frame) at the top, blue around it"
        if c["warmth"] > 0.08:
            return "sky_sunset", f"a large clipped area ({pct:.0f}%) at the top, warm around it"
        return "sky_overcast", f"a large clipped area ({pct:.0f}%) at the top, neutral around it"
    if c["area"] < SOURCE and c["circularity"] > 0.6:
        if c["warmth"] > 0.08:
            return "sun_disc", f"a small round clipped area ({pct:.2f}%), warm around it"
        return "light_source", f"a small round clipped area ({pct:.2f}%), neutral around it"
    if c["warmth"] > 0.08 and c["circularity"] < 0.4:
        return "flame", f"an irregular clipped area ({pct:.2f}%), warm around it"
    return "light_source", f"a clipped area ({pct:.2f}%)"


def suggest(frames: list[np.ndarray], tau_high: int = 250, tau_low: int = -1) -> dict:
    """Ranked suggestions for SDR code frames (one is enough; several let a flame show itself).
    tau_high / tau_low: the fill's own thresholds (the clipped test is code >= max(tau_high, 254),
    the fully clipped core the fill blends into)."""
    if not frames:
        raise ValueError("no frames")
    tau = max(254, int(tau_high))
    votes: dict[str, dict] = {}

    def vote(pid, score, why):
        v = votes.setdefault(pid, {"score": 0.0, "why": why})
        v["score"] += score
        if score >= v.get("best", 0):
            v["best"], v["why"] = score, why
    warm_area = []
    clipped = []
    for codes in frames:
        comps = components(codes, tau)
        clipped.append(sum(c["area"] for c in comps))
        specks = [c for c in comps if c["area"] < TINY]
        if len(specks) >= GLINTS:
            vote("specular_glints", sum(c["area"] for c in specks) + 0.001 * len(specks),
                 f"{len(specks)} small clipped specks")
        warm = 0.0
        for c in comps:
            if c["area"] < TINY:
                continue
            pid, why = classify(c)
            vote(pid, c["area"], why)
            if c["warmth"] > 0.08:
                warm += c["area"]
        warm_area.append(warm)
    if len(frames) >= 3 and np.mean(warm_area) > 0:
        cv = float(np.std(warm_area) / np.mean(warm_area))
        if cv > 0.15:   # a warm area that changes size from frame to frame
            vote("flame", float(np.mean(warm_area)) * (1 + cv) * len(frames),   # scores are divided by n below
                 f"a warm clipped area whose size changes {100 * cv:.0f}% from frame to frame")
    if tau_low >= 0:
        crushed = float(np.mean([(f.max(-1) <= max(1, tau_low)).mean() for f in frames]))
        if crushed > 0.002:
            vote("shadow_detail", crushed, f"{100 * crushed:.1f}% of the frame is crushed to black")
    n = len(frames)
    ranked = sorted(votes.items(), key=lambda kv: -kv[1]["score"])
    return {"clipped_fraction": float(np.mean(clipped)),
            "suggestions": [{**asdict(BY_ID[pid]), "score": v["score"] / n, "reason": v["why"]} for pid, v in ranked]}


# ------------------------------------------------------------------ the shot, in words (1.2.6)
DESCRIBE_MODEL = "florence-community/Florence-2-base"

# What a caption may say about the bright area, in the order they are tried: (words, subject, preset).
_SUBJECTS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("fire", "flame", "flames", "candle", "bonfire", "fireplace", "burning"), "flame core, bright yellow center with an orange glow", "flame"),
    (("sunset", "sunrise", "dusk", "golden hour", "orange sky", "pink sky"), "glowing sunset sky, warm orange to pink gradient", "sky_sunset"),
    (("blue sky", "clear sky", "clear blue", "sunny", "sunshine"), "clear blue sky, smooth gradient toward the horizon", "sky_blue"),
    (("sun", "sunlight"), "warm sun disc with a soft halo and a smooth glow", "sun_disc"),
    (("lamp", "light bulb", "bulb", "chandelier", "lantern", "neon", "headlight", "spotlight", "street light", "streetlight"),
     "practical lamp, bright core with a soft falloff", "light_source"),
    (("snow", "snowy", "ice"), "bright snow, soft white texture", "sky_overcast"),
    (("sky", "clouds", "cloud", "cloudy", "overcast", "horizon", "haze", "hazy", "fog", "foggy", "mist", "misty"),
     "bright sky", "sky_overcast"),
    (("window",), "window daylight, soft bright exterior", "window_daylight"),
)
_LOOKS = (("hazy", "hazy"), ("haze", "hazy"), ("fog", "foggy"), ("foggy", "foggy"), ("mist", "misty"), ("misty", "misty"),
          ("overcast", "overcast"), ("cloudy", "cloudy"))
_THROUGH = (("doorway", "an open doorway"), ("cargo door", "an open cargo door"), ("door", "an open doorway"),
            ("hatch", "an open hatch"), ("window", "a window"), ("tunnel", "the end of a tunnel"),
            ("garage", "an open garage door"), ("archway", "an archway"), ("arch", "an archway"), ("ramp", "an open ramp"))
_BEYOND = (("ocean", "the ocean"), ("sea", "the sea"), ("beach", "a beach"), ("lake", "a lake"), ("river", "a river"),
           ("water", "water"), ("mountain", "mountains"), ("hills", "hills"), ("desert", "a desert"), ("city", "a city"),
           ("buildings", "buildings"), ("trees", "trees"), ("forest", "a forest"), ("field", "fields"), ("runway", "a runway"))


def _has(text: str, word: str) -> bool:
    """`word` (or its plural) as a word of `text`."""
    import re
    return re.search(r"(?<![a-z])" + re.escape(word.strip()) + r"(?:s|es)?(?![a-z])", text) is not None


def shot_prompt(scene: str, area: str, fallback: str = "sky_overcast") -> tuple[str, str]:
    """(prompt, preset id it is closest to) from a caption of the frame and one of a crop round its
    brightest area. Only what the bright area is, how it looks, what it is seen through and what lies
    beyond it: never the people or things in front of it."""
    a, sc = (area or "").lower(), (scene or "").lower()
    both = a + " " + sc
    subject, pid = None, fallback
    for words, subj, preset in _SUBJECTS:          # the crop first: it is the bright area itself
        if any(_has(a, w) for w in words):
            subject, pid = subj, preset
            break
    if subject is None:
        for words, subj, preset in _SUBJECTS:
            if any(_has(sc, w) for w in words):
                subject, pid = subj, preset
                break
    if subject is None:
        subject = BY_ID[fallback].text.split(",")[0]
    clouds = _has(both, "clouds") or _has(both, "cloud")
    if pid == "sky_overcast" and subject == "bright sky":
        looks = []
        for w, look in _LOOKS:
            if _has(both, w) and look not in looks:
                looks.append(look)
        subject = "bright " + " ".join(looks[:2] + ["sky"]) + ", soft even white light" + (", subtle cloud texture" if clouds else "")
    elif pid == "sky_blue" and clouds:
        subject += ", soft high clouds"
    parts = [subject]
    if pid not in ("window_daylight", "light_source", "flame"):
        for w, phrase in _THROUGH:
            if _has(both, w):
                parts.append("seen through " + phrase)
                break
    for w, phrase in _BEYOND:
        if _has(both, w) and pid not in ("light_source", "flame", "sun_disc"):
            if pid == "window_daylight":
                parts.append(phrase + " outside")
            else:
                parts.append(("over " if phrase in ("the sea", "the ocean", "water", "a lake", "a river") else "above ") + phrase)
            break
    return ", ".join(parts), pid


class Describer:
    """Florence-2 captions of a frame (RGB uint8). Loads once; raises if transformers or the model
    cannot be had (the caller falls back to the presets)."""

    def __init__(self, model_id: str = DESCRIBE_MODEL, device: str = "auto"):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        # 1.2.6 (checklist item 7): a folder (setup.py --florence / --florence-from) loads from
        # that folder only, never the network; the app also runs this with HF_HUB_OFFLINE=1.
        local = {"local_files_only": True} if Path(model_id).is_dir() else {}
        self.model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype, **local).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, **local)

    def caption(self, rgb: np.ndarray, task: str = "<DETAILED_CAPTION>") -> str:
        import torch
        from PIL import Image
        img = Image.fromarray(rgb)
        inputs = self.processor(text=task, images=img, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=96, num_beams=3)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        out = self.processor.post_process_generation(text, task=task, image_size=img.size)
        return str(out.get(task, "") if isinstance(out, dict) else out).strip()


def bright_crop(codes: np.ndarray, tau: int = 240, grow: float = 0.5) -> np.ndarray | None:
    """The frame round its largest bright area (max code >= tau), grown by `grow` of its size each side."""
    import cv2
    mask = (codes.max(-1) >= tau).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[i, :4])
    H, W = codes.shape[:2]
    x0, y0 = max(0, int(x - grow * w)), max(0, int(y - grow * h))
    x1, y1 = min(W, int(x + w + grow * w)), min(H, int(y + h + grow * h))
    if (x1 - x0) * (y1 - y0) < 32 * 32:
        return None
    return codes[y0:y1, x0:x1]


def describe_suggestion(codes: np.ndarray, model_id: str = DESCRIBE_MODEL, describer=None,
                        fallback: str = "sky_overcast") -> dict:
    """The "From this shot" suggestion for one frame, or raises (no model). `fallback`: the preset
    the rules picked, used when the captions name no bright subject."""
    d = describer or Describer(model_id)
    scene = d.caption(codes)
    crop = bright_crop(codes)
    area = d.caption(crop) if crop is not None else ""
    text, pid = shot_prompt(scene, area, fallback)
    return {"id": "shot", "label": "From this shot", "text": text, "region": "highlight", "score": 1e6,
            "reason": f"Florence-2: the frame \"{scene}\"" + (f"; its bright area \"{area}\"" if area else ""),
            "like": pid}


# ------------------------------------------------------------------ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("suggest")
    s.add_argument("frames", nargs="+")
    s.add_argument("--tau-high", type=int, default=250)
    s.add_argument("--tau-low", type=int, default=-1)
    s.add_argument("--cache", default="")
    s.add_argument("--json", action="store_true")
    s.add_argument("--describe", default="off", choices=("off", "auto"),
                   help="auto: Florence-2 describes the shot and a prompt is made from it (1.2.6)")
    s.add_argument("--describe-model", default=DESCRIBE_MODEL)
    li = sub.add_parser("list")
    li.add_argument("--cache", default="")
    li.add_argument("--json", action="store_true")
    pc = sub.add_parser("precache")
    pc.add_argument("--base", required=True, help="Wan2.1-VACE-1.3B-diffusers (its text_encoder/ and tokenizer/)")
    pc.add_argument("--cache", required=True, help="the prompt cache beside the LoRA")
    a = ap.parse_args(argv)
    if a.cmd == "precache":
        from .model import BASE_PROMPT, PromptBank
        bank = PromptBank(a.base, a.cache, device="cpu")
        for text in (BASE_PROMPT,) + tuple(p.text for p in PRESETS):
            was = cached(text, a.cache)
            bank.get(text)
            print(("cached   " if was else "embedded ") + text)
        bank.release()
        return 0
    presets = [{**asdict(p), "cached": cached(p.text, a.cache)} for p in PRESETS]
    if a.cmd == "list":
        print(json.dumps({"presets": presets}, indent=1) if a.json else
              "\n".join(f"{p['id']:16} {'ready' if p['cached'] else '     '}  {p['text']}" for p in presets))
        return 0
    try:
        res = suggest([codes_of(f) for f in a.frames], a.tau_high, a.tau_low)
    except Exception as e:                                              # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}) if a.json else f"error: {e}")
        return 2
    if a.describe == "auto":
        try:
            top_id = res["suggestions"][0]["id"] if res["suggestions"] else "sky_overcast"
            res["suggestions"].insert(0, describe_suggestion(codes_of(a.frames[len(a.frames) // 2]), a.describe_model,
                                                             fallback=top_id if top_id in BY_ID else "sky_overcast"))
        except Exception as e:                                          # noqa: BLE001
            res["describe_error"] = f"{type(e).__name__}: {e}"[:300]
    for sgg in res["suggestions"]:
        sgg["cached"] = cached(sgg["text"], a.cache)
    res.update(ok=True, presets=presets)
    if a.json:
        print(json.dumps(res))
    else:
        print(f"clipped: {100 * res['clipped_fraction']:.2f}% of the frame")
        for sgg in res["suggestions"]:
            print(f"  {sgg['label']:18} {sgg['text']}\n  {'':18} ({sgg['reason']})")
        if not res["suggestions"]:
            print("  nothing fully clipped: the fill has nothing to fill here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
