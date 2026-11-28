"""The OFX plug-in's prompt helpers (1.3.2, 26 Nov 2026), started beside the host as ofx_job.py is:

    python rudra/genfill/ofx_prompt.py suggest --repo <RUDRA's Python source> --frame <frame.tif> --out <suggest.json>
                                               [--florence <Florence-2-base folder>] [--cache <prompt cache>]
                                               [--device auto|cpu] [--tau-high 240]
    python rudra/genfill/ofx_prompt.py embed --repo <src> --base <Wan2.1-VACE-1.3B> --lora <v5> --cache <prompt cache>
                                             --text <prompt> [--device cpu|auto]

suggest: the plug-in's Suggest prompt button. The frame's SDR codes (the 16-bit TIFF the plug-in wrote) go through
rudra.genfill.prompts as the app's Suggest a prompt runs it: the presets the clipped areas point to and, with
Florence-2 set up, "From this shot" first. Florence-2 loads from its folder only (HF_HUB_OFFLINE): never the
network. The answer is written to --out whole (a .part renamed), as JSON:
    {"ok": true, "suggestions": [{"id", "label", "text", "region", "reason", "cached"}, ...], "clipped_fraction": f,
     "describe_error": "..." (when there is no "From this shot")}   or   {"ok": false, "error": "..."}

embed: started by Generate when it has a prompt, so a new prompt is read while the frames are collected (minutes),
not after: the base prompt and the prompt into the plug-in's prompt cache, seeded from the cache beside the LoRA.
It holds <cache>/.embedding.lock meanwhile (ofx_job.EMBED_LOCK, touched every EMBED_BEAT seconds), which the fill
waits for; a second embed while one runs leaves at once. On the CPU by default: the learned engine may be on the GPU
while the frames are collected.

Nothing is written to stdout (the plug-in does not read it); what happens goes to stderr, the plug-in's log file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True)


def write_json(path: Path, obj: dict) -> None:
    """`obj` to `path` whole: a .part, renamed over it."""
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(part, path)


def suggest(a) -> int:
    out = Path(a.out)
    try:
        from rudra.genfill import prompts as P
        florence = Path(a.florence) if a.florence else None
        if florence is not None and florence.is_dir():
            os.environ["HF_HUB_OFFLINE"] = "1"   # 1.2.6: Florence-2 from its folder, never the network
        codes = P.codes_of(a.frame)
        res = P.suggest([codes], a.tau_high, -1)
        if florence is not None and (florence / "config.json").is_file():
            top = res["suggestions"][0]["id"] if res["suggestions"] else "sky_overcast"
            t0 = time.time()
            try:
                res["suggestions"].insert(0, P.describe_suggestion(codes, str(florence), device=a.device,
                                                                   fallback=top if top in P.BY_ID else "sky_overcast"))
                say(f"Florence-2 described the frame in {time.time() - t0:.1f} s ({a.device})")
            except Exception as e:  # noqa: BLE001 - the presets still answer
                res["describe_error"] = f"{type(e).__name__}: {e}"[:300]
        else:
            res["describe_error"] = "Florence-2 is not set up (RUDRA's AI fill setup installs it)"
        for s in res["suggestions"]:
            s["cached"] = P.cached(s["text"], a.cache)
        res["ok"] = True
        write_json(out, res)
        say(f"suggested: {[s['text'] for s in res['suggestions'][:3]]}")
        return 0
    except Exception as e:  # noqa: BLE001 - the plug-in reads why from the file
        write_json(out, {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]})
        say(f"suggest failed: {type(e).__name__}: {e}")
        return 2


def embed(a) -> int:
    from rudra.genfill.ofx_job import EMBED_BEAT, EMBED_LOCK, embedding_running, seed_cache
    from rudra.genfill.prompts import cached
    cache = seed_cache(Path(a.cache), Path(a.lora))
    base_prompt = model_base_prompt(Path(a.repo))
    texts = [t for t in dict.fromkeys([base_prompt, a.text.strip()]) if t]
    missing = [t for t in texts if not cached(t, cache)]
    if not missing:
        say("embed: every prompt is in the cache already")
        return 0
    if embedding_running(cache):
        say("embed: another reading of a prompt holds the cache; leaving it to that one")
        return 0
    lock = cache / EMBED_LOCK
    lock.write_text(json.dumps({"pid": os.getpid(), "started": time.time(), "texts": missing}), encoding="utf-8")
    stop = threading.Event()

    def beat():
        while not stop.wait(EMBED_BEAT):
            try:
                os.utime(lock, None)
            except OSError:
                pass
    threading.Thread(target=beat, daemon=True).start()
    try:
        import torch
        from rudra.genfill.model import PromptBank
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32   # as the fill would read it here
        bank = PromptBank(a.base, cache, device=a.device, dtype=dtype)
        t0 = time.time()
        for t in missing:
            bank.get(t)
        bank.release()
        say(f"embed: {len(missing)} prompt(s) read in {time.time() - t0:.1f} s on the {bank.encoder_device}")
        return 0
    except Exception as e:  # noqa: BLE001 - the fill reads the prompt itself then
        say(f"embed failed (the fill reads the prompt itself): {type(e).__name__}: {e}")
        return 2
    finally:
        stop.set()
        try:
            lock.unlink()
        except OSError:
            pass


def model_base_prompt(repo: Path) -> str:
    """model.BASE_PROMPT, read from the source (the fill always uses it; importing model.py needs torch)."""
    import ast
    src = (repo / "rudra" / "genfill" / "model.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "BASE_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    return ""


def main(argv: list[str] | None = None) -> int:
    out, sys.stdout = sys.stdout, sys.stderr   # the plug-in does not read stdout: nothing may fill its pipe
    try:
        ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
        sub = ap.add_subparsers(dest="cmd", required=True)
        s = sub.add_parser("suggest")
        s.add_argument("--repo", required=True)
        s.add_argument("--frame", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--florence", default="")
        s.add_argument("--cache", default="")
        s.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
        s.add_argument("--tau-high", type=int, default=240)
        e = sub.add_parser("embed")
        e.add_argument("--repo", required=True)
        e.add_argument("--base", required=True)
        e.add_argument("--lora", required=True)
        e.add_argument("--cache", required=True)
        e.add_argument("--text", required=True)
        e.add_argument("--device", default="cpu", choices=("cpu", "auto", "cuda"))
        a = ap.parse_args(argv)
        here = Path(__file__).resolve().parent   # started as a script: its own folder is sys.path[0]; the package
        sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != here]   # names come from the repo
        repo = str(Path(a.repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        return suggest(a) if a.cmd == "suggest" else embed(a)
    finally:
        sys.stdout = out


if __name__ == "__main__":
    raise SystemExit(main())
