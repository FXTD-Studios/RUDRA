#!/usr/bin/env python3
"""Write the models.json the app's Model packages window downloads from (29 Oct 2026).

Each model is a local folder, published as-is to a Hugging Face repo. This
walks the folder, takes every file's size and SHA-256, and writes the entry
with the file's Hugging Face URL. The app (native/app/model_downloads.hpp)
refuses an entry with a path outside its folder, a non-https URL or a bad
checksum, and checks every file's SHA-256 before it uses it.

    python tools/make_models_json.py --repo fxtdstudios/RUDRA --revision main --out models.json \\
        --model dist/models/aces_v2_guarded_projected600 --id aces_v2_gp600 \\
            --title "Learned ACES 2.0" --version 2.0 --channel preview --kind aces_v2 \\
            --summary "rebuilds clipped highlights through the ACES 2.0 view" \\
            --licence "RUDRA model terms" --licence-url https://www.fxtdstudios.com/rudra/licence

--repo, --revision, --prefix, --out and --merge come before the first --model.
Repeat --model ... for each model (its options follow it). --merge keeps the
other models already in --out. --prefix is the folder in the repo the model
folder sits in (default: models). Upload the folder to
<repo>/<prefix>/<folder name> before publishing models.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import quote

SKIP = {".rudra-download.json", ".DS_Store", "Thumbs.db"}
CHANNELS = ("release", "preview", "legacy")
KINDS = ("aces_v2", "package", "genfill")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_entry(folder: Path, opts: dict, repo: str, revision: str, prefix: str) -> dict:
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    files = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.name in SKIP or p.name.endswith(".part") or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(folder).as_posix()
        repo_path = "/".join(x for x in (prefix.strip("/"), folder.name, rel) if x)
        files.append({
            "path": rel,
            "url": f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{quote(repo_path)}",
            "size": p.stat().st_size,
            "sha256": sha256_of(p),
        })
    if not files:
        raise SystemExit(f"no files in {folder}")
    title = opts.get("title")
    manifest = folder / "manifest.json"
    if not title and manifest.is_file():
        try:
            title = json.loads(manifest.read_text(encoding="utf-8")).get("name")
        except (OSError, ValueError):
            title = None
    entry = {
        "id": opts.get("id") or folder.name,
        "title": title or folder.name,
        "version": opts.get("version") or "1.0",
        "channel": opts.get("channel") or "release",
        "kind": opts.get("kind") or "package",
        "summary": opts.get("summary") or "",
        "licence": opts.get("licence") or "",
        "licence_url": opts.get("licence_url") or "",
        "folder": folder.name,
        "files": files,
    }
    if opts.get("min_app"):
        entry["min_app"] = opts["min_app"]
    if entry["channel"] not in CHANNELS:
        raise SystemExit(f"--channel must be one of {', '.join(CHANNELS)}")
    if entry["kind"] not in KINDS:
        raise SystemExit(f"--kind must be one of {', '.join(KINDS)}")
    return entry


def split_models(argv: list[str]) -> tuple[list[str], list[list[str]]]:
    """The arguments before the first --model, then one list per --model."""
    head, groups = [], []
    for a in argv:
        if a == "--model":
            groups.append([])
        elif groups:
            groups[-1].append(a)
        else:
            head.append(a)
    return head, groups


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    head, groups = split_models(argv)
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    top.add_argument("--repo", required=True, help="Hugging Face repo, e.g. fxtdstudios/RUDRA")
    top.add_argument("--revision", default="main", help="branch, tag or commit the URLs pin (a tag or commit is safer)")
    top.add_argument("--prefix", default="models", help="folder in the repo that holds the model folders")
    top.add_argument("--out", required=True, type=Path)
    top.add_argument("--merge", action="store_true", help="keep the other models already in --out")
    args, rest = top.parse_known_args(head)
    if rest:
        top.error(f"unknown arguments before --model: {' '.join(rest)}")
    if not groups:
        top.error("give at least one --model <folder> [options]")
    per = argparse.ArgumentParser(prog="--model")
    per.add_argument("folder", type=Path)
    for k in ("id", "title", "version", "channel", "kind", "summary", "licence", "licence-url", "min-app"):
        per.add_argument("--" + k)
    entries = []
    for g in groups:
        o = per.parse_args(g)
        opts = {k.replace("-", "_"): getattr(o, k.replace("-", "_")) for k in
                ("id", "title", "version", "channel", "kind", "summary", "licence", "licence-url", "min-app")}
        entries.append(model_entry(o.folder, opts, args.repo, args.revision, args.prefix))
    models = []
    if args.merge and args.out.is_file():
        old = json.loads(args.out.read_text(encoding="utf-8"))
        ids = {e["id"] for e in entries}
        models = [m for m in old.get("models", []) if m.get("id") not in ids]
    models += entries
    args.out.write_text(json.dumps({"version": 1, "models": models}, indent=2) + "\n", encoding="utf-8")
    for e in entries:
        total = sum(f["size"] for f in e["files"])
        print(f"{e['id']}: {len(e['files'])} files, {total / 1e9:.2f} GB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
