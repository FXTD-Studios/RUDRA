#!/usr/bin/env python3
"""Publish a staged models folder to Hugging Face and tag it (31 Oct 2026).

    python tools/hf_publish_models.py --repo fxtdstudios/rudra-models --stage dist/hf_v1 --revision v1.0.0

The stage holds README.md (the model card), LICENSE and models/<folder>/...
(scripts/UPLOAD_MODELS_HF.ps1 makes it). This creates the repo if needed,
uploads the stage in one commit, tags that commit `--revision` so the URLs in
models.json never move, and checks every file is there at its size.

Sign in first: `hf auth login` (a write token for the fxtdstudios
organisation), or set HF_TOKEN. Nothing is printed of the token.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stage", required=True, type=Path)
    ap.add_argument("--revision", required=True, help="the tag the published files are pinned to (v1.0.0)")
    ap.add_argument("--private", action="store_true", help="create the repo private (it is public otherwise)")
    ap.add_argument("--message", default="RUDRA 1.0 models")
    ap.add_argument("--dry-run", action="store_true", help="sign-in and repo checks only; nothing uploaded")
    a = ap.parse_args(argv)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed: python -m pip install -U huggingface_hub", file=sys.stderr)
        return 2

    stage = a.stage.resolve()
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    if not (stage / "README.md").is_file() or not any(p.parts[-3] == "models" for p in files if len(p.parts) >= 3):
        print(f"{stage} is not a staged models folder (README.md and models/<folder>/...)", file=sys.stderr)
        return 2

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as e:  # not signed in, or a bad token
        print(f"Not signed in to Hugging Face ({type(e).__name__}): run `hf auth login` or set HF_TOKEN", file=sys.stderr)
        return 3
    orgs = [o.get("name") for o in who.get("orgs", [])]
    owner = a.repo.split("/")[0]
    if owner != who.get("name") and owner not in orgs:
        print(f"Signed in as {who.get('name')}, who cannot write to {owner} (organisations: {', '.join(orgs) or 'none'})", file=sys.stderr)
        return 3
    print(f"signed in as {who.get('name')}; {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB to {a.repo}")
    tags = []
    if api.repo_exists(a.repo):
        tags = [t.name for t in api.list_repo_refs(a.repo).tags]
        if a.revision in tags:
            print(f"{a.repo} already has the tag {a.revision}: published files never move; use a new revision", file=sys.stderr)
            return 4
    if a.dry_run:
        print("dry run: nothing uploaded")
        return 0

    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    commit = api.upload_folder(repo_id=a.repo, folder_path=str(stage), commit_message=f"{a.message} ({a.revision})",
                               ignore_patterns=["*.part", ".rudra-download.json", "__pycache__/*"])
    sha = getattr(commit, "oid", None) or str(commit).rsplit("/", 1)[-1]
    try:
        api.create_tag(a.repo, tag=a.revision, revision=sha, tag_message=a.message)
    except Exception as e:  # HfHubHTTPError (its module moved in huggingface_hub 1.x)
        print(f"uploaded, but the tag {a.revision} failed: {e}", file=sys.stderr)
        return 5
    # Every file there, at its size, at the tag.
    remote = {f.rfilename: f.size for f in api.model_info(a.repo, revision=a.revision, files_metadata=True).siblings}
    bad = []
    for p in files:
        rel = p.relative_to(stage).as_posix()
        if remote.get(rel) != p.stat().st_size:
            bad.append(f"{rel}: {remote.get(rel)} on the hub, {p.stat().st_size} here")
    if bad:
        print("missing or different on the hub:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 6
    print(f"published {len(files)} files to https://huggingface.co/{a.repo}/tree/{a.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
