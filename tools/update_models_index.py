#!/usr/bin/env python3
"""Change the published models.json in place (RUDRA 1.1, 7 Nov 2026).

make_models_json.py writes an index from scratch; this edits the one already published, so
the models nobody touched keep their exact entries (and the app keeps them installed: an
installed model is matched by its id and version, which this never changes).

    python tools/update_models_index.py --in https://www.fxtdstudios.com/rudra-docs/models.json \\
        --out models.json --channel rudra_aces_2=preview \\
        --refresh rudra_genfill_v3=<local folder> --revision v1.1.0

--channel ID=CHANNEL   set a model's channel (release, preview, legacy): the app badges preview
--refresh ID=FOLDER    list that model's files again from FOLDER (sizes and SHA-256), at the same
                       Hugging Face repo and folder as before, pinned to --revision. Upload the
                       folder's new files there first (and tag the revision).
Then publish --out where the app reads it (www.fxtdstudios.com/rudra-docs/models.json).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_models_json import CHANNELS, model_entry  # noqa: E402

HF_URL = re.compile(r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)$")


# 7 Nov 2026: the website answers Python's default "Python-urllib" agent with 403 (the app's
# Qt downloader is let through); a plain agent of our own, as a browser would send one.
USER_AGENT = "Mozilla/5.0 (compatible; RUDRA-models-index/1.1)"


def load(src: str) -> dict:
    if src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:   # noqa: S310 (https only)
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{src}: the site answered {e.code}. Open it in a browser, save it as a file "
                             f"(for example dist\\models_live.json) and pass --in with that file.") from None
        except urllib.error.URLError as e:
            raise SystemExit(f"{src}: {e.reason}. Save it from a browser and pass --in with the file.") from None
    return json.loads(Path(src).read_text(encoding="utf-8"))


def pairs(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{what} wants ID=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def where(entry: dict) -> tuple[str, str]:
    """(Hugging Face repo, the folder's prefix in it) from the entry's first file URL."""
    m = HF_URL.match(entry["files"][0]["url"])
    if not m:
        raise SystemExit(f"{entry['id']}: its URLs are not Hugging Face resolve URLs; use make_models_json.py")
    path, rel, folder = m["path"], entry["files"][0]["path"], entry["folder"]
    suffix = f"{folder}/{rel}"
    if not path.endswith(suffix):
        raise SystemExit(f"{entry['id']}: cannot find {suffix} in {path}")
    return m["repo"], path[: -len(suffix)].strip("/")


def update(index: dict, channels: dict[str, str], refresh: dict[str, Path], revision: str | None) -> dict:
    models = index.get("models", [])
    by_id = {m["id"]: m for m in models}
    for mid in list(channels) + list(refresh):
        if mid not in by_id:
            raise SystemExit(f"no model {mid!r} in the index (it has {', '.join(by_id)})")
    for mid, ch in channels.items():
        if ch not in CHANNELS:
            raise SystemExit(f"channel must be one of {', '.join(CHANNELS)}")
        by_id[mid]["channel"] = ch
    for mid, folder in refresh.items():
        old = by_id[mid]
        if folder.name != old["folder"]:
            raise SystemExit(f"{mid}: the folder must be named {old['folder']!r} (it is published under that name)")
        repo, prefix = where(old)
        rev = revision or HF_URL.match(old["files"][0]["url"])["rev"]
        opts = {k: old.get(k) for k in ("id", "title", "version", "channel", "kind", "summary", "licence", "licence_url",
                                         "min_app")}
        new = model_entry(folder, opts, repo, rev, prefix)
        gone = {f["path"] for f in old["files"]} - {f["path"] for f in new["files"]}
        if gone:
            raise SystemExit(f"{mid}: {folder} lacks files the published entry has: {', '.join(sorted(gone))}")
        for k in old:                                   # anything else the entry carried stays
            new.setdefault(k, old[k])
        models[models.index(old)] = new
    return {**index, "models": models}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="the published models.json (https URL or a file)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--channel", action="append", default=[], metavar="ID=CHANNEL")
    ap.add_argument("--refresh", action="append", default=[], metavar="ID=FOLDER")
    ap.add_argument("--revision", help="the tag or commit a refreshed model's URLs pin (default: as before)")
    a = ap.parse_args(argv)
    out = update(load(a.src), pairs(a.channel, "--channel"),
                 {k: Path(v) for k, v in pairs(a.refresh, "--refresh").items()}, a.revision)
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for m in out["models"]:
        print(f"{m['id']}: {m['channel']}, {len(m['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
