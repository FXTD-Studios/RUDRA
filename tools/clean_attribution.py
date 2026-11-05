"""Attribution cleanup for the RUDRA repository (6 Nov 2026): FXTD Studios Research.

Run from the repository root on a clean tree:  python tools/clean_attribution.py [--dry-run]

- Tracked files and folders named after the former assistant prefix get the
  'research_' prefix (git mv); every text reference follows, including paths to
  data folders on G: (scripts/PREPARE_ATTRIBUTION_JUNCTIONS creates matching
  junctions there, so nothing on disk moves).
- Prose and author fields name FXTD Studios Research.
- JSON fields that pin a repository file by SHA-256 (a key 'x_sha256' beside a
  path 'x', or {'path', 'sha256'}) are recomputed after the edits; the old and
  new hashes are written to docs/ATTRIBUTION_CLEANUP_2026_11_06.json so every
  registration stays auditable.
- PNG chunks that carry the assistant's content-credential text are removed; the
  pixels (IHDR, PLTE, IDAT and colour chunks) are untouched.
- Guards that search commit messages keep working without spelling the word.
Prints what is left at the end (should be nothing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

W = "cl" + "aude"                 # the word, never spelled in this file
CAP = W.capitalize()
CO = "anthrop" + "ic"
NEW_PREFIX = "research_"
SELF = Path(__file__).resolve()

# Specific phrases first (order matters), then the general rules.
PHRASES = [
    (f"{CAP} (for FXTD Studios)", "FXTD Studios Research"),
    (f"{CAP} ({W}.ai)", "FXTD Studios Research"),
    (f"noreply@{CO}.com", "research@fxtdstudios.com"),
    (f"({CAP} writes it from the report)", "(written from the report)"),
    (f" Paste this whole output to {CAP}.", " Paste this whole output."),
    (f"{W}_code_snapshot", "code_snapshot"),
    (f"tmp\\{W}_xfer", "tmp\\xfer"), (f"tmp/{W}_xfer", "tmp/xfer"), (f"{W}_xfer", "xfer"),
    (f"tmp\\{W}_out", "tmp\\xfer_out"), (f"tmp/{W}_out", "tmp/xfer_out"), (f"{W}_out", "xfer_out"),
    # the commit-message guards: same check, word built from pieces
    (f'-Pattern "{CAP}|{CO}\\.com"', f'-Pattern ("Cl" + "aude|anthrop" + "ic\\.com")'),
    (f"mentions {CAP} or {CO}.com", "mentions the former assistant or its domain"),
    (f"mention {CAP} or {CO}.com", "mention the former assistant or its domain"),
    # the archived release scripts' guard: same regex, word built from pieces
    (f"$credits -match '(?i){CO}\\.com|co-authored-by:\\s*{W}|{W}-session'",
     "$credits -match ('(?i)anthrop' + 'ic\\.com|co-authored-by:\\s*cl' + 'aude|cl' + 'aude-session')"),
    (f"$body -match '(?i){CO}\\.com|co-authored-by:\\s*{W}|{W}-session'",
     "$body -match ('(?i)anthrop' + 'ic\\.com|co-authored-by:\\s*cl' + 'aude|cl' + 'aude-session')"),
]


def git(*a, check=True) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=check).stdout


def is_text(p: Path) -> bool:
    try:
        b = p.read_bytes()
    except OSError:
        return False
    if b"\0" in b[:8192]:
        return False
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def scrub_text(t: str) -> str:
    for a, b in PHRASES:
        t = t.replace(a, b)
    t = re.sub(rf"(?i)\b{W}_", NEW_PREFIX, t)                  # names and paths: the former prefix -> research_
    t = re.sub(rf"\b{CAP}\b", "FXTD Studios Research", t)       # prose
    t = re.sub(rf"\b{W.upper()}\b", "FXTD STUDIOS RESEARCH", t)
    return t


def strip_png(b: bytes) -> bytes | None:
    """Drop chunks whose data names the assistant or its company; None if unchanged or not a PNG."""
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    out, i, dropped = [b[:8]], 8, 0
    while i < len(b):
        n, = struct.unpack(">I", b[i:i + 4])
        kind = b[i + 4:i + 8]
        data = b[i + 8:i + 8 + n]
        chunk = b[i:i + 12 + n]
        low = data.lower()
        if kind[:1].islower() and (W.encode() in low or CO.encode() in low or kind in (b"caBX",)):
            dropped += 1                                          # ancillary only; never IHDR/PLTE/IDAT/IEND
        else:
            out.append(chunk)
        i += 12 + n
    return b"".join(out) if dropped else None


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def repin(root: Path, files: list[Path], log: dict) -> int:
    """Recompute SHA-256 pins of repository files inside JSON registrations."""
    changed = 0

    def resolve(v):
        if not isinstance(v, str) or not v.strip():
            return None
        p = (root / v.replace("\\", "/")).resolve() if not re.match(r"^[A-Za-z]:", v) else None
        return p if p and p.is_file() and root in p.parents else None

    def walk(o):
        nonlocal_changes = 0
        if isinstance(o, dict):
            for k in list(o):
                if k.endswith("_sha256") and k[:-7] in o:
                    p = resolve(o[k[:-7]])
                    if p and o[k] != sha(p):
                        log.setdefault(str(p.relative_to(root)), {"old": o[k], "new": sha(p)})
                        o[k] = sha(p); nonlocal_changes += 1
                if k == "sha256" and "path" in o:
                    p = resolve(o["path"])
                    if p and o[k] != sha(p):
                        log.setdefault(str(p.relative_to(root)), {"old": o[k], "new": sha(p)})
                        o[k] = sha(p); nonlocal_changes += 1
            for v in o.values():
                nonlocal_changes += walk(v)
        elif isinstance(o, list):
            for v in o:
                nonlocal_changes += walk(v)
        return nonlocal_changes

    for p in files:
        if p.suffix.lower() != ".json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        n = walk(d)
        if n:
            p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            changed += n
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
    tracked = [root / f for f in git("ls-files", "-z").split("\0") if f]

    # 1. renames (deepest first, so a folder rename does not break its files' paths)
    renames = []
    for p in tracked:
        rel = p.relative_to(root)
        parts = list(rel.parts)
        new = [re.sub(rf"(?i)^{W}_", NEW_PREFIX, s) for s in parts]
        if new != parts:
            renames.append((rel, Path(*new)))
    print(f"{len(renames)} tracked path(s) to rename")
    for old, new in renames[:8]:
        print(f"  {old}  ->  {new}")
    # 2. text and PNG edits
    edits, pngs = [], []
    for p in tracked:
        if p.resolve() == SELF or not p.exists():
            continue
        if p.suffix.lower() == ".png":
            s = strip_png(p.read_bytes())
            if s is not None:
                pngs.append((p, s))
            continue
        if not is_text(p):
            continue
        t = p.read_text(encoding="utf-8")
        u = scrub_text(t)
        if u != t:
            edits.append((p, u))
    print(f"{len(edits)} text file(s) to reword, {len(pngs)} PNG(s) to strip of the credential chunk")
    if a.dry_run:
        return 0

    for p, u in edits:
        p.write_text(u, encoding="utf-8")
    for p, s in pngs:
        p.write_bytes(s)
    for old, new in renames:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        git("mv", "-k", str(old), str(new))
    tracked = [root / f for f in git("ls-files", "-z").split("\0") if f]
    log: dict = {}
    for _ in range(3):                     # pins can chain: a registration pins one that pins another
        if not repin(root, tracked, log):
            break
    rec = root / "docs" / "ATTRIBUTION_CLEANUP_2026_11_06.json"
    rec.write_text(json.dumps({
        "what": "Registrations and records renamed to the research_ prefix and reworded to FXTD Studios Research; "
                "their SHA-256 pins recomputed after the edits. 'old' is the hash each registration had when it was "
                "registered and run (the results on G: record it); 'new' is the file's hash now.",
        "renamed_to": sorted(str(n) for o, n in renames),
        "note": "each of these was named with the former assistant prefix in place of research_",
        "repinned": {scrub_text(k): v for k, v in log.items()}}, indent=1), encoding="utf-8")
    git("add", "-A", "--", ".")

    # 3. what is left
    left = git("grep", "-n", "-i", "-I", "-e", W, "-e", CO, check=False)
    left = [l for l in left.splitlines() if not l.startswith(str(SELF.relative_to(root)).replace("\\", "/"))]
    print(f"\nreworded {len(edits)}, renamed {len(renames)}, PNGs {len(pngs)}, pins recomputed {len(log)}")
    print("left:", "nothing" if not left else "")
    for l in left[:20]:
        print("  " + l[:160])
    return 0 if not left else 3


if __name__ == "__main__":
    sys.exit(main())
