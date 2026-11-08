#!/usr/bin/env python3
"""The visual review of a RUDRA ACES 2 release candidate (7 Nov 2026).

Puts masters of the same frames side by side, as the colourist sees them: each master
through the model's own ACES 2.0 SDR 100-nit view at -3, 0 and +3 EV (the grading
exposures the gates measure), beside the source. Typical columns: the analytic inverse
(`rudra-native aces-v2 ... --tau-high 256 --tau-low -1`), the model without its lock
(`--no-lock`) and the release candidate. Writes:

    <out>/index.html      one row per frame and exposure, click an image for full size
    <out>/<stem>/*.png    the views
    <out>/review.json     the sign-off, to fill in: approved, reviewer, date, notes, and
                          per frame "ok" / "issue"; tools/promote_aces2_release.py reads it

    python tools/aces2_review_kit.py --source <frames> --out reports/aces2_review ^
        --master analytic=<run>/review_analytic --master no_lock=<run>/review_nolock ^
        --master candidate=<run>/master --package dist/models/RUDRA_ACES_2_w4

It only reads masters; it never writes into them. A review on an HDR monitor (the EXRs
in Resolve or Nuke, ACEScg) is the other half of the sign-off; note it in review.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_aces_masters import STILLS, read_master  # noqa: E402

EVS = (-3, 0, 3)


def to_png(path: Path, codes01: np.ndarray) -> None:
    """Write display codes in [0, 1] (HWC RGB) as an 8-bit PNG."""
    import cv2
    img = np.clip(np.rint(codes01 * 255.0), 0, 255).astype(np.uint8)[..., ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"cannot write {path}")


def sha256(p: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_masters(specs: list[str]) -> list[tuple[str, Path]]:
    """label=folder pairs, in the order given; labels unique, folders present."""
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--master takes label=folder, got {s!r}")
        label, folder = s.split("=", 1)
        if not Path(folder).is_dir():
            raise SystemExit(f"no master folder {folder}")
        out.append((label, Path(folder)))
    if len({l for l, _ in out}) != len(out):
        raise SystemExit("master labels must be unique")
    return out


def main(argv: list[str] | None = None) -> int:
    """Render the views, the page and the sign-off template."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="the source stills the masters were made from")
    ap.add_argument("--master", action="append", required=True, help="label=folder of <stem>.exr masters (repeat)")
    ap.add_argument("--package", type=Path, help="the candidate package (its manifest is recorded in review.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0: every frame")
    a = ap.parse_args(argv)
    from training.aces_reference import reference_view
    view = reference_view()
    masters = parse_masters(a.master)
    sources = sorted(p for p in a.source.iterdir() if p.suffix.lower() in STILLS)
    sources = [p for p in sources if all((f / f"{p.stem}.exr").is_file() for _, f in masters)]
    if a.max_frames:
        sources = sources[: a.max_frames]
    if not sources:
        raise SystemExit("no source frame has a master in every folder")
    if (a.out / "review.json").exists():
        raise SystemExit(f"{a.out / 'review.json'} exists: a review is not overwritten; choose a new --out")
    rows, frames = [], []
    for src in sources:
        import cv2
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]
        scale = 65535.0 if img.dtype == np.uint16 else 255.0
        to_png(a.out / src.stem / "source.png", img.astype(np.float32) / scale)
        cells = {}
        for label, folder in masters:
            rgb, _ = read_master(folder / f"{src.stem}.exr")
            for ev in EVS:
                shown = np.asarray(view.forward((rgb * np.float32(2.0 ** ev)).astype(np.float32)), np.float32)
                name = f"{label}_ev{ev:+d}.png"
                to_png(a.out / src.stem / name, shown)
                cells[(label, ev)] = f"{src.stem}/{name}"
        frames.append({"frame": src.name, "source_sha256": sha256(src), "verdict": "", "notes": ""})
        for ev in EVS:
            imgs = "".join(f'<td><a href="{html.escape(cells[(l, ev)])}"><img src="{html.escape(cells[(l, ev)])}"></a></td>'
                           for l, _ in masters)
            src_cell = (f'<td><a href="{src.stem}/source.png"><img src="{src.stem}/source.png"></a></td>' if ev == 0 else "<td></td>")
            rows.append(f"<tr><th>{html.escape(src.name)}<br>{ev:+d} EV</th>{src_cell}{imgs}</tr>")
    head = "".join(f"<th>{html.escape(l)}</th>" for l, _ in masters)
    page = f"""<!doctype html><meta charset="utf-8"><title>RUDRA ACES 2 review</title>
<style>body{{font:14px system-ui;margin:16px;background:#111;color:#ddd}}img{{width:320px;display:block}}
td,th{{padding:4px;vertical-align:top}}th{{text-align:left}}</style>
<h1>RUDRA ACES 2: visual review</h1>
<p>Each master through the ACES 2.0 SDR 100-nit view at -3, 0 and +3 EV. Look at the highlights: shape,
colour, banding, edges, anything the analytic inverse does better. Record the verdict in review.json.</p>
<table><tr><th></th><th>source</th>{head}</tr>{''.join(rows)}</table>"""
    (a.out / "index.html").write_text(page, encoding="utf-8")
    review = {
        "schema": "rudra_aces2_visual_review_v1",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "approved": False,
        "reviewer": "",
        "date": "",
        "hdr_monitor_review": "",
        "notes": "",
        "package_manifest_sha256": sha256(a.package / "manifest.json") if a.package else None,
        "masters": {l: str(f) for l, f in masters},
        "frames": frames,
        "how": "Set approved to true, your name, the date, and what you looked at on the HDR monitor. "
               "Give every frame a verdict ('ok' or 'issue') and a note for each issue.",
    }
    (a.out / "review.json").write_text(json.dumps(review, indent=1), encoding="utf-8")
    print(f"{len(sources)} frames x {len(EVS)} exposures x {len(masters)} masters -> {a.out / 'index.html'}")
    print(f"sign-off: {a.out / 'review.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
