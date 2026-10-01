"""Real SDR / HDR10 frame pairs from Netflix Open Content, fetched frame by frame.

Source: ``aom_test_materials`` on the public Open Content bucket. Netflix cut
36 shots from five titles and published each twice, frame for frame:

  HDR/<title>/<first>-<last>.zip   12-bit RGB JPEG 2000, P3-D65, ST-2084 (PQ)
  SDR/<title>/<first>-<last>.zip   10-bit Y'CbCr 4:2:0 JPEG 2000, BT.709, made
                                   from the HDR master with the Dolby Vision
                                   trim pass (a professional SDR derivative)

Both zips of a shot hold the same member names (``frame00000031.j2k``), so a
pair is the two members with one name. Only the sampled frames are read: the
zip central directory and each member come over HTTP range requests, so a
2,000-pair corpus costs ~20 GB of transfer instead of the ~100 GB of zips.

Licence: every title folder on the bucket carries CC BY 4.0 (checked 1 Oct
2026; the NC-ND links in the old aom README are dead). Attribution:
"Netflix Open Content" plus the title, kept per record.

Conversion (each step is the repo's own convention, not a new one):

  SDR  Y'CbCr limited/full range (measured per frame, recorded) -> R'G'B' 709
       -> inverse BT.709 OETF (canonicalize_sdr 'rec709', as rudra/video.py
       reads a bt709 stream) -> linear 709 -> 2x area downsample in LINEAR
       light -> 709->2020 -> sRGB OETF -> uint16 PNG.  Canonical sRGB code in
       BT.2020, what SDR2HDRNet expects.
  HDR  12-bit PQ code / 4095 -> ST-2084 EOTF nits -> 2x area downsample in
       LINEAR light -> P3-D65 -> 2020 -> nits / 203 -> pipeline.hdr_io
       ``log2_extended`` -> uint16 PNG.  Sentinel _ingest_config.json written
       first, so every loader decodes it through hdr_io.

Downsampling both sides in linear light, from the full-resolution decode, is
deliberate: averaging PQ codes and gamma codes separately would make the two
images disagree at every edge, and edges around highlights are the signal.

Resumable: a pair whose two PNGs exist and whose record is in
``_records.jsonl`` is skipped. Re-running with the same arguments rebuilds the
same selection (it depends only on the bucket listing and the strides).

    python pipeline/fetch_netflix_pairs.py --out G:/datasets/corpora/rudra_netflix_realsdr_20261001
    python pipeline/fetch_netflix_pairs.py --out ... --finalize      # manifest + report only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.hdr_io import HDR_IO_VERSION, HDRStorage, encode_hdr_u16, pq_eotf  # noqa: E402
from pipeline.licences import annotate  # noqa: E402
from rudra.delivery.colorspace import rgb_to_rgb_matrix  # noqa: E402

BUCKET = "http://download.opencontent.netflix.com.s3.amazonaws.com/"
PREFIX = "aom_test_materials/"
DIFFUSE_WHITE = 203.0

# Split by title, so no title is in two splits. Sol Levante and Sparks were
# both touched on 1 Oct 2026 (H6 eval, real-SDR pilot), so they train; Meridian
# (live action, never used) is the untouched test title; Cosmos Laundromat
# (animation, never used) is validation.
SPLITS = {"nocturne": "train", "sparks": "train", "sol_levante": "train",
          "cosmos": "val", "meridian": "test"}
# Frame stride per split. Consecutive frames at 24-60 fps are near-duplicates;
# these strides give ~1,900 train, ~170 val, ~300 test pairs.
STRIDES = {"train": 4, "val": 8, "test": 10}
TITLE_NAMES = {"nocturne": "Nocturne", "sparks": "Sparks", "sol_levante": "Sol Levante",
               "cosmos": "Cosmos Laundromat", "meridian": "Meridian"}

STORAGE = HDRStorage(mode="log2_extended")
M_709_2020 = rgb_to_rgb_matrix("rec709", "rec2020").astype(np.float32)
M_P3_2020 = rgb_to_rgb_matrix("p3d65", "rec2020").astype(np.float32)

# Frames with no HDR content are dropped by the same rule the corpus builder
# uses ("pairs peaking below 1 nit -- no HDR content to learn"), plus fades /
# title cards with an SDR mean under 2% code. Dark night frames that do carry
# highlights stay: shadows are where real-SDR training has measured gains.
MIN_SDR_MEAN = 0.02          # canonical sRGB code
MIN_HDR_PEAK_NITS = 1.0


def is_dark(rec: dict) -> bool:
    return rec["sdr_mean_code"] < MIN_SDR_MEAN or rec["peak_nits"] < MIN_HDR_PEAK_NITS


# ----------------------------------------------------------------------------
# HTTP + zip, by range
# ----------------------------------------------------------------------------
_CONN = threading.local()


def _http_get(url: str, headers: dict) -> bytes:
    """GET on a per-thread kept-alive connection; honours HTTP(S)_PROXY."""
    import http.client
    parts = urllib.parse.urlsplit(url)
    proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    key = (parts.netloc, proxy)
    conn = getattr(_CONN, "conn", None)
    if conn is None or getattr(_CONN, "key", None) != key:
        if proxy:
            p = urllib.parse.urlsplit(proxy)
            conn = http.client.HTTPConnection(p.hostname, p.port or 80, timeout=120)
        else:
            conn = http.client.HTTPConnection(parts.netloc, timeout=120)
        _CONN.conn, _CONN.key = conn, key
    target = url if proxy else (parts.path + ("?" + parts.query if parts.query else ""))
    try:
        conn.request("GET", target, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
    except Exception:
        conn.close()
        _CONN.conn = None
        raise
    if resp.status not in (200, 206):
        raise IOError(f"HTTP {resp.status} for {url}")
    return data


def _get(url: str, start: int | None = None, end: int | None = None, tries: int = 6) -> bytes:
    for attempt in range(tries):
        try:
            headers = {"Range": f"bytes={start}-{end}"} if start is not None else {}
            data = _http_get(url, headers)
            if start is not None and len(data) != end - start + 1 and not _at_eof(url, start, len(data)):
                raise IOError(f"short read {len(data)} of {end - start + 1}")
            return data
        except Exception:  # noqa: BLE001 - network: retry with backoff
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


_SIZES: dict[str, int] = {}


def _at_eof(url: str, start: int, got: int) -> bool:
    """A range that runs past the end of the object is legally short."""
    return start + got == _size(url)


def _size(url: str) -> int:
    if url in _SIZES:
        return _SIZES[url]
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        _SIZES[url] = int(r.headers["Content-Length"])
    return _SIZES[url]


def list_shots() -> dict[tuple[str, str], dict[str, str]]:
    """{(title, range): {'HDR': key, 'SDR': key}} for shots present in both."""
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys, token = [], None
    while True:
        q = f"?list-type=2&prefix={urllib.parse.quote(PREFIX)}"
        if token:
            q += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(_get(BUCKET + q))
        keys += [c.find("s:Key", ns).text for c in root.findall("s:Contents", ns)]
        if root.find("s:IsTruncated", ns).text != "true":
            break
        token = root.find("s:NextContinuationToken", ns).text
    shots: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for k in keys:
        parts = k.split("/")
        if len(parts) == 4 and parts[1] in ("HDR", "SDR") and parts[3].endswith(".zip"):
            shots[(parts[2], parts[3][:-4])][parts[1]] = k
    return {s: v for s, v in sorted(shots.items()) if set(v) == {"HDR", "SDR"} and s[0] in SPLITS}


def zip_index(url: str) -> dict[str, tuple[int, int, int, int]]:
    """member name -> (local header offset, compressed size, method, crc32)."""
    size = _size(url)
    tail = _get(url, max(0, size - 65_557), size - 1)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise ValueError(f"no end-of-central-directory in {url}")
    cd_size, cd_off = struct.unpack("<II", tail[pos + 12:pos + 20])
    if cd_off == 0xFFFFFFFF:
        raise ValueError(f"zip64 not handled: {url}")
    cd = _get(url, cd_off, cd_off + cd_size - 1)
    out, i = {}, 0
    while i < len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[i + 10:i + 12])
        crc, csize = struct.unpack("<II", cd[i + 16:i + 24])
        nlen, elen, clen = struct.unpack("<HHH", cd[i + 28:i + 34])
        off, = struct.unpack("<I", cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nlen].decode("utf-8")
        if not name.endswith("/"):
            out[name.rsplit("/", 1)[-1]] = (off, csize, method, crc)
        i += 46 + nlen + elen + clen
    return out


def fetch_member(url: str, entry: tuple[int, int, int, int]) -> bytes:
    """One range request: local header + name + extra (slack 1 KiB) + data."""
    off, csize, method, crc = entry
    blob = _get(url, off, off + 30 + 1024 + csize - 1)
    if blob[:4] != b"PK\x03\x04":
        raise IOError(f"no local file header at {off} in {url}")
    nlen, elen = struct.unpack("<HH", blob[26:30])
    start = 30 + nlen + elen
    if start + csize > len(blob):
        blob += _get(url, off + len(blob), off + start + csize - 1)
    data = blob[start:start + csize]
    if method == 8:
        data = zlib.decompress(data, -15)
    elif method != 0:
        raise ValueError(f"zip method {method} not handled")
    if zlib.crc32(data) & 0xFFFFFFFF != crc:
        raise IOError(f"CRC mismatch for member at {off} in {url}")
    return data


# ----------------------------------------------------------------------------
# Decode + convert
# ----------------------------------------------------------------------------
def _siz(j2k: bytes) -> tuple[int, int]:
    """Image width/height from the JPEG 2000 SIZ marker (no ffprobe round trip)."""
    i = j2k.find(b"\xff\x51", 2, 256)
    if i < 0:
        raise ValueError("no SIZ marker")
    xsiz, ysiz, xo, yo = struct.unpack(">IIII", j2k[i + 6:i + 22])
    return xsiz - xo, ysiz - yo


def _ffmpeg_raw(j2k: bytes, pix_fmt: str) -> tuple[np.ndarray, int, int]:
    w, h = _siz(j2k)
    with tempfile.NamedTemporaryFile(suffix=".j2k", delete=False) as f:
        f.write(j2k)
        path = f.name
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                              "-pix_fmt", pix_fmt, "-"], capture_output=True, check=True).stdout
    finally:
        os.unlink(path)
    return np.frombuffer(raw, dtype="<u2"), w, h


def _bt709_inverse_oetf(v: np.ndarray) -> np.ndarray:
    return np.where(v < 0.081, v / 4.5, np.power((v + 0.099) / 1.099, 1.0 / 0.45))


def _srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _half(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def convert_sdr(j2k: bytes) -> tuple[np.ndarray, dict]:
    raw, w, h = _ffmpeg_raw(j2k, "yuv420p10le")
    y = raw[: w * h].reshape(h, w).astype(np.float32)
    cb = raw[w * h: w * h + (w // 2) * (h // 2)].reshape(h // 2, w // 2).astype(np.float32)
    cr = raw[w * h + (w // 2) * (h // 2):].reshape(h // 2, w // 2).astype(np.float32)
    cb = cv2.resize(cb, (w, h), interpolation=cv2.INTER_LINEAR)
    cr = cv2.resize(cr, (w, h), interpolation=cv2.INTER_LINEAR)
    # Range from the data: a limited-range 10-bit signal never uses codes < 4
    # or > 1019 except in footroom/headroom excursions; the recorded min/max
    # make the decision auditable per frame.
    ymin, ymax = float(np.percentile(y, 0.01)), float(np.percentile(y, 99.99))
    limited = RANGE_OVERRIDE == "limited" or (RANGE_OVERRIDE == "auto" and ymin >= 32 and ymax <= 960)
    if limited:
        yn, cbn, crn = (y - 64.0) / 876.0, (cb - 512.0) / 896.0, (cr - 512.0) / 896.0
    else:
        yn, cbn, crn = y / 1023.0, (cb - 512.0) / 1023.0, (cr - 512.0) / 1023.0
    r = yn + 1.5748 * crn
    g = yn - 0.187324 * cbn - 0.468124 * crn
    b = yn + 1.8556 * cbn
    rgb = np.clip(np.stack([r, g, b], -1), 0.0, 1.0).astype(np.float32)
    clipped = float((rgb.max(-1) >= 0.999).mean())
    lin = _bt709_inverse_oetf(rgb.astype(np.float32)).astype(np.float32)
    lin = _half(lin)
    lin2020 = np.clip(lin @ M_709_2020.T, 0.0, 1.0)
    code = _srgb_oetf(lin2020)
    u16 = np.rint(code * 65535.0).astype(np.uint16)
    return u16, {"sdr_range": "limited" if limited else "full", "sdr_y_p0.01": ymin,
                 "sdr_y_p99.99": ymax, "sdr_clipped_fraction": clipped,
                 "sdr_mean_code": float(code.mean()), "source_size": [w, h]}


def convert_hdr(j2k: bytes) -> tuple[np.ndarray, dict]:
    # OpenJPEG through OpenCV (multithreaded, ~5x faster than ffmpeg's decoder
    # here); ffmpeg is the fallback. The two irreversible-9/7 decoders differ by
    # at most 1 of 4,095 codes (~0.004 stops); the record says which was used.
    img = cv2.imdecode(np.frombuffer(j2k, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3 and img.dtype == np.uint16 and int(img.max()) <= 4095:
        code12 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.int32)
        h, w = code12.shape[:2]
        decoder = "openjpeg(opencv)"
    else:
        raw, w, h = _ffmpeg_raw(j2k, "rgb48le")
        # ffmpeg widens 12-bit samples to 16 bits by bit replication.
        code12 = (raw.reshape(h, w, 3) >> 4).astype(np.int32)
        decoder = "ffmpeg"
    # 4,096 codes, so the PQ EOTF is an exact lookup.
    nits = _PQ_LUT[code12]
    nits = np.maximum(_half(nits), 0.0)
    nits2020 = np.maximum(nits @ M_P3_2020.T, 0.0)
    linear = nits2020 / DIFFUSE_WHITE
    u16, stats = encode_hdr_u16(linear, STORAGE)
    lum = nits2020 @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    stats.update({"hdr_decoder": decoder, "hdr_p99_nits": float(np.percentile(lum, 99)),
                  "hdr_mean_nits": float(lum.mean()), "source_size": [w, h]})
    return u16, stats


RANGE_OVERRIDE = "auto"
_PQ_LUT = pq_eotf(np.arange(4096) / 4095.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------
def plan(shots: dict) -> list[dict]:
    """Deterministic selection: every stride-th common frame, offset stride//2."""
    jobs = []
    for (title, rng), keys in shots.items():
        split = SPLITS[title]
        stride = STRIDES[split]
        hdr_idx = zip_index(BUCKET + keys["HDR"])
        sdr_idx = zip_index(BUCKET + keys["SDR"])
        common = sorted(set(hdr_idx) & set(sdr_idx))
        for name in common[stride // 2::stride]:
            stem = name.rsplit(".", 1)[0]
            scene = f"nf_{title}_{rng}"
            jobs.append({"title": title, "shot": rng, "split": split, "member": name,
                         "asset_id": f"{scene}_{stem}", "scene_id": scene,
                         "hdr_url": BUCKET + keys["HDR"], "sdr_url": BUCKET + keys["SDR"],
                         "hdr_entry": hdr_idx[name], "sdr_entry": sdr_idx[name],
                         "shot_frames": len(common), "stride": stride})
    return jobs


def write_sentinel(out: Path) -> None:
    sentinel = out / "_ingest_config.json"
    payload = {
        "corpus": "rudra_netflix_realsdr",
        "created": time.strftime("%Y-%m-%d"),
        "storage": STORAGE.as_dict(),
        "hdr_io_version": HDR_IO_VERSION,
        "sdr_convention": "canonical sRGB code in BT.2020, from BT.709 Y'CbCr via inverse "
                          "BT.709 OETF (rudra/video.py 'rec709'), linear-light 2x area downsample",
        "hdr_convention": "P3-D65 PQ 12-bit -> nits -> linear-light 2x area downsample -> "
                          "BT.2020 -> /203 -> log2_extended",
        "source": BUCKET + PREFIX,
        "licence": "CC BY 4.0, Netflix Open Content (licence file in every title folder, 1 Oct 2026)",
        "attribution": "Contains Netflix Open Content: Nocturne, Sparks, Sol Levante, "
                       "Cosmos Laundromat (Blender Foundation), Meridian. CC BY 4.0.",
        "splits": SPLITS, "strides": STRIDES,
        "source_curve": "unknown",
        "tool": "pipeline/fetch_netflix_pairs.py",
    }
    if sentinel.exists():
        old = json.loads(sentinel.read_text(encoding="utf-8"))
        if old.get("storage") != payload["storage"]:
            raise SystemExit(f"{sentinel} records a different storage; refusing to mix")
        return
    sentinel.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process(job: dict, out: Path, hdr_bytes: bytes | None = None, sdr_bytes: bytes | None = None) -> dict:
    sdr_path = out / "pairs" / "sdr" / f"{job['asset_id']}.png"
    hdr_path = out / "pairs" / "hdr" / f"{job['asset_id']}.png"
    if hdr_bytes is None:
        hdr_bytes = fetch_member(job["hdr_url"], tuple(job["hdr_entry"]))
    if sdr_bytes is None:
        sdr_bytes = fetch_member(job["sdr_url"], tuple(job["sdr_entry"]))
    sdr, sstats = convert_sdr(sdr_bytes)
    hdr, hstats = convert_hdr(hdr_bytes)
    if sdr.shape != hdr.shape:
        raise ValueError(f"{job['asset_id']}: geometry {sdr.shape} vs {hdr.shape}")
    sdr, hdr, crop = trim_letterbox(sdr, hdr)
    sstats["letterbox_crop_yxhw"] = crop
    dark = is_dark({"sdr_mean_code": sstats["sdr_mean_code"], "peak_nits": hstats["peak_nits"]})
    rec = {
        "asset_id": job["asset_id"], "scene_id": job["scene_id"], "split": job["split"],
        "title": TITLE_NAMES[job["title"]], "shot": job["shot"],
        "frame_member": job["member"], "frame_index": int(job["member"][5:13]),
        "sdr_path": f"pairs/sdr/{job['asset_id']}.png", "hdr_path": f"pairs/hdr/{job['asset_id']}.png",
        "width": int(sdr.shape[1]), "height": int(sdr.shape[0]),
        "sdr_kind": "real", "sdr_source": "Dolby Vision trim pass, BT.709 10-bit",
        "source_curve": "unknown", "tonemap_ev": None,
        "hdr_source": "P3-D65 PQ 12-bit master",
        "source_path": f"netflix/{job['title']}/{job['shot']}/{job['member']}",
        "hdr_url": job["hdr_url"], "sdr_url": job["sdr_url"],
        "hdr_member_crc32": f"{job['hdr_entry'][3]:08x}", "sdr_member_crc32": f"{job['sdr_entry'][3]:08x}",
        "excluded": "near_black" if dark else None,
        **{k: v for k, v in sstats.items() if k != "source_size"},
        **{k: v for k, v in hstats.items() if k != "source_size"},
        "source_width": sstats["source_size"][0], "source_height": sstats["source_size"][1],
    }
    if not dark:
        sdr_path.parent.mkdir(parents=True, exist_ok=True)
        hdr_path.parent.mkdir(parents=True, exist_ok=True)
        for path, img in ((sdr_path, sdr), (hdr_path, hdr)):
            # Encode in memory, then one write: on a network/VM mount libpng's
            # many small writes cost ~3x a single buffered write.
            ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not ok:
                raise IOError(f"failed to encode {path}")
            tmp = path.with_suffix(".tmp.png")
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, path)
    return rec


def trim_letterbox(sdr: np.ndarray, hdr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Crop rows/columns that are black in BOTH images (scope bars, pillarbox).

    A line counts as bar when every pixel's SDR code is below 1% and every HDR
    code is at the log2 floor region (< 0.05 nits). Only outer runs are cut,
    and the box is made even so later 2x operations stay aligned.
    """
    floor = np.rint(((np.log2(0.05 / STORAGE.floor_nits) /
                      np.log2(STORAGE.ceiling_nits / STORAGE.floor_nits)) * 65534) + 1)
    blank = (sdr.max(-1) < 655) & (hdr.max(-1) < floor)
    rows = np.where(~blank.all(1))[0]
    cols = np.where(~blank.all(0))[0]
    if rows.size == 0 or cols.size == 0:
        return sdr, hdr, [0, 0, sdr.shape[0], sdr.shape[1]]
    y0, y1 = int(rows[0]) & ~1, int(rows[-1]) + 1
    x0, x1 = int(cols[0]) & ~1, int(cols[-1]) + 1
    y1 += (y1 - y0) & 1
    x1 += (x1 - x0) & 1
    y1, x1 = min(y1, sdr.shape[0]), min(x1, sdr.shape[1])
    return (np.ascontiguousarray(sdr[y0:y1, x0:x1]), np.ascontiguousarray(hdr[y0:y1, x0:x1]),
            [y0, x0, y1 - y0, x1 - x0])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _absolute(root: str, rel: str) -> str:
    windows = "\\" in root or (len(root) > 1 and root[1] == ":")
    sep = "\\" if windows else "/"
    return root.rstrip("\\/") + sep + rel.replace("/", sep)


def finalize(out: Path, root: str | None = None) -> None:
    """Manifest with absolute paths under ``root`` (where the corpus will be READ;
    defaults to --out). Records on disk keep paths relative to the corpus."""
    root = root or str(out.resolve())
    recs = {}
    for line in (out / "_records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["asset_id"]] = r
    for r in recs.values():
        if r["excluded"] and not is_dark(r):
            raise SystemExit(f"{r['asset_id']} is kept by the current rule but was never written; re-run the fetch")
    kept = [r for r in recs.values() if not r["excluded"]]
    kept.sort(key=lambda r: (r["split"], r["asset_id"]))
    for r in kept:
        missing = [k for k in ("sdr_path", "hdr_path") if not (out / r[k]).exists()]
        if missing:
            raise SystemExit(f"{r['asset_id']}: {missing} missing on disk")
        r["sdr_path"], r["hdr_path"] = _absolute(root, r["sdr_path"]), _absolute(root, r["hdr_path"])
    annotate(kept)
    manifest = out / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    by = Counter((r["split"], r["title"]) for r in kept)
    scenes = defaultdict(set)
    for r in kept:
        scenes[r["scene_id"]].add(r["split"])
    report = {
        "manifest": str(manifest), "manifest_sha256": _sha256(manifest), "path_root": root,
        "records": len(kept), "excluded": sorted(r["asset_id"] for r in recs.values() if r["excluded"]),
        "by_split_title": {f"{s}/{t}": n for (s, t), n in sorted(by.items())},
        "by_split": dict(Counter(r["split"] for r in kept)),
        "scenes": len(scenes), "scenes_straddling_splits": [s for s, v in scenes.items() if len(v) > 1],
        "sdr_range": dict(Counter(r["sdr_range"] for r in kept)),
        "commercial_ok": dict(Counter(str(r.get("commercial_ok")) for r in kept)),
        "hdr_peak_nits_p50": float(np.median([r["peak_nits"] for r in kept])) if kept else None,
        "hdr_peak_nits_max": float(max(r["peak_nits"] for r in kept)) if kept else None,
        "sdr_clipped_fraction_mean": float(np.mean([r["sdr_clipped_fraction"] for r in kept])) if kept else None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    global RANGE_OVERRIDE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--titles", nargs="*", default=sorted(SPLITS))
    ap.add_argument("--limit", type=int, default=0, help="stop after N new pairs (0 = all)")
    ap.add_argument("--convert", type=int, default=2, help="concurrent conversions (RAM ~0.5 GB each)")
    ap.add_argument("--downloaders", type=int, default=1,
                    help="concurrent download streams (many parallel range requests get throttled)")
    ap.add_argument("--sdr-range", choices=["auto", "limited", "full"], default="auto")
    ap.add_argument("--finalize", action="store_true", help="only rebuild manifest + report")
    ap.add_argument("--path-root", default=None,
                    help="absolute root written into manifest paths (default: --out)")
    ap.add_argument("--no-finalize", action="store_true", help="fetch only (chunked runs)")
    ap.add_argument("--deadline", type=float, default=0,
                    help="seconds after which no new pair is started; in-flight pairs finish")
    args = ap.parse_args(argv)
    RANGE_OVERRIDE = args.sdr_range
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_sentinel(out)
    if args.finalize:
        finalize(out, args.path_root)
        return 0

    plan_path = out / "_plan.json"
    if plan_path.exists():
        jobs = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        jobs = plan(list_shots())
        plan_path.write_text(json.dumps(jobs), encoding="utf-8")
    jobs = [j for j in jobs if j["title"] in args.titles]
    done = set()
    rec_path = out / "_records.jsonl"
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                on_disk = (out / r["sdr_path"]).exists() and (out / r["hdr_path"]).exists()
                # A record excluded under an older, stricter rule is re-fetched
                # when the current rule keeps it.
                ok = on_disk or (r["excluded"] and is_dark(r))
                if ok:
                    done.add(r["asset_id"])
    todo = [j for j in jobs if j["asset_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(jobs)} planned, {len(done)} done, {len(todo)} to fetch", flush=True)
    t0, n = time.time(), 0
    # One downloader on one kept-alive connection (parallel range requests were
    # throttled to ~0.8 MB/s in total on the 1 Oct 2026 link, one stream ran
    # at 5.4 MB/s), feeding --convert converter threads through a short queue.
    import queue as _queue
    feed: "_queue.Queue" = _queue.Queue(maxsize=args.convert)
    lock = threading.Lock()
    errors: list[BaseException] = []

    source = iter(todo)
    source_lock = threading.Lock()
    live = [args.downloaders]

    def downloader() -> None:
        try:
            while True:
                if (args.deadline and time.time() - t0 > args.deadline) or errors:
                    break
                with source_lock:
                    job = next(source, None)
                if job is None:
                    break
                hb = fetch_member(job["hdr_url"], tuple(job["hdr_entry"]))
                sb = fetch_member(job["sdr_url"], tuple(job["sdr_entry"]))
                feed.put((job, hb, sb))
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
        finally:
            with source_lock:
                live[0] -= 1
                last = live[0] == 0
            if last:
                for _ in range(args.convert):
                    feed.put(None)

    def converter(log) -> None:
        nonlocal n
        while True:
            item = feed.get()
            if item is None:
                return
            try:
                rec = process(item[0], out, item[1], item[2])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return
            with lock:
                log.write(json.dumps(rec) + "\n")
                log.flush()
                n += 1
                if n % 25 == 0:
                    rate = (time.time() - t0) / n
                    print(f"  {n}/{len(todo)}  {rate:.1f} s/pair  eta {rate * (len(todo) - n) / 60:.0f} min", flush=True)

    with open(rec_path, "a", encoding="utf-8", newline="\n") as log:
        threads = [threading.Thread(target=downloader, daemon=True) for _ in range(args.downloaders)]
        threads += [threading.Thread(target=converter, args=(log,), daemon=True) for _ in range(args.convert)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    print(f"fetched {n} pairs this run, {len(done) + n} of {len(jobs)} total", flush=True)
    if not args.no_finalize:
        finalize(out, args.path_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
