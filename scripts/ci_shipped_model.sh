#!/usr/bin/env bash
# The model users download (RUDRA_ACES_2 from Hugging Face), through the native
# engine, checked with tools/check_aces_masters.py. Added for 1.0.3: until now
# CI's Gate A tested the retired sdr2hdr_shadow_v1 and never the shipped model.
#
#   scripts/ci_shipped_model.sh <rudra-native> <runtime[:device]>...
#   scripts/ci_shipped_model.sh build/cli/rudra-native libtorch:cpu onnxruntime:cpu
#   scripts/ci_shipped_model.sh build/cli/rudra-native onnxruntime:cpu onnxruntime:coreml
#
# 1. downloads models/RUDRA_ACES_2 and refuses it unless both files match the
#    published SHA-256 (the same hashes PUBLISH_HF_MODELS.ps1 uploaded);
# 2. stills: 6 frames of an 8-bit BT.709 clip, through every runtime given:
#    R (round trip, 1 code) on each, B (backends, the manifest's tolerance)
#    between the first and every other;
# 3. video: the 8-bit H.264 clip and a 10-bit ProRes 422 HQ made from it, as
#    movies, through the first runtime: R against the engine's own decode.
# Reports in $OUT (default reports/shipped_model). Exit 1 on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."
CLI=${1:?usage: ci_shipped_model.sh <rudra-native> <runtime[:device]>...}; shift
[ $# -ge 1 ] || { echo "give at least one runtime[:device]" >&2; exit 2; }
PY=${PYTHON:-python3}
OUT=${OUT:-reports/shipped_model}
HF_REPO=${HF_REPO:-fxtdstudios/rudra-models}
HF_REVISION=${HF_REVISION:-main}
CLIP=native/tests/golden/video/clips/h264_709.mp4
N=6
# RUDRA ACES 2 (2.0, w4 + lock), published 8 Nov 2026 (docs/MODELS.md; HF ab8da1b2).
# Update both with every publish, or this check refuses the model users get.
G1_TS=7082eb869d265d6e8c4fc209f7cac7e5dce2f40c10df5e73a889fb85efb9dbcc
G1_ONNX=880e7b698a099b8dd60549444392d51af3022efbae8f68f05c9f0f3a29f8ee49
fail() { echo "FAILED: $*" >&2; exit 1; }
say() { printf '\n== %s\n' "$*"; }
rm -rf "$OUT"; mkdir -p "$OUT"
WORK=$OUT/work; mkdir -p "$WORK"

say "RUDRA_ACES_2 from $HF_REPO@$HF_REVISION"
"$PY" - "$HF_REPO" "$HF_REVISION" "$WORK/hf" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
                  allow_patterns=["models/RUDRA_ACES_2/*"])
EOF
PKG=$WORK/hf/models/RUDRA_ACES_2
sha() { "$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
[ "$(sha "$PKG/rudra_g1.ts")" = "$G1_TS" ] || fail "rudra_g1.ts is not the published file"
[ "$(sha "$PKG/rudra_g1.onnx")" = "$G1_ONNX" ] || fail "rudra_g1.onnx is not the published file"
"$PY" -c "import json,sys;m=json.load(open(sys.argv[1]));print('manifest: checkpoint',m.get('checkpoint_sha256'),'release_authorized',m.get('release_authorized'),'onnx limit',m.get('onnx_tolerance',{}).get('limit_display_codes'))" "$PKG/manifest.json"

# The engine's decode (native/media/src/aces_frames.cpp aces_decoder_command):
# swscale, accurate rounding, full chroma, the stream's matrix and range, rgb48.
decode() {   # decode <video> <dir> : first $N frames, <stem>_<NNNNNN>.png from 0, 16-bit
  local v=$1 d=$2 range stem
  range=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_range -of csv=p=0 "$v")
  [ "$range" = "pc" ] && range=pc || range=tv
  stem=$(basename "${v%.*}")
  mkdir -p "$d"
  ffmpeg -hide_banner -v error -nostdin -noautorotate -i "$v" -map 0:v:0 \
    -vf "scale=in_color_matrix=bt709:in_range=$range:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb48le" \
    -fps_mode passthrough -frames:v $N -pix_fmt rgb48be -start_number 0 "$d/${stem}_%06d.png"
}

say "Frames"
decode "$CLIP" "$WORK/ref_h264"
STILLS=$WORK/stills; mkdir -p "$STILLS"
"$PY" - "$WORK/ref_h264" "$STILLS" <<'EOF'
import sys, pathlib, cv2, numpy as np
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.png")):   # 8-bit stills, rounded as the engine rounds
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(pathlib.Path(sys.argv[2]) / p.name), np.rint(img.astype(np.float64) / 257.0).astype(np.uint8))
EOF
PRORES=$WORK/prores10_709.mov
ffmpeg -hide_banner -v error -nostdin -i "$CLIP" -frames:v $N -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv -an "$PRORES"
decode "$PRORES" "$WORK/ref_prores"

run() {   # run <runtime:device> <input> <outdir> [extra...]
  local rt=${1%%:*} dev=cpu in=$2 o=$3
  case $1 in *:*) dev=${1#*:} ;; esac
  shift 3
  "$CLI" aces-v2 "$PKG" "$in" --out "$o" --runtime "$rt" --device "$dev" --ocio-threads 4 "$@" \
    > "$o.log" 2>&1 || { cat "$o.log"; fail "rudra-native aces-v2 --runtime $rt --device $dev on $(basename "$in")"; }
}

ok=1
FIRST=$1
say "Stills through every runtime"
for r in "$@"; do
  tag=${r//:/_}
  run "$r" "$STILLS" "$OUT/stills_$tag"
  "$PY" tools/check_aces_masters.py --source "$STILLS" --master "$OUT/stills_$tag" --package "$PKG" \
    --report "$OUT/R_stills_$tag.json" || ok=0
  if [ "$r" != "$FIRST" ]; then
    "$PY" tools/check_aces_masters.py --source "$STILLS" --master "$OUT/stills_${FIRST//:/_}" \
      --other "$OUT/stills_$tag" --package "$PKG" --report "$OUT/B_${FIRST//:/_}_vs_$tag.json" || ok=0
  fi
done

say "Movies through $FIRST"
run "$FIRST" "$CLIP" "$OUT/video_h264" --range 0:$((N - 1))
"$PY" tools/check_aces_masters.py --source "$WORK/ref_h264" --master "$OUT/video_h264" --round-codes \
  --report "$OUT/R_video_h264.json" || ok=0
run "$FIRST" "$PRORES" "$OUT/video_prores10"
# 10-bit codes fall between 8-bit ones; a few may sit outside what the SDR view
# can show, so the share within 1 code is checked, not every pixel.
"$PY" tools/check_aces_masters.py --source "$WORK/ref_prores" --master "$OUT/video_prores10" \
  --r-fraction 0.999 --report "$OUT/R_video_prores10.json" || ok=0

rm -rf "$WORK/hf"
[ "$ok" = 1 ] || fail "the shipped model failed a check (reports in $OUT)"
echo "PASS: the shipped RUDRA_ACES_2 on $* (reports in $OUT)"
