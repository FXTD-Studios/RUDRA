#!/usr/bin/env bash
# ONNX Runtime's C/C++ API from NuGet (1.2.6, 19 Nov 2026).
#
# github.com/microsoft/onnxruntime, its releases included, has answered 404 since 17-18 Nov 2026
# (the repository is gone from GitHub; Microsoft has not said why), so the release archives the
# builds and CI used (onnxruntime-<platform>-<version>.tgz) can no longer be downloaded. NuGet's
# Microsoft.ML.OnnxRuntime carries the same libraries (the Linux soname libonnxruntime.so.1, the
# macOS install name @rpath/libonnxruntime.<version>.dylib, Core ML included) and the headers, as the
# Windows build already takes them. This lays the package out as the archive was:
#   <dest>/include/*.h
#   <dest>/lib/libonnxruntime.so -> .so.1 -> .so.<version>   (Linux)
#   <dest>/lib/libonnxruntime.dylib -> libonnxruntime.<version>.dylib   (macOS)
# so ONNXRUNTIME_ROOT=<dest> works as before. The package is checked against its SHA-256 below.
#
#   tools/fetch_onnxruntime.sh 1.24.4 linux-x64 tmp/native_deps/onnxruntime-linux-x64-1.24.4
#   tools/fetch_onnxruntime.sh 1.24.4 osx-arm64 onnxruntime-osx-arm64-1.24.4
# Platforms: linux-x64, linux-arm64, osx-arm64. Another version needs ORT_NUPKG_SHA256=<its hash>;
# ORT_NUPKG_URL fetches the package from elsewhere (a mirror), checked the same way.
# The NuGet package has no coreml_provider_factory.h; infer/src/onnxruntime_backend.cpp reaches
# Core ML through AppendExecutionProvider("CoreML") then.
set -euo pipefail
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }
[ $# -eq 3 ] || fail "usage: $0 <version> <linux-x64|linux-arm64|osx-arm64> <dest>"
ver=$1 plat=$2 dest=$3
case "$plat" in linux-x64|linux-arm64|osx-arm64) ;; *) fail "platform $plat: linux-x64, linux-arm64 or osx-arm64" ;; esac
case "$ver" in
  # https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.24.4/microsoft.ml.onnxruntime.1.24.4.nupkg,
  # 125194303 bytes, read 19 Nov 2026 (its linux-x64 and osx-arm64 libraries and headers checked then).
  1.24.4) sha=4b978d5065b85e7004b6c6f60ca494bd978fbe6836cbf0a0b52d82b61ab99638 ;;
  *) sha=${ORT_NUPKG_SHA256:-} ;;
esac
[ -n "$sha" ] || fail "no SHA-256 pinned for ONNX Runtime $ver: set ORT_NUPKG_SHA256"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
url=${ORT_NUPKG_URL:-"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$ver/microsoft.ml.onnxruntime.$ver.nupkg"}   # ORT_NUPKG_URL: a mirror (still checked)
retry=(--retry 5)
case "$(curl --help all 2>/dev/null)" in *--retry-all-errors*) retry+=(--retry-all-errors) ;; esac
echo "ONNX Runtime $ver ($plat) from NuGet"
curl -fsSL "${retry[@]}" -o "$tmp/ort.nupkg" "$url" || fail "download $url"
if command -v sha256sum >/dev/null; then got=$(sha256sum "$tmp/ort.nupkg" | cut -d' ' -f1)
else got=$(shasum -a 256 "$tmp/ort.nupkg" | cut -d' ' -f1); fi
[ "$got" = "$sha" ] || fail "microsoft.ml.onnxruntime.$ver.nupkg has SHA-256 $got, expected $sha"

mkdir -p "$tmp/x"
if command -v unzip >/dev/null; then
  unzip -q "$tmp/ort.nupkg" 'build/native/include/*' "runtimes/$plat/native/*" -d "$tmp/x"
else
  py=$(command -v python3 || command -v python3.12 || command -v python) || fail "no unzip and no Python to unpack the package"
  "$py" - "$tmp/ort.nupkg" "$tmp/x" "$plat" <<'EOF'
import sys, zipfile
src, out, plat = sys.argv[1:]
with zipfile.ZipFile(src) as z:
    for n in z.namelist():
        if n.startswith("build/native/include/") or n.startswith(f"runtimes/{plat}/native/"):
            z.extract(n, out)
EOF
fi
native="$tmp/x/runtimes/$plat/native"
[ -f "$tmp/x/build/native/include/onnxruntime_cxx_api.h" ] || fail "the package has no C++ headers"
rm -rf "$dest"
mkdir -p "$dest/include" "$dest/lib"
cp "$tmp/x/build/native/include/"*.h "$dest/include/"
case "$plat" in
  linux-*)
    [ -f "$native/libonnxruntime.so" ] || fail "the package has no $plat library"
    cp "$native/libonnxruntime.so" "$dest/lib/libonnxruntime.so.$ver"
    ln -s "libonnxruntime.so.$ver" "$dest/lib/libonnxruntime.so.1"
    ln -s "libonnxruntime.so.1" "$dest/lib/libonnxruntime.so"
    for f in "$native"/libonnxruntime_providers_*.so; do [ -e "$f" ] && cp "$f" "$dest/lib/"; done
    ;;
  osx-*)
    [ -f "$native/libonnxruntime.dylib" ] || fail "the package has no $plat library"
    cp "$native/libonnxruntime.dylib" "$dest/lib/libonnxruntime.$ver.dylib"
    ln -s "libonnxruntime.$ver.dylib" "$dest/lib/libonnxruntime.dylib"
    ;;
esac
echo "ONNX Runtime $ver at $dest"
