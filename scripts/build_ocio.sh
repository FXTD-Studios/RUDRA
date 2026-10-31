#!/usr/bin/env bash
# OpenColorIO 2.5.2 for the learned ACES 2.0 engine (1 Nov 2026), macOS or Linux.
#
#   scripts/build_ocio.sh                 # -> tmp/native_deps/ocio-2.5.2 (once; later runs reuse it)
#
# The ACES 2.0 model is pinned to OCIO 2.5.2's built-in ACES configs (aces_v2::make_ocio_view
# refuses another version), and no package manager ships exactly that, so it is built here:
# one shared library, its own dependencies (yaml-cpp, pystring, expat, Imath, minizip-ng, zlib)
# fetched and linked inside it, no apps, no Python, no tests. A few minutes, once.
set -euo pipefail
cd "$(dirname "$0")/.."
OCIO_VERSION=${OCIO_VERSION:-2.5.2}
DEPS=tmp/native_deps
PREFIX=$PWD/$DEPS/ocio-$OCIO_VERSION
JOBS=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)
mkdir -p "$DEPS"

if [ -f "$PREFIX/lib/cmake/OpenColorIO/OpenColorIOConfig.cmake" ]; then
  echo "OpenColorIO $OCIO_VERSION at $PREFIX"
  exit 0
fi
SRC=$DEPS/OpenColorIO-$OCIO_VERSION
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  # The release archive, else a shallow clone of the tag.
  if curl -fL --retry 3 -o "$DEPS/ocio-$OCIO_VERSION.tar.gz" \
       "https://github.com/AcademySoftwareFoundation/OpenColorIO/archive/refs/tags/v$OCIO_VERSION.tar.gz"; then
    tar -xzf "$DEPS/ocio-$OCIO_VERSION.tar.gz" -C "$DEPS"
  else
    rm -rf "$SRC"
    git clone --quiet --depth 1 --branch "v$OCIO_VERSION" https://github.com/AcademySoftwareFoundation/OpenColorIO.git "$SRC"
  fi
  rm -f "$DEPS/ocio-$OCIO_VERSION.tar.gz"
fi
GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
cmake -S "$SRC" -B "$DEPS/ocio-$OCIO_VERSION-build" ${GEN[@]+"${GEN[@]}"} -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$PREFIX" -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}" \
  -DOCIO_INSTALL_EXT_PACKAGES=ALL -DOCIO_BUILD_APPS=OFF -DOCIO_BUILD_PYTHON=OFF \
  -DOCIO_BUILD_TESTS=OFF -DOCIO_BUILD_GPU_TESTS=OFF -DOCIO_BUILD_DOCS=OFF -DOCIO_BUILD_OPENFX=OFF >/dev/null
cmake --build "$DEPS/ocio-$OCIO_VERSION-build" --parallel "$JOBS"
cmake --install "$DEPS/ocio-$OCIO_VERSION-build" >/dev/null
echo "OpenColorIO $OCIO_VERSION installed at $PREFIX"
