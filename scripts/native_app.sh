#!/usr/bin/env bash
# The RUDRA app on macOS (Apple silicon) or Linux: build it, run its Qt tests,
# open it. The Windows counterpart is NATIVE_PHASE3_EXIT.ps1.
#
#   scripts/native_app.sh                  # build, test, open the app
#   NO_LAUNCH=1 scripts/native_app.sh      # build and test only
#   SKIP_TESTS=1 scripts/native_app.sh     # build and open
#   SKIP_BUILD=1 scripts/native_app.sh     # open what was built last
#
# Everything it needs goes into tmp/native_deps, nothing system-wide:
#   Qt 6.8 with Shader Tools (aqtinstall, from a venv), as native_gate_b.sh;
#   ONNX Runtime, the release archive, as native_gate_a.sh (Core ML on a Mac);
#   OpenCV core + imgproc + imgcodecs, built from source with its own PNG,
#   JPEG, TIFF, WebP and OpenEXR (a few minutes, once): the still decoder the
#   Python uses, without Homebrew's opencv and everything it pulls in;
#   OpenColorIO 2.5.2 (scripts/build_ocio.sh, once): the learned ACES 2.0
#   engine in rudra-native (aces-v2, aces-v2-serve) on ONNX Runtime.
# The model packages are found in dist/models (native_gate_a.sh exports one).
# Movies need ffmpeg and ffprobe on the PATH (brew install ffmpeg).
# Needs CMake 3.24+, Ninja or Make, a C++20 compiler (Xcode CLT on macOS),
# python3 and curl.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
QT_VERSION=${QT_VERSION:-6.8.3}
ORT_VERSION=${ORT_VERSION:-1.24.4}   # 1.2.6: as Windows and CI; pinned in tools/fetch_onnxruntime.sh
CV_VERSION=${CV_VERSION:-4.10.0}
DEPS=tmp/native_deps
BUILD=build/native_app
JOBS=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)
mkdir -p "$DEPS"

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }
# --retry-all-errors needs curl 7.71; RHEL / Alma 8 have 7.61, which stops on the unknown option
# with exit 2 (1.2.6: the Linux release job in almalinux:8).
CURL_RETRY=(--retry 5)
case "$(curl --help all 2>/dev/null)" in *--retry-all-errors*) CURL_RETRY+=(--retry-all-errors) ;; esac

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) OS=mac; QT_HOST=mac; QT_ARCH=clang_64; QT_DIR=macos; ORT_ARCHIVE=onnxruntime-osx-arm64-$ORT_VERSION ;;
  Linux-x86_64) OS=linux; QT_HOST=linux; QT_ARCH=linux_gcc_64; QT_DIR=gcc_64; ORT_ARCHIVE=onnxruntime-linux-x64-$ORT_VERSION ;;
  *) fail "Apple silicon or x86-64 Linux; on Windows use scripts/NATIVE_PHASE3_EXIT.ps1" ;;
esac
QT_ROOT=$PWD/$DEPS/Qt/$QT_VERSION/$QT_DIR
ORT_ROOT=$PWD/$DEPS/$ORT_ARCHIVE
CV_ROOT=$PWD/$DEPS/opencv-$CV_VERSION-install
OCIO_ROOT=$PWD/$DEPS/ocio-${OCIO_VERSION:-2.5.2}

if [ -z "${SKIP_BUILD:-}" ]; then
  # -------------------------------------------------------------------------
  if [ ! -f "$QT_ROOT/lib/cmake/Qt6ShaderTools/Qt6ShaderToolsConfig.cmake" ]; then
    say "Qt $QT_VERSION ($QT_ARCH, qtshadertools) into $DEPS/Qt"
    VENV=$DEPS/venv_tools
    [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade aqtinstall
    "$VENV/bin/python" -m aqt install-qt "$QT_HOST" desktop "$QT_VERSION" "$QT_ARCH" -m qtshadertools -O "$DEPS/Qt"
  fi
  # Qt 6.8 links -framework AGL, which the macOS 15+ SDKs no longer ship.
  if [ "$OS" = mac ]; then
    sed -i '' '/target_link_libraries.*__opengl_agl_fw_path/d' "$QT_ROOT/lib/cmake/Qt6/FindWrapOpenGL.cmake"
  fi

  # -------------------------------------------------------------------------
  if [ ! -f "$ORT_ROOT/include/onnxruntime_cxx_api.h" ]; then
    say "ONNX Runtime $ORT_VERSION"
    # 1.2.6: from NuGet, checked by its SHA-256; github.com/microsoft/onnxruntime is gone (17-18 Nov 2026).
    case "$OS" in mac) ORT_PLATFORM=osx-arm64 ;; *) ORT_PLATFORM=linux-x64 ;; esac
    bash tools/fetch_onnxruntime.sh "$ORT_VERSION" "$ORT_PLATFORM" "$ORT_ROOT" || fail "ONNX Runtime"
  fi

  # -------------------------------------------------------------------------
  if [ ! -f "$CV_ROOT/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
    say "OpenCV $CV_VERSION (core, imgproc, imgcodecs) from source"
    CV_SRC=$DEPS/opencv-$CV_VERSION
    if [ ! -f "$CV_SRC/CMakeLists.txt" ]; then
      # The release archive, else a shallow clone of the tag (some networks
      # refuse GitHub's archive host but not git).
      if curl -fL --retry 3 -o "$DEPS/opencv-$CV_VERSION.tar.gz" \
           "https://github.com/opencv/opencv/archive/refs/tags/$CV_VERSION.tar.gz"; then
        tar -xzf "$DEPS/opencv-$CV_VERSION.tar.gz" -C "$DEPS"
      else
        rm -rf "$CV_SRC"
        git clone --quiet --depth 1 --branch "$CV_VERSION" https://github.com/opencv/opencv.git "$CV_SRC"
      fi
      rm -f "$DEPS/opencv-$CV_VERSION.tar.gz"
    fi
    GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
    cmake -S "$CV_SRC" -B "$DEPS/opencv-$CV_VERSION-build" ${GEN[@]+"${GEN[@]}"} \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CV_ROOT" -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON -DBUILD_TIFF=ON -DBUILD_WEBP=ON \
      -DWITH_OPENEXR=ON -DBUILD_OPENEXR=ON \
      -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
      -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF \
      -DWITH_PROTOBUF=OFF -DWITH_ADE=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_1394=OFF \
      -DWITH_AVFOUNDATION=OFF -DWITH_V4L=OFF -DWITH_IMGCODEC_HDR=ON \
      -DWITH_VTK=OFF -DBUILD_opencv_python2=OFF -DPYTHON3_EXECUTABLE= -Wno-dev \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
      -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_DOCS=OFF >/dev/null
    cmake --build "$DEPS/opencv-$CV_VERSION-build" --parallel "$JOBS"
    cmake --install "$DEPS/opencv-$CV_VERSION-build" >/dev/null
  fi

  # -------------------------------------------------------------------------
  say "OpenColorIO (the learned ACES 2.0 engine)"
  scripts/build_ocio.sh || fail "OpenColorIO"

  # -------------------------------------------------------------------------
  say "Configure and build RUDRA, rudra-native and the app tests (Release)"
  GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
  TESTS=ON; [ -n "${SKIP_TESTS:-}" ] && TESTS=OFF
  # Qt6_DIR pins the fetched Qt over any Homebrew or distro Qt 6; --fresh drops
  # whatever an earlier configure found.
  cmake --fresh -S native -B "$BUILD" ${GEN[@]+"${GEN[@]}"} -DCMAKE_BUILD_TYPE=Release \
    -DRUDRA_BUILD_APP=ON -DRUDRA_BUILD_RENDER=ON -DRUDRA_BUILD_CLI=ON -DRUDRA_BUILD_TESTS=$TESTS \
    -DRUDRA_WITH_ONNXRUNTIME=ON "-DONNXRUNTIME_ROOT=$ORT_ROOT" -DRUDRA_WITH_LIBTORCH=OFF \
    -DRUDRA_WITH_OPENCV=ON "-DOpenCV_DIR=$CV_ROOT/lib/cmake/opencv4" \
    -DRUDRA_WITH_OCIO=ON "-DOpenColorIO_DIR=$OCIO_ROOT/lib/cmake/OpenColorIO" -DRUDRA_BUILD_OFX=ON -DRUDRA_OPENEXR_FETCH=ON \
    "-DCMAKE_PREFIX_PATH=$QT_ROOT" "-DQt6_DIR=$QT_ROOT/lib/cmake/Qt6" \
    "-DCMAKE_BUILD_RPATH=$ORT_ROOT/lib;$OCIO_ROOT/lib" > "$DEPS/rudra_configure.log" 2>&1 ||
    { tail -n 40 "$DEPS/rudra_configure.log" >&2; fail "cmake configure (the whole output: $DEPS/rudra_configure.log)"; }
  targets=(RUDRA rudra-native rudra_ofx_plugin)   # 26 Nov 2026: the OFX plug-in, installed from Help
  [ "$TESTS" = ON ] && targets+=(rudra_app_tests)
  cmake --build "$BUILD" --target "${targets[@]}" --parallel "$JOBS" || fail "build"
  aces_usage=$("$BUILD/cli/rudra-native" aces-v2 --help 2>&1 || true)   # it exits non-zero after the usage
  grep -q "usage: rudra-native aces-v2 <package>" <<<"$aces_usage" ||
    fail "rudra-native has no learned ACES 2.0 engine (aces-v2)"

  # -------------------------------------------------------------------------
  if [ "$TESTS" = ON ]; then
    say "The app's Qt tests (offscreen)"
    TEST_EXE=$(find "$BUILD/app" -name rudra_app_tests -type f -perm -u+x | head -1)
    [ -n "$TEST_EXE" ] || fail "rudra_app_tests not built"
    TEST_LOG=$DEPS/app_tests.log
    # APP_TEST_RUNS above 1: the whole run again until one fails, for a crash that comes and goes (release.yml's
    # app_test_runs, by hand; 26 Nov 2026).
    runs=${APP_TEST_RUNS:-1}
    case "$runs" in '' | *[!0-9]*) runs=1 ;; esac
    code=0
    for ((run = 1; run <= runs && code == 0; run++)); do
      [ "$runs" -gt 1 ] && say "The app's Qt tests, run $run of $runs"
      QT_QPA_PLATFORM=offscreen "$TEST_EXE" 2>&1 | tee "$TEST_LOG" || code=${PIPESTATUS[0]}
    done
    if [ "$code" -ne 0 ]; then
      # A crash (a signal: exit above 128) leaves no failing assertion to read. On a Mac the test it crashed in is
      # run again under lldb, alone and then (if it passes alone) with the whole run, and the backtrace goes
      # between the markers tools/ci_run.sh publishes (26 Nov 2026: v1.3.1's macOS job crashed twice in
      # AppDeliver.AFirstRunWithoutAModelCanMaster, Linux and Windows passed).
      if [ "$code" -gt 128 ] && [ "$OS" = mac ] && command -v lldb >/dev/null; then
        crashed_in=$(grep -E '^\[ RUN      \] ' "$TEST_LOG" | tail -n 1 | awk '{print $4}')
        sudo -n DevToolsSecurity -enable >/dev/null 2>&1 || true
        backtrace() {   # the crashed thread's frames, short: lldb's own lines, without the addresses
          QT_QPA_PLATFORM=offscreen lldb --batch -o run -k "thread backtrace -c 24" -k "quit 1" -- "$TEST_EXE" "$@" 2>&1 |
            grep -E 'stop reason|frame #|exited with status|^error' | sed -E '/frame #/ s/0x[0-9a-f]{8,16} //' | cut -c1-150 || true
        }
        say "The app tests crashed (signal $((code - 128))) in ${crashed_in:-an unknown test}: that test alone, under lldb"
        alone=$(backtrace "--gtest_filter=${crashed_in:-*}")
        echo "=== crash backtrace ==="
        printf '%s\n' "$alone"
        if grep -q 'exited with status' <<<"$alone"; then
          echo "(it passed alone: the whole run again, under lldb)"
          backtrace | grep -v 'exited with status' | head -n 26
        fi
        echo "=== end crash backtrace ==="
      fi
      fail "app tests"
    fi
  fi
fi

# ---------------------------------------------------------------------------
if [ "$OS" = mac ]; then APP=$BUILD/app/RUDRA.app/Contents/MacOS/RUDRA; else APP=$BUILD/app/RUDRA; fi
[ -x "$APP" ] || fail "no app at $APP (run without SKIP_BUILD)"
for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || echo "note: no $tool on the PATH: stills work, movies will not (brew install ffmpeg)"
done
ls dist/models/*/manifest.json >/dev/null 2>&1 ||
  echo "note: no model package in dist/models: run scripts/native_gate_a.sh (it exports one), or copy one from the PC"
if [ -n "${NO_LAUNCH:-}" ]; then echo; echo "Built: $APP"; exit 0; fi

say "Open RUDRA"
export RUDRA_PACKAGE_ROOTS="$PWD/dist/models${RUDRA_PACKAGE_ROOTS:+:$RUDRA_PACKAGE_ROOTS}"
exec "$APP" "$@"
