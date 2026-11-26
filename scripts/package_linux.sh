#!/usr/bin/env bash
# The RUDRA package for Linux x86-64 (1.2.6, readiness report checklist item 10): one folder in a
# tar.gz that runs on glibc 2.28 and later, so on Rocky / Alma / RHEL 8 and 9, the floor of the
# VFX Reference Platform CY2026 (Qt 6.8, OCIO 2.5, C++20). Built on that floor (release.yml runs it
# in an almalinux:8 container with gcc-toolset-13), it runs on everything newer.
#
#   scripts/package_linux.sh                 # build, test, package -> dist/beta/RUDRA-<version>-linux-x86_64.tar.gz
#   SKIP_TESTS=1 scripts/package_linux.sh    # without the app's Qt tests
#   SKIP_BUILD=1 scripts/package_linux.sh    # package what native_app.sh built last
#   PUBLIC_RELEASE=1 scripts/package_linux.sh   # what release.yml builds: no model weights at all
#                                               # (every model is an in-app download with its licence)
# Knobs: BUILD_DIR (build/native_app), QT_ROOT and QT_PLUGINS (the fetched Qt 6.8.3), ORT_ROOT,
# OCIO_ROOT, GLIBC_MAX (2.28). For a check of the packaging on a newer distro only:
# ALLOW_NEWER_GLIBC=1 (the package then needs that distro's glibc, and says so), ALLOW_NO_ACES=1
# (a build without ONNX Runtime / OpenColorIO). Never for a release.
#
# The folder, like the Windows one (the app looks beside itself for python/, models/, ffmpeg/):
#   RUDRA, rudra-native        the app and the CLI; RUNPATH $ORIGIN/lib
#   lib/                       Qt, ONNX Runtime, OpenColorIO and every library they need that a
#                              Rocky 8 install may not have (xcb-util, xkbcommon-x11 ...); the
#                              system's own (glibc, libstdc++, GL, X11, fontconfig, freetype, dbus,
#                              wayland, glib) are left to the system, as every VFX app does
#   plugins/                   Qt's: xcb, wayland, offscreen (render nodes), image formats, themes
#   qt.conf                    plugins/ and lib/ beside the app, whatever the folder is called
#   python/                    the AI fill's Python side (rudra/, config/, the bootstrap), no weights
#   models/                    the model packages (internal builds only; none in a public release)
#   rudra.desktop, rudra.png, install-desktop-entry.sh   an entry in the desktop's menu, per user
#   LICENSE, NOTICE, Read me first.md, Privacy.md
# ffmpeg: LINUX_FFMPEG_DIR=<a folder with a static x86-64 ffmpeg and ffprobe (libx265, prores_ks,
# zscale) and their LICENSE> bundles them in ffmpeg/; otherwise movies need ffmpeg on the PATH.
# Needs patchelf (pip install patchelf), objdump (binutils) and ldd.
set -euo pipefail
cd "$(dirname "$0")/.."
say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }
[ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] || fail "package_linux.sh runs on x86-64 Linux"
PY=${PYTHON:-python3}
QT_VERSION=${QT_VERSION:-6.8.3}
ORT_VERSION=${ORT_VERSION:-1.24.4}
DEPS=$PWD/tmp/native_deps
BUILD=${BUILD_DIR:-build/native_app}
QT_ROOT=${QT_ROOT:-$DEPS/Qt/$QT_VERSION/gcc_64}
QT_PLUGINS=${QT_PLUGINS:-$QT_ROOT/plugins}
ORT_ROOT=${ORT_ROOT:-$DEPS/onnxruntime-linux-x64-$ORT_VERSION}
OCIO_ROOT=${OCIO_ROOT:-$DEPS/ocio-${OCIO_VERSION:-2.5.2}}
GLIBC_MAX=${GLIBC_MAX:-2.28}
GLIBCXX_MAX=${GLIBCXX_MAX:-3.4.25}   # GCC 8's libstdc++, Rocky 8's; gcc-toolset links newer parts in statically

command -v patchelf >/dev/null || "$PY" -m pip install --quiet --user patchelf
command -v patchelf >/dev/null || PATH="$("$PY" -m site --user-base)/bin:$PATH"
command -v patchelf >/dev/null || fail "no patchelf (pip install patchelf)"
command -v objdump >/dev/null || fail "no objdump (binutils)"

BUNDLE_MODELS=1
[ -n "${PUBLIC_RELEASE:-}" ] && BUNDLE_MODELS=""
if [ -n "$BUNDLE_MODELS" ]; then
  ls dist/models/*/manifest.json >/dev/null 2>&1 ||
    fail "no model package in dist/models (scripts/native_gate_a.sh exports one), or PUBLIC_RELEASE=1"
  if [ -z "${ALLOW_NONCOMMERCIAL_WEIGHTS:-}" ] &&
     find dist/models | grep -Eq '(^|/)(sdr2hdr_[^/]*|[^/]*shadow_v1[^/]*|[^/]*shadow_s[23][^/]*|[^/]*temporal_v1[^/]*)$'; then
    fail "dist/models holds non-commercial Beta 2 weights; the bundle is Apache 2.0 (ALLOW_NONCOMMERCIAL_WEIGHTS=1 for an internal build only)"
  fi
fi

if [ -z "${SKIP_BUILD:-}" ]; then
  NO_LAUNCH=1 PYTHON="$PY" SKIP_TESTS="${SKIP_TESTS:-}" QT_VERSION="$QT_VERSION" ORT_VERSION="$ORT_VERSION" scripts/native_app.sh
fi
APP_SRC=$BUILD/app/RUDRA
CLI_SRC=$BUILD/cli/rudra-native
[ -x "$APP_SRC" ] && [ -x "$CLI_SRC" ] || fail "no build at $BUILD (run without SKIP_BUILD)"

cache() { sed -n "s/^$1:[A-Z]*=//p" "$BUILD/CMakeCache.txt" | head -1; }
VERSION=$(cache CMAKE_PROJECT_VERSION)
[ -n "$(cache RUDRA_RELEASE)" ] && VERSION="$VERSION-$(cache RUDRA_RELEASE)"
[ -n "$VERSION" ] || fail "no version in $BUILD/CMakeCache.txt"
NAME=RUDRA-$VERSION-linux-x86_64
OUT=dist/beta/$NAME
say "Package $NAME"
rm -rf "$OUT" "dist/beta/$NAME.tar.gz" "dist/beta/$NAME.tar.gz.sha256"
mkdir -p "$OUT/lib" "$OUT/plugins"
cp "$APP_SRC" "$OUT/RUDRA"
cp "$CLI_SRC" "$OUT/rudra-native"
# 26 Nov 2026: the OFX plug-in for Resolve and Nuke beside the app: Help > Install the OFX plug-in copies it into
# /usr/OFX/Plugins (pkexec; app/ofx_install.hpp). ofx/nuke: Nuke's toolbar icon.
[ -f "$BUILD/RUDRA.ofx.bundle/Contents/Linux-x86-64/RUDRA.ofx" ] || fail "no OFX plug-in at $BUILD/RUDRA.ofx.bundle"
mkdir -p "$OUT/ofx/nuke"
cp -R "$BUILD/RUDRA.ofx.bundle" "$OUT/ofx/"
cp native/ofx/nuke/* "$OUT/ofx/nuke/"

say "Qt plug-ins"
[ -d "$QT_PLUGINS/platforms" ] || fail "no Qt plug-ins at $QT_PLUGINS"
for d in platforms xcbglintegrations platformthemes platforminputcontexts imageformats iconengines \
         wayland-decoration-client wayland-graphics-integration-client wayland-shell-integration tls; do
  [ -d "$QT_PLUGINS/$d" ] || continue
  mkdir -p "$OUT/plugins/$d"
  for f in "$QT_PLUGINS/$d"/*.so; do
    [ -e "$f" ] || continue
    case "$(basename "$f")" in
      # Desktop themes that pull a whole toolkit (GTK) in; Qt's own look does not need them.
      libqgtk3.so|libqxdgdesktopportal.so) continue ;;
      # Image formats the app does not read through Qt (OpenCV and its own EXR reader do).
      libqtiff.so|libqwebp.so|libqicns.so|libqtga.so|libqwbmp.so|libqpdf.so|libqmacheif.so|libqmacjp2.so) continue ;;
    esac
    cp "$f" "$OUT/plugins/$d/"
  done
done
[ -f "$OUT/plugins/platforms/libqxcb.so" ] || fail "no xcb platform plug-in (X11)"
[ -f "$OUT/plugins/platforms/libqoffscreen.so" ] || fail "no offscreen platform plug-in (render nodes)"
printf '[Paths]\nPrefix = .\nPlugins = plugins\nLibraries = lib\n' > "$OUT/qt.conf"

say "Libraries"
# The system's own libraries stay the system's: a bundled glibc, libstdc++, GL driver, X11,
# fontconfig or dbus breaks on a machine with another (AppImage's exclude list, trimmed).
SYSTEM_RE='^(linux-vdso|ld-linux-x86-64|libc|libm|libdl|libpthread|librt|libresolv|libutil|libnsl|libanl|libmvec|libBrokenLocale|libgcc_s|libstdc\+\+|libGL|libGLX|libEGL|libOpenGL|libGLdispatch|libGLESv2|libdrm|libgbm|libvulkan|libX11|libX11-xcb|libXext|libXau|libXdmcp|libXrender|libXi|libXfixes|libxcb|libxcb-glx|libxcb-dri2|libxcb-dri3|libxcb-present|libxcb-sync|libxshmfence|libfontconfig|libfreetype|libexpat|libz|libdbus-1|libsystemd|libudev|libselinux|libpcre|libpcre2-8|libuuid|libglib-2\.0|libgobject-2\.0|libgthread-2\.0|libgmodule-2\.0|libgio-2\.0|libwayland-client|libwayland-cursor|libwayland-egl|libwayland-server|libxkbcommon|libasound|libpulse|libcap|libgcrypt|libgpg-error|liblzma|libzstd|liblz4|libbz2|libffi|libmount|libblkid|libresolv|libkeyutils|libcom_err|libgssapi_krb5|libkrb5|libk5crypto|libkrb5support|libcrypt)\.so'
export LD_LIBRARY_PATH="$QT_ROOT/lib:$ORT_ROOT/lib:$OCIO_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elfs=("$OUT/RUDRA" "$OUT/rudra-native")
while IFS= read -r -d '' f; do elfs+=("$f"); done < <(find "$OUT/plugins" -name '*.so' -print0)
missing=""
for e in "${elfs[@]}"; do
  while read -r soname arrow path _; do
    [ "$arrow" = "=>" ] || continue
    if [ "$path" = "not" ]; then missing+="$(basename "$e"): $soname"$'\n'; continue; fi
    [[ "$soname" =~ $SYSTEM_RE ]] && continue
    [ -f "$OUT/lib/$soname" ] || cp -L "$path" "$OUT/lib/$soname"
  done < <(ldd "$e")
done
[ -z "$missing" ] || { printf '%s' "$missing"; fail "libraries not found while packaging (install them, or check QT_ROOT / ORT_ROOT / OCIO_ROOT)"; }
unset LD_LIBRARY_PATH
# Qt's own libraries read their plug-ins through qt.conf; their ICU data comes with them.
patchelf --set-rpath '$ORIGIN/lib' "$OUT/RUDRA" "$OUT/rudra-native"
for f in "$OUT"/lib/*.so*; do chmod u+w "$f"; patchelf --set-rpath '$ORIGIN' "$f"; done
while IFS= read -r -d '' f; do chmod u+w "$f"; patchelf --set-rpath '$ORIGIN/../../lib' "$f"; done < <(find "$OUT/plugins" -name '*.so' -print0)
strip --strip-unneeded "$OUT/RUDRA" "$OUT/rudra-native" 2>/dev/null || true

say "Models, licences, the AI fill's Python side"
mkdir -p "$OUT/models"
if [ -n "$BUNDLE_MODELS" ]; then
  cp -R dist/models/. "$OUT/models/"
  find "$OUT/models" -name "*.safetensors" -delete -o -name "*.config.json" -delete
fi
cp LICENSE NOTICE "$OUT/"
PYDIR=$(cd "$OUT" && pwd)/python
mkdir -p "$PYDIR"
for d in rudra config; do
  (cd "$d" && find . \( -name '*.py' -o -name 'model_manifest.json' \) -not -path '*/__pycache__/*' -print0 |
     while IFS= read -r -d '' f; do mkdir -p "$PYDIR/$d/$(dirname "$f")"; cp "$f" "$PYDIR/$d/$f"; done)
done
cp scripts/genfill_bootstrap.sh scripts/genfill_bootstrap.ps1 scripts/genfill_constraints.txt "$PYDIR/"
chmod +x "$PYDIR/genfill_bootstrap.sh"
[ -f "$PYDIR/rudra/genfill/run.py" ] || fail "the package has no python/rudra/genfill"
[ -f docs/USER_GUIDE.md ] && cp docs/USER_GUIDE.md "$OUT/User guide.md"
cp docs/READ_ME_FIRST.md "$OUT/Read me first.md" 2>/dev/null || true
cp docs/PRIVACY.md "$OUT/Privacy.md" 2>/dev/null || true

say "Desktop entry"
cp native/app/icons/rudra-1024.png "$OUT/rudra.png"
cat > "$OUT/rudra.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=RUDRA
GenericName=HDR reconstruction
Comment=SDR to HDR reconstruction and ACES delivery
Exec=RUDRA %F
Icon=rudra
Terminal=false
Categories=AudioVideo;Video;Graphics;
MimeType=image/png;image/jpeg;image/tiff;image/x-exr;video/mp4;video/quicktime;
EOF
cat > "$OUT/install-desktop-entry.sh" <<'EOF'
#!/usr/bin/env bash
# Adds RUDRA to this user's application menu (no root; nothing outside ~/.local changes).
#   ./install-desktop-entry.sh            add or update the entry for RUDRA in this folder
#   ./install-desktop-entry.sh --remove   take it out again
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
apps=${XDG_DATA_HOME:-$HOME/.local/share}/applications
icons=${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/1024x1024/apps
if [ "${1:-}" = "--remove" ]; then
  rm -f "$apps/rudra.desktop" "$icons/rudra.png"
  echo "RUDRA removed from the menu"
  exit 0
fi
mkdir -p "$apps" "$icons"
cp "$here/rudra.png" "$icons/rudra.png"
sed "s|^Exec=.*|Exec=\"$here/RUDRA\" %F|" "$here/rudra.desktop" > "$apps/rudra.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$apps" >/dev/null 2>&1 || true
echo "RUDRA added to the menu ($apps/rudra.desktop)"
EOF
chmod +x "$OUT/install-desktop-entry.sh" "$OUT/RUDRA" "$OUT/rudra-native"

FFMPEG_BUNDLED=""
if [ -n "${LINUX_FFMPEG_DIR:-}" ]; then
  say "ffmpeg into the package ($LINUX_FFMPEG_DIR)"
  for t in ffmpeg ffprobe; do [ -x "$LINUX_FFMPEG_DIR/$t" ] || fail "LINUX_FFMPEG_DIR has no executable $t"; done
  ls "$LINUX_FFMPEG_DIR" | grep -qi '^licen[sc]e' || fail "LINUX_FFMPEG_DIR has no LICENSE for the ffmpeg build (GPL: it must travel with it)"
  mkdir -p "$OUT/ffmpeg" "$OUT/ffmpeg-licence"
  cp "$LINUX_FFMPEG_DIR/ffmpeg" "$LINUX_FFMPEG_DIR/ffprobe" "$OUT/ffmpeg/"
  find "$LINUX_FFMPEG_DIR" -maxdepth 1 -type f \( -iname 'licen*' -o -iname 'readme*' -o -iname 'source*' \) -exec cp {} "$OUT/ffmpeg-licence/" \;
  for t in ffmpeg ffprobe; do
    # The ELF's own dependency list, not ldd (it returns 1 for a static binary, which pipefail took as a
    # failure); captured first, so grep -q stopping early cannot fail the pipe either.
    # readelf must read it (a shell wrapper around a host ffmpeg is no ELF at all, review 19 Nov).
    deps=$(readelf -d "$OUT/ffmpeg/$t" 2>/dev/null) || fail "the bundled $t is not an ELF executable"
    if grep -q '(NEEDED)' <<<"$deps"; then fail "the bundled $t is not static"; fi
  done
  "$OUT/ffmpeg/ffmpeg" -hide_banner -h encoder=prores_ks >/dev/null || fail "the bundled ffmpeg does not run"
  FFMPEG_BUNDLED=1
else
  echo "ffmpeg not bundled (LINUX_FFMPEG_DIR unset): movies need one on the PATH"
fi

say "Check: glibc $GLIBC_MAX, libstdc++ $GLIBCXX_MAX, nothing outside the package"
ver_max() { grep -o "$1_[0-9][0-9.]*" | sed "s/^$1_//" | sort -uV | tail -1; }
all=$(find "$OUT" -type f \( -name '*.so*' -o -name RUDRA -o -name rudra-native \) -print)
need_glibc=$(for f in $all; do objdump -T "$f" 2>/dev/null; done | ver_max GLIBC)
need_cxx=$(for f in $all; do objdump -T "$f" 2>/dev/null; done | ver_max GLIBCXX)
echo "needs glibc $need_glibc, libstdc++ (GLIBCXX) ${need_cxx:-none}"
newer() { [ -n "$1" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" != "$2" ]; }
if newer "$need_glibc" "$GLIBC_MAX" || newer "$need_cxx" "$GLIBCXX_MAX"; then
  culprits=$(for f in $all; do
               v=$(objdump -T "$f" 2>/dev/null | ver_max GLIBC); newer "$v" "$GLIBC_MAX" && echo "  $(basename "$f"): GLIBC_$v"
               v=$(objdump -T "$f" 2>/dev/null | ver_max GLIBCXX); newer "$v" "$GLIBCXX_MAX" && echo "  $(basename "$f"): GLIBCXX_$v"
             done; true)
  printf '%s\n' "$culprits"
  [ -n "${ALLOW_NEWER_GLIBC:-}" ] || fail "built on a newer system than glibc $GLIBC_MAX / GLIBCXX $GLIBCXX_MAX: build in almalinux:8 (release.yml)"
  echo "WARNING: ALLOW_NEWER_GLIBC: this package needs glibc $need_glibc; it is not the release package"
  NAME=$NAME-glibc$need_glibc
fi
leaks=""
for f in $all; do
  leaks+=$(env -u LD_LIBRARY_PATH ldd "$f" 2>/dev/null | awk -v f="$(basename "$f")" '/not found/ { print f": "$1" not found" }
                                                                                       /=>/ && $3 ~ /tmp\/native_deps|\/build\// { print f": "$3 }')
done
[ -z "$leaks" ] || { printf '%s\n' "$leaks"; fail "the package links outside itself, or misses a library"; }

say "Check: it starts from where it is"
home=$(mktemp -d)
env -i PATH=/usr/bin:/bin HOME="$home" QT_QPA_PLATFORM=offscreen "$OUT/RUDRA" --theme-check "$home/theme-check.json" >/dev/null ||
  fail "RUDRA in the package does not start"
[ -s "$home/theme-check.json" ] || fail "RUDRA in the package wrote no theme check"
aces_usage=$(env -i PATH=/usr/bin:/bin HOME="$home" "$OUT/rudra-native" aces-v2 --help 2>&1 || true)   # non-zero after the usage
if ! grep -q "usage: rudra-native aces-v2 <package>" <<<"$aces_usage"; then
  [ -n "${ALLOW_NO_ACES:-}" ] || fail "the packaged rudra-native has no ACES 2.0 engine, or cannot load OpenColorIO / ONNX Runtime"
  echo "WARNING: ALLOW_NO_ACES: no learned ACES 2.0 engine in this package"
  NAME=$NAME-noaces
fi
rm -rf "$home"
if [ -n "$BUNDLE_MODELS" ]; then
  "$OUT/rudra-native" info "$OUT/models/$(ls "$OUT/models" | grep -v models.json | head -1)" >/dev/null ||
    fail "rudra-native in the package cannot read its model"
else
  "$PY" tools/check_package_weights.py "$OUT" || fail "the public package carries model weights"
fi

say "Archive"
if [ "dist/beta/$NAME" != "$OUT" ]; then rm -rf "dist/beta/$NAME"; mv "$OUT" "dist/beta/$NAME"; OUT=dist/beta/$NAME; fi
tar -C dist/beta --owner=0 --group=0 --numeric-owner -czf "dist/beta/$NAME.tar.gz" "$NAME"
(cd dist/beta && sha256sum "$NAME.tar.gz" | tee "$NAME.tar.gz.sha256")
size=$(du -sh "dist/beta/$NAME.tar.gz" | cut -f1)
echo
echo "Built: dist/beta/$NAME.tar.gz ($size; glibc $need_glibc; ffmpeg ${FFMPEG_BUNDLED:+bundled}${FFMPEG_BUNDLED:-not bundled})"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "Linux tar.gz: $NAME.tar.gz ($size, needs glibc $need_glibc)" >> "$GITHUB_STEP_SUMMARY"; fi
