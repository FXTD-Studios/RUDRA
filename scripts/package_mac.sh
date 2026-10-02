#!/usr/bin/env bash
# The RUDRA beta for macOS (Apple silicon): a self-contained RUDRA.app in a DMG.
#
#   scripts/package_mac.sh                 # build, test, package -> dist/beta/RUDRA-<version>-macos-arm64.dmg
#   SKIP_TESTS=1 scripts/package_mac.sh    # without the app's Qt tests
#   SKIP_BUILD=1 scripts/package_mac.sh    # package what native_app.sh built last
#
# The app is built by scripts/native_app.sh (Qt 6.8, ONNX Runtime with Core ML,
# OpenCV from source, all in tmp/native_deps). Into the bundle go:
#   Contents/Frameworks  Qt (macdeployqt) and ONNX Runtime
#   Contents/MacOS       RUDRA and rudra-native (the CLI: diff, video, deliver, batch)
#   Contents/Resources   the icon, the model package(s) from dist/models, LICENSE, NOTICE
# Signing (docs/MACOS_SIGNING.md):
#   MAC_SIGN_IDENTITY="Developer ID Application: <name> (<team>)"   a certificate in the keychain
#   and one way to notarize:
#     MAC_NOTARY_KEY=<AuthKey_XXXX.p8> MAC_NOTARY_KEY_ID=<key id> MAC_NOTARY_ISSUER=<issuer uuid>
#     MAC_NOTARY_PROFILE=<name>   (xcrun notarytool store-credentials <name> ...)
# With both, the app is signed with the hardened runtime, notarized and
# stapled, and so is the DMG: it opens with no warning. Without an identity it
# is signed ad hoc as before and macOS warns on first open (right-click > Open,
# or `xattr -dr com.apple.quarantine RUDRA.app`). An identity without
# notarization fails, since that build would warn too (ALLOW_UNNOTARIZED=1 to
# keep it for a local test). ffmpeg is not bundled
# (movies need an ffmpeg with libx265, prores_ks and zscale on the PATH, e.g.
# `brew install ffmpeg@6` with its bin directory first on PATH); stills need nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || { echo "package_mac.sh runs on Apple silicon" >&2; exit 2; }
PY=${PYTHON:-python3}
QT_VERSION=${QT_VERSION:-6.8.3}
ORT_VERSION=${ORT_VERSION:-1.22.0}
BUILD=build/native_app
QT_ROOT=$PWD/tmp/native_deps/Qt/$QT_VERSION/macos
ORT_ROOT=$PWD/tmp/native_deps/onnxruntime-osx-arm64-$ORT_VERSION

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

ls dist/models/*/manifest.json >/dev/null 2>&1 ||
  fail "no model package in dist/models (scripts/native_gate_a.sh exports one)"

if [ -z "${SKIP_BUILD:-}" ]; then
  NO_LAUNCH=1 PYTHON="$PY" SKIP_TESTS="${SKIP_TESTS:-}" scripts/native_app.sh
fi
APP_SRC=$BUILD/app/RUDRA.app
CLI_SRC=$BUILD/cli/rudra-native
[ -d "$APP_SRC" ] && [ -x "$CLI_SRC" ] || fail "no build at $BUILD (run without SKIP_BUILD)"

VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP_SRC/Contents/Info.plist")
NAME=RUDRA-$VERSION-macos-arm64
OUT=dist/beta/$NAME
say "Package $NAME"
rm -rf "$OUT" "dist/beta/$NAME.dmg"
mkdir -p "$OUT"
APP=$OUT/RUDRA.app
cp -R "$APP_SRC" "$APP"
cp "$CLI_SRC" "$APP/Contents/MacOS/rudra-native"

# ONNX Runtime beside Qt, found through @rpath from both executables.
mkdir -p "$APP/Contents/Frameworks"
cp -a "$ORT_ROOT"/lib/libonnxruntime*.dylib "$APP/Contents/Frameworks/"
for exe in "$APP/Contents/MacOS/RUDRA" "$APP/Contents/MacOS/rudra-native"; do
  install_name_tool -add_rpath "@executable_path/../Frameworks" "$exe" 2>/dev/null || true
  # The build's absolute rpaths (tmp/native_deps) mean nothing on another Mac.
  for rp in $(otool -l "$exe" | awk '/LC_RPATH/ { getline; getline; print $2 }' | grep "^/" || true); do
    install_name_tool -delete_rpath "$rp" "$exe" 2>/dev/null || true
  done
done

say "Qt into the bundle (macdeployqt)"
"$QT_ROOT/bin/macdeployqt" "$APP" -executable="$APP/Contents/MacOS/rudra-native" -verbose=1

say "Models, licences"
mkdir -p "$APP/Contents/Resources/models"
cp -R dist/models/. "$APP/Contents/Resources/models/"
find "$APP/Contents/Resources/models" -name "*.safetensors" -delete -o -name "*.config.json" -delete
cp LICENSE NOTICE "$APP/Contents/Resources/"
[ -f checkpoints/LICENSE ] && cp checkpoints/LICENSE "$APP/Contents/Resources/LICENSE-weights"

IDENTITY=${MAC_SIGN_IDENTITY:-}
notary_args=()
if [ -n "${MAC_NOTARY_PROFILE:-}" ]; then
  notary_args=(--keychain-profile "$MAC_NOTARY_PROFILE")
elif [ -n "${MAC_NOTARY_KEY:-}" ]; then
  [ -f "$MAC_NOTARY_KEY" ] && [ -n "${MAC_NOTARY_KEY_ID:-}" ] && [ -n "${MAC_NOTARY_ISSUER:-}" ] ||
    fail "MAC_NOTARY_KEY needs the .p8 file, MAC_NOTARY_KEY_ID and MAC_NOTARY_ISSUER"
  notary_args=(--key "$MAC_NOTARY_KEY" --key-id "$MAC_NOTARY_KEY_ID" --issuer "$MAC_NOTARY_ISSUER")
fi
if [ -n "$IDENTITY" ] && [ ${#notary_args[@]} -eq 0 ] && [ -z "${ALLOW_UNNOTARIZED:-}" ]; then
  fail "MAC_SIGN_IDENTITY is set but nothing to notarize with: a signed, unnotarized app still warns"
fi

# Notarize one file and wait; on rejection print Apple's log, which names the binary.
notarize() {
  say "Notarize $(basename "$1")"
  local out id
  out=$(xcrun notarytool submit "$1" "${notary_args[@]}" --wait --output-format json) || true
  echo "$out"
  id=$(printf '%s' "$out" | sed -n 's/.*"id"[^"]*"\([^"]*\)".*/\1/p' | head -1)
  if ! printf '%s' "$out" | grep -q '"status"[^"]*"Accepted"'; then
    [ -n "$id" ] && xcrun notarytool log "$id" "${notary_args[@]}" || true
    fail "notarization of $1 was not accepted"
  fi
}

if [ -n "$IDENTITY" ]; then
  say "Sign (Developer ID, hardened runtime)"
  sign() { codesign --force --timestamp --options runtime --sign "$IDENTITY" "$@"; }
  # Inside out, never --deep: every library and plug-in, then the frameworks,
  # then the second executable, then the app with its main executable.
  while IFS= read -r -d '' f; do sign "$f"; done < <(find "$APP/Contents" -type f \( -name "*.dylib" -o -name "*.so" \) -print0)
  for fw in "$APP"/Contents/Frameworks/*.framework; do [ -d "$fw" ] && sign "$fw"; done
  sign "$APP/Contents/MacOS/rudra-native"
  sign "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP"
  if [ ${#notary_args[@]} -gt 0 ]; then
    ditto -c -k --keepParent "$APP" "$OUT/RUDRA.zip"
    notarize "$OUT/RUDRA.zip"
    rm -f "$OUT/RUDRA.zip"
    xcrun stapler staple "$APP"
    verdict=$(spctl --assess --type execute --verbose=2 "$APP" 2>&1 || true)
    echo "$verdict"
    case "$verdict" in *"Notarized Developer ID"*) ;; *) fail "Gatekeeper does not accept the notarized app" ;; esac
  fi
else
  say "Sign (ad hoc: no MAC_SIGN_IDENTITY, macOS will warn on first open)"
  codesign --force --deep --sign - "$APP"
  codesign --verify --deep --strict "$APP"
fi

say "Check the bundle runs from where it is"
# Nothing may still point into the build tree or Homebrew.
leaks=$(for f in "$APP/Contents/MacOS/"* "$APP/Contents/Frameworks/"*.dylib; do otool -L "$f" | tail -n +2; done |
        grep -E "$PWD|/opt/homebrew|/usr/local" || true)
[ -z "$leaks" ] || { printf '%s\n' "$leaks"; fail "the bundle links outside itself"; }
"$APP/Contents/MacOS/rudra-native" info "$APP/Contents/Resources/models/$(ls "$APP/Contents/Resources/models" | grep -v models.json | head -1)" >/dev/null ||
  fail "rudra-native in the bundle cannot read its model"
# The Cocoa platform: macdeployqt ships that one, not offscreen.
"$APP/Contents/MacOS/RUDRA" --theme-check "$OUT/theme-check.json" >/dev/null ||
  fail "RUDRA in the bundle does not start"
rm -f "$OUT/theme-check.json"

say "Disk image"
cp docs/BETA.md "$OUT/Read me first.md" 2>/dev/null || true
ln -s /Applications "$OUT/Applications"
hdiutil create -volname "RUDRA $VERSION" -srcfolder "$OUT" -ov -format UDZO "dist/beta/$NAME.dmg" >/dev/null
if [ -n "$IDENTITY" ]; then
  codesign --force --timestamp --sign "$IDENTITY" "dist/beta/$NAME.dmg"
  if [ ${#notary_args[@]} -gt 0 ]; then
    notarize "dist/beta/$NAME.dmg"
    xcrun stapler staple "dist/beta/$NAME.dmg"
    xcrun stapler validate "dist/beta/$NAME.dmg"
    spctl --assess --type open --context context:primary-signature --verbose=2 "dist/beta/$NAME.dmg"
  fi
fi
shasum -a 256 "dist/beta/$NAME.dmg" | tee "dist/beta/$NAME.dmg.sha256"
echo
if [ -n "$IDENTITY" ] && [ ${#notary_args[@]} -gt 0 ]; then signed="Developer ID, notarized"
elif [ -n "$IDENTITY" ]; then signed="Developer ID, NOT notarized (warns on first open)"
else signed="ad hoc (warns on first open)"; fi
echo "Built: dist/beta/$NAME.dmg  [$signed]"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "macOS DMG: $NAME.dmg, signed $signed" >> "$GITHUB_STEP_SUMMARY"; fi
