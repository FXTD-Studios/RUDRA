#!/usr/bin/env bash
# Runs a command up to N times, waiting between attempts. For the network steps
# CI cannot control: GitHub release downloads (ONNX Runtime, nlohmann/json,
# vcpkg's own CMake) returned 504s on 8 Nov 2026 and failed four native jobs
# that had nothing wrong with them.
#
#   tools/ci_retry.sh 4 curl -fsSL -o ort.tgz "$url"
#   CI_RETRY_DELAY=30 tools/ci_retry.sh 3 vcpkg install gtest
set -uo pipefail
attempts=$1
shift
delay=${CI_RETRY_DELAY:-15}
for ((i = 1; i <= attempts; i++)); do
  "$@" && exit 0
  code=$?
  if [ "$i" -lt "$attempts" ]; then
    echo "::warning::attempt $i/$attempts failed (exit $code), retrying in ${delay}s: $*"
    sleep "$delay"
    delay=$((delay * 2))
  fi
done
echo "::error::failed after $attempts attempts (exit $code): $*"
exit "$code"
