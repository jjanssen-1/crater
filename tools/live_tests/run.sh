#!/usr/bin/env bash
# Runs the crater live-device tests.
#
# Usage: tools/live_tests/run.sh [results-file]
#
# Requirements: crater installed (pip install .), g++, Vulkan headers
# and a loader (vulkan SDK or libvulkan), a vk.xml reachable (SDK).

set -euo pipefail
cd "$(dirname "$0")/../.."

OUT="${1:-live_results.md}"
GEN="$(mktemp -d)/gen"
mkdir -p "$GEN"

echo "== generating headers"
crater generate tools/live_tests/specs/common.yaml        -o "$GEN/common.hpp"
crater generate tools/live_tests/specs/device_objects.yaml -o "$GEN/device_objects.hpp"
crater generate tools/live_tests/specs/render_objects.yaml -o "$GEN/render_objects.hpp"

echo "== building harness"
g++ -std=c++17 tools/live_tests/harness.cpp -o "$GEN/live_tests" \
    -I "$GEN" -I "${VK_HEADERS_DIR:-/usr/include}" -lvulkan

echo "== running (requires a Vulkan device)"
"$GEN/live_tests" "$OUT"
