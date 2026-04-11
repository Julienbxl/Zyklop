#!/usr/bin/env bash
# =============================================================================
# scripts/download_circom_libs.sh
#
# Downloads the circom library dependencies needed to compile the benchmark
# circuits. The circuit sources themselves (circuit.circom) are included in
# the Zyklop repository under test/*/. This script only fetches the external
# circom libraries they depend on.
#
# Required by:
#   - ecdsa_4x     : circom-ecdsa/circuits (secp256k1, bigint arithmetic)
#   - keccak_40    : keccak256-circom/circuits
#   - all circuits : circomlib via npm (run 'npm install' separately)
#
# Usage:
#   bash scripts/download_circom_libs.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY_DIR="$ROOT/third_party"

mkdir -p "$THIRD_PARTY_DIR"

clone_or_update() {
    local repo_url="$1"
    local dest_dir="$2"
    if [ -d "$dest_dir/.git" ]; then
        echo "[libs] Updating $(basename "$dest_dir")..."
        git -C "$dest_dir" fetch --depth=1 origin
        git -C "$dest_dir" reset --hard FETCH_HEAD
        return
    fi
    echo "[libs] Cloning $(basename "$dest_dir")..."
    git clone --depth=1 "$repo_url" "$dest_dir"
}

clone_or_update "https://github.com/0xPARC/circom-ecdsa.git" \
    "$THIRD_PARTY_DIR/circom-ecdsa"

clone_or_update "https://github.com/vocdoni/keccak256-circom.git" \
    "$THIRD_PARTY_DIR/keccak256-circom"

clone_or_update "https://github.com/zkmopro/benchmark-app.git" \
    "$THIRD_PARTY_DIR/benchmark-app"

echo "[libs] Done. Now run: npm install"
