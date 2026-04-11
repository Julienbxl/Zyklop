#!/usr/bin/env bash
# =============================================================================
# scripts/setup_benchmarks.sh
#
# Compiles circuits and generates the zkey + witness for each benchmark.
# The circuit.circom and input.json files are already in the repository.
#
# Usage:
#   bash scripts/setup_benchmarks.sh poseidon5000    # ~1.2M constraints
#   bash scripts/setup_benchmarks.sh 50blocks        # ~3M constraints
#   bash scripts/setup_benchmarks.sh ecdsa4          # ~6M constraints
#   bash scripts/setup_benchmarks.sh keccak40        # ~9.5M constraints
#   bash scripts/setup_benchmarks.sh all             # all four (hours + ~30 GB disk)
#
# Prerequisites:
#   1. circom 2.1+
#   2. snarkjs 0.7+       npm install -g snarkjs
#   3. circomlib           npm install           (in repo root)
#   4. circom-ecdsa + keccak256-circom libs:
#        bash scripts/download_circom_libs.sh   (needed for ecdsa4 and keccak40)
#
# PTAU files are downloaded automatically to ptau/ on first use:
#   pot23 (~4 GB) : poseidon5000, sha256 50blocks, ecdsa4
#   pot24 (~9 GB) : keccak40
# =============================================================================

set -euo pipefail
# Increase Node.js heap limit — required by snarkjs groth16 setup and zkey contribute
# on large circuits (ecdsa4 ~6M, keccak40 ~9.5M) which can use 30-80 GB of RAM.
# Without this, snarkjs will crash with JavaScript heap out of memory.
export NODE_OPTIONS="--max-old-space-size=100000"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

TARGET="${1:-all}"

COMMON_CIRCOM_FLAGS=(
    -l "$ROOT/node_modules"
    -l "$ROOT/third_party/circom-ecdsa/circuits"
    -l "$ROOT/third_party/keccak256-circom"
    -l "$ROOT/third_party/benchmark-app/mopro-core/examples/circom/anonAadhaar"
)

PTAU_DIR="ptau"
mkdir -p "$PTAU_DIR"

log() { echo "[setup] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

require() { command -v "$1" >/dev/null 2>&1 || die "$1 not found."; }

get_ptau() {
    local LEVEL="$1"
    local FILE="$PTAU_DIR/pot${LEVEL}_final.ptau"
    if [ -f "$FILE" ]; then echo "$FILE"; return; fi
    log "Downloading pot${LEVEL} ($([ "$LEVEL" -ge 24 ] && echo '~9 GB' || echo '~4 GB'))..."
    wget -q --show-progress \
        "https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_${LEVEL}.ptau" \
        -O "$FILE" || die "Download failed for pot${LEVEL}"
    echo "$FILE"
}

ensure_symlink() {
    local TARGET_PATH="$1" LINK_PATH="$2"
    mkdir -p "$(dirname "$LINK_PATH")"
    if [ -L "$LINK_PATH" ]; then
        [ "$(readlink "$LINK_PATH")" = "$TARGET_PATH" ] && return
        rm -f "$LINK_PATH"
    elif [ -e "$LINK_PATH" ]; then
        die "$LINK_PATH exists and is not a symlink"
    fi
    ln -s "$TARGET_PATH" "$LINK_PATH"
}

# circomlib is accessed via node_modules from the repo root.
# circom-ecdsa and keccak256-circom expect node_modules alongside their sources.
prepare_circom_deps() {
    [ -d "$ROOT/node_modules/circomlib" ] || \
        die "node_modules/circomlib missing. Run: npm install"
    if [ -d "$ROOT/third_party/circom-ecdsa" ]; then
        ensure_symlink "$ROOT/node_modules" "$ROOT/third_party/circom-ecdsa/node_modules"
    fi
}

generate_witness_cpp() {
    local DIR="$1"
    if [ ! -f "$DIR/circuit_cpp/circuit" ]; then
        log "Compiling C++ witness generator..."
        make -j -C "$DIR/circuit_cpp"
    fi
    if [ ! -f "$DIR/witness.wtns" ]; then
        log "Generating witness..."
        "$DIR/circuit_cpp/circuit" "$DIR/input.json" "$DIR/witness.wtns"
    fi
}

# ---------------------------------------------------------------------------
# Poseidon x5000  (~1.2M constraints, pot23)
# ---------------------------------------------------------------------------
setup_poseidon5000() {
    local DIR="test/poseidon_5000"
    log "=== poseidon_5000 ==="

    if [ ! -f "$DIR/circuit.r1cs" ]; then
        log "Compiling circuit..."
        circom "$DIR/circuit.circom" "${COMMON_CIRCOM_FLAGS[@]}" --r1cs --c --sym -o "$DIR/"
    fi

    local PTAU; PTAU=$(get_ptau 23)

    if [ ! -f "$DIR/circuit_final.zkey" ]; then
        log "Generating zkey..."
        snarkjs groth16 setup "$DIR/circuit.r1cs" "$PTAU" "$DIR/circuit_0000.zkey" 2>&1 | tail -2
        snarkjs zkey contribute "$DIR/circuit_0000.zkey" "$DIR/circuit_final.zkey" \
            --name="zyklop" -e="zyklop poseidon5000" 2>&1 | tail -2
    fi

    [ -f "$DIR/verification_key.json" ] || \
        snarkjs zkey export verificationkey "$DIR/circuit_final.zkey" "$DIR/verification_key.json"

    generate_witness_cpp "$DIR"
    log "poseidon_5000 done."
}

# ---------------------------------------------------------------------------
# SHA-256 x50  (~3M constraints, pot23)
# ---------------------------------------------------------------------------
setup_50blocks() {
    local DIR="test/sha256_50blocks"
    log "=== sha256_50blocks ==="

    if [ ! -f "$DIR/circuit.r1cs" ]; then
        log "Compiling circuit..."
        circom "$DIR/circuit.circom" "${COMMON_CIRCOM_FLAGS[@]}" --r1cs --c --sym -o "$DIR/"
    fi

    local PTAU; PTAU=$(get_ptau 23)

    if [ ! -f "$DIR/circuit_final.zkey" ]; then
        log "Generating zkey..."
        snarkjs groth16 setup "$DIR/circuit.r1cs" "$PTAU" "$DIR/circuit_0000.zkey" 2>&1 | tail -2
        snarkjs zkey contribute "$DIR/circuit_0000.zkey" "$DIR/circuit_final.zkey" \
            --name="zyklop" -e="zyklop sha256 50blocks" 2>&1 | tail -2
    fi

    [ -f "$DIR/verification_key.json" ] || \
        snarkjs zkey export verificationkey "$DIR/circuit_final.zkey" "$DIR/verification_key.json"

    generate_witness_cpp "$DIR"
    log "sha256_50blocks done."
}

# ---------------------------------------------------------------------------
# ECDSA verify x4  (~6M constraints, pot23)
# Requires: bash scripts/download_circom_libs.sh
# WARNING: circom compilation is very RAM-intensive (32+ GB recommended)
# ---------------------------------------------------------------------------
setup_ecdsa4() {
    local DIR="test/ecdsa_4x"
    log "=== ecdsa_4x ==="

    [ -d "$ROOT/third_party/circom-ecdsa" ] || \
        die "third_party/circom-ecdsa not found. Run: bash scripts/download_circom_libs.sh"

    prepare_circom_deps

    if [ ! -f "$DIR/circuit.r1cs" ]; then
        log "Compiling circuit (may take 30+ min and need 32+ GB RAM)..."
        circom "$DIR/circuit.circom" "${COMMON_CIRCOM_FLAGS[@]}" --r1cs --c --sym -o "$DIR/"
    fi

    local PTAU; PTAU=$(get_ptau 23)

    if [ ! -f "$DIR/circuit_final.zkey" ]; then
        log "Generating zkey..."
        snarkjs groth16 setup "$DIR/circuit.r1cs" "$PTAU" "$DIR/circuit_0000.zkey" 2>&1 | tail -2
        snarkjs zkey contribute "$DIR/circuit_0000.zkey" "$DIR/circuit_final.zkey" \
            --name="zyklop" -e="zyklop ecdsa4" 2>&1 | tail -2
    fi

    [ -f "$DIR/verification_key.json" ] || \
        snarkjs zkey export verificationkey "$DIR/circuit_final.zkey" "$DIR/verification_key.json"

    generate_witness_cpp "$DIR"
    log "ecdsa_4x done."
}

# ---------------------------------------------------------------------------
# Keccak-256 x40  (~9.5M constraints, pot24 ~9 GB)
# Requires: bash scripts/download_circom_libs.sh
# ---------------------------------------------------------------------------
setup_keccak40() {
    local DIR="test/keccak_40"
    log "=== keccak_40 ==="

    [ -d "$ROOT/third_party/keccak256-circom" ] || \
        die "third_party/keccak256-circom not found. Run: bash scripts/download_circom_libs.sh"

    if [ ! -f "$DIR/circuit.r1cs" ]; then
        log "Compiling circuit..."
        circom "$DIR/circuit.circom" "${COMMON_CIRCOM_FLAGS[@]}" --r1cs --c --sym -o "$DIR/"
    fi

    local PTAU; PTAU=$(get_ptau 24)

    if [ ! -f "$DIR/circuit_final.zkey" ]; then
        log "Generating zkey (large circuit, may take 1+ h)..."
        snarkjs groth16 setup "$DIR/circuit.r1cs" "$PTAU" "$DIR/circuit_0000.zkey" 2>&1 | tail -2
        snarkjs zkey contribute "$DIR/circuit_0000.zkey" "$DIR/circuit_final.zkey" \
            --name="zyklop" -e="zyklop keccak40" 2>&1 | tail -2
    fi

    [ -f "$DIR/verification_key.json" ] || \
        snarkjs zkey export verificationkey "$DIR/circuit_final.zkey" "$DIR/verification_key.json"

    generate_witness_cpp "$DIR"
    log "keccak_40 done."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
require circom
require snarkjs
require python3

case "$TARGET" in
    poseidon5000) setup_poseidon5000 ;;
    50blocks)     setup_50blocks ;;
    ecdsa4)       setup_ecdsa4 ;;
    keccak40)     setup_keccak40 ;;
    all)
        setup_poseidon5000
        setup_50blocks
        setup_ecdsa4
        setup_keccak40
        ;;
    *)
        die "Unknown target: '$TARGET'"
        echo "Usage: $0 [poseidon5000|50blocks|ecdsa4|keccak40|all]" >&2
        exit 1
        ;;
esac

log "Done. Run 'make bench' to benchmark."
