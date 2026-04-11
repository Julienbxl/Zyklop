#!/usr/bin/env python3
"""
gen_input_sha256.py — generate input.json for the sha256_50blocks circuit.

The input is the ASCII string "Zyklop benchmark" zero-padded to 64 bytes,
represented as 512 individual bits in big-endian order.

Usage:
    python3 scripts/gen_input_sha256.py > test/sha256_50blocks/input.json
"""

import json


def bytes_to_bits(b: bytes) -> list:
    bits = []
    for byte in b:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def main():
    msg = b"Zyklop benchmark"
    padded = msg + b"\x00" * (64 - len(msg))
    assert len(padded) == 64
    bits = bytes_to_bits(padded)
    assert len(bits) == 512
    print(json.dumps({"in": bits}, indent=1))


if __name__ == "__main__":
    main()
