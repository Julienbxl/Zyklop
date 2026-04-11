#!/usr/bin/env python3
"""
gen_input_poseidon.py -- generate input.json for the poseidon_100 circuit

Usage:
    python3 scripts/gen_input_poseidon.py > test/poseidon_100/input.json

The inputs are two arbitrary Fr field elements derived from the
ASCII string "Zyklop benchmark" for reproducibility.
"""

import json
import sys

def main():
    # Fr modulus for BN254
    r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001

    # Two deterministic field elements derived from "Zyklop benchmark"
    msg = b"Zyklop benchmark"
    in0 = int.from_bytes(msg[:16], 'big') % r
    in1 = int.from_bytes(msg[8:] + b'\x00' * 8, 'big') % r

    inp = {"in": [str(in0), str(in1)]}
    print(json.dumps(inp, indent=1))

if __name__ == "__main__":
    main()
