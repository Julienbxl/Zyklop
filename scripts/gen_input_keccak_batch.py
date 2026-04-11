#!/usr/bin/env python3

import json


def bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def main() -> None:
    messages = []
    for i in range(40):
        msg = f"Zyklop keccak batch #{i:02d}".encode("ascii")
        padded = msg + b"\x00" * (32 - len(msg))
        messages.append(bytes_to_bits(padded))

    print(json.dumps({"in": messages}, indent=1))


if __name__ == "__main__":
    main()
