#!/usr/bin/env python
"""Merge JSONL files without changing record order."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for input_path in args.inputs:
            with Path(input_path).open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        target.write(line.rstrip("\n") + "\n")
    print(output)


if __name__ == "__main__":
    main()
