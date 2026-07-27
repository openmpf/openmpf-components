#!/usr/bin/env python3
"""Build a reproducible aligned test sample from the OPUS TED2020 moses corpus.

Draws a fixed-seed random sample of source/reference sentence pairs, skipping
degenerate pairs (empty or non-alphabetic on either side), and writes three
aligned files:

    <out>.pt   source (Portuguese), one sentence per line
    <out>.en   reference (English),  one sentence per line
    <out>.idx  0-based line index of each pair in the original corpus

Usage:
    python3 make_sample.py -n 5000 --seed 42 -o eval_data/sample5000
"""
import argparse
import os
import random
import re

SRC = "moses/OPUS-TED2020/en-pt/TED2020.en-pt.pt"
REF = "moses/OPUS-TED2020/en-pt/TED2020.en-pt.en"

HAS_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)  # at least one letter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=5000, help="sample size")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("-o", "--out", default="eval_data/sample5000",
                    help="output path prefix")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--ref", default=REF)
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        src = [ln.rstrip("\n") for ln in f]
    with open(args.ref, encoding="utf-8") as f:
        ref = [ln.rstrip("\n") for ln in f]
    if len(src) != len(ref):
        raise SystemExit(f"corpus not aligned: {len(src)} src vs {len(ref)} ref lines")

    # Eligible = both sides contain at least one letter (drops blanks, pure
    # punctuation/number lines that carry no translatable content).
    eligible = [i for i in range(len(src))
                if HAS_ALPHA.search(src[i]) and HAS_ALPHA.search(ref[i])]

    n = min(args.num, len(eligible))
    idx = sorted(random.Random(args.seed).sample(eligible, n))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".pt", "w", encoding="utf-8") as f:
        f.write("".join(src[i].strip() + "\n" for i in idx))
    with open(args.out + ".en", "w", encoding="utf-8") as f:
        f.write("".join(ref[i].strip() + "\n" for i in idx))
    with open(args.out + ".idx", "w", encoding="utf-8") as f:
        f.write("".join(f"{i}\n" for i in idx))

    print(f"corpus lines      : {len(src)}")
    print(f"eligible (letters): {len(eligible)}")
    print(f"sampled           : {n}  (seed={args.seed})")
    print(f"wrote             : {args.out}.pt / .en / .idx")


if __name__ == "__main__":
    main()
