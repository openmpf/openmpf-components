#!/usr/bin/env python3
"""Extract a reproducible aligned sample from a TMX file.

Streams a (possibly huge) TMX, pulls the source-language and reference-language
segments from each <tu>, skips degenerate pairs (no letter on either side),
draws a fixed-seed random sample, and writes:

    <out>.src   source text, one segment per line
    <out>.ref   reference text (English), one segment per line
    <out>.idx   0-based index of each pair among eligible pairs

Usage:
    ./venv/bin/python3 -m mteval.tmx_sample --tmx tmx/ar-en.tmx --src-lang ar --ref-lang en \
        -n 5000 --seed 42 -o results/ar-en/sample
"""
import argparse
import os
import random
import re
import xml.etree.ElementTree as ET

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)  # any Unicode letter (incl. zh/ar)


def seg_for(tu, lang):
    """Return the <seg> text for the tuv whose xml:lang matches `lang` (prefix ok)."""
    for tuv in tu.findall("tuv"):
        code = (tuv.get(XML_LANG) or "").lower()
        if code == lang or code.split("-")[0] == lang:
            seg = tuv.find("seg")
            if seg is not None and seg.text:
                return seg.text.strip()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tmx", required=True)
    ap.add_argument("--src-lang", required=True, help="TMX lang code of source (e.g. ar, zh)")
    ap.add_argument("--ref-lang", default="en", help="TMX lang code of reference (default en)")
    ap.add_argument("-n", "--num", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", required=True, help="output path prefix")
    args = ap.parse_args()

    src_lang = args.src_lang.lower()
    ref_lang = args.ref_lang.lower()

    pairs = []
    for _ev, el in ET.iterparse(args.tmx, events=("end",)):
        if el.tag != "tu":
            continue
        s = seg_for(el, src_lang)
        r = seg_for(el, ref_lang)
        el.clear()
        if s and r and HAS_LETTER.search(s) and HAS_LETTER.search(r):
            pairs.append((s, r))

    total = len(pairs)
    n = min(args.num, total)
    idx = sorted(random.Random(args.seed).sample(range(total), n))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".src", "w", encoding="utf-8") as fs, \
         open(args.out + ".ref", "w", encoding="utf-8") as fr, \
         open(args.out + ".idx", "w", encoding="utf-8") as fi:
        for i in idx:
            fs.write(pairs[i][0].replace("\n", " ").strip() + "\n")
            fr.write(pairs[i][1].replace("\n", " ").strip() + "\n")
            fi.write(f"{i}\n")

    print(f"tmx            : {args.tmx}")
    print(f"eligible pairs : {total}")
    print(f"sampled        : {n}  (seed={args.seed}, src={src_lang}, ref={ref_lang})")
    print(f"wrote          : {args.out}.src / .ref / .idx")


if __name__ == "__main__":
    main()
