#!/usr/bin/env python3
"""Measure the throughput cost of SENTENCE_SPLITTER_MODE, in document mode.

SENTENCE mode fixed the as-deployed under-generation (see REPORT.md) but produces
6-8x more chunks than packing sentences to a token budget. CTranslate2 batches
internally, so the extra chunks may cost nothing -- this measures whether that is
actually true, because it is the only plausible price of the fix.

Runs inside the component image with the component imported directly, so model
load and container startup are excluded from the timings:

  docker run --rm --gpus '"device=0"' \\
    -v "$PWD/eval":/eval:ro -v "$PWD/eval/results":/results:ro \\
    --entrypoint /opt/mpf/plugin-venv/bin/python IMAGE /eval/bench_split_mode.py \\
    --pair bn-en -n 200
"""
import argparse
import statistics
import time

import mpf_component_api as mpf
from nllb_component import NllbTranslationComponent

PAIRS = {
    "bn-en": ("ben", "Beng"),
    "zh-en": ("zho", "Hans"),
    "fa-en": ("pes", "Arab"),
    "pt-en": ("por", "Latn"),
}

MODES = ("SENTENCE", "DEFAULT")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="bn-en", choices=sorted(PAIRS))
    ap.add_argument("-n", type=int, default=200, help="sentences of the sample to use")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--results", default="/results", help="dir holding <pair>/sample.src")
    args = ap.parse_args()

    lang, script = PAIRS[args.pair]
    with open(f"{args.results}/{args.pair}/sample.src", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f][: args.n]
    blob = "\n".join(lines)

    component = NllbTranslationComponent()

    def run(mode):
        props = {
            "DEFAULT_SOURCE_LANGUAGE": lang,
            "DEFAULT_SOURCE_SCRIPT": script,
            "SENTENCE_SPLITTER_MODE": mode,
        }
        job = mpf.GenericJob("bench", "x.txt", props, {},
                             mpf.GenericTrack(-1, dict(TEXT=blob)))
        start = time.perf_counter()
        out = component.get_detections_from_generic(job)[0]
        return time.perf_counter() - start, out.detection_properties["TRANSLATION"]

    # Warm up so the first timed run does not absorb lazy CUDA/tokenizer setup.
    run("SENTENCE")

    print(f"pair={args.pair} sentences={len(lines)} repeats={args.repeats}")
    print(f"{'mode':10} {'median s':>9} {'sent/s':>8} {'chars/s':>9} {'out_chars':>10}")
    results = {}
    for mode in MODES:
        times, text = [], ""
        for _ in range(args.repeats):
            elapsed, text = run(mode)
            times.append(elapsed)
        med = statistics.median(times)
        results[mode] = med
        print(f"{mode:10} {med:9.2f} {len(lines)/med:8.2f} {len(text)/med:9.0f} {len(text):10d}")

    slow, fast = results["SENTENCE"], results["DEFAULT"]
    ratio = slow / fast if fast else float("nan")
    verdict = ("SENTENCE is FASTER" if ratio < 1
               else f"SENTENCE costs {(ratio - 1) * 100:.0f}% more wall time")
    print(f"\nSENTENCE / DEFAULT wall-time ratio: {ratio:.2f}x  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
