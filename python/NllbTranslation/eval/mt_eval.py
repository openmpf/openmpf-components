#!/usr/bin/env python3
"""Evaluate OpenMPF NLLB translation output against a reference translation.

Reads an OpenMPF JSON output file, extracts the machine-translation
hypotheses, and scores them against a plain-text reference (one segment per
line) using sacrebleu. Reports BLEU, chrF, chrF++ and — when it is cheap and
safe to do so — TER, in a table and/or as CSV.

Alignment note
--------------
The OpenMPF NLLB component re-segments its input with its own sentence
splitter and joins the translated output back together, so a job run over a
whole file typically emits a SINGLE track whose TRANSLATION property is one
continuous blob with no line breaks. That blob cannot be scored line-by-line
against a 1200-line reference. When the number of extracted hypotheses does
not match the number of reference lines, this tool falls back to
DOCUMENT-level scoring: it concatenates both sides into one segment and scores
that. BLEU/chrF are corpus metrics, so this is a faithful measure; it is
called out explicitly in the output so the result is never misread as
segment-aligned.

TER safety guard
----------------
TER computes edit distance WITH word shifting, which is ~O(n^2-n^3) in time
and memory in the segment length. On a single ~18k-word document it allocates
several GB and can trigger the OOM killer. TER is therefore skipped by default
whenever any segment exceeds --max-ter-tokens words. Pass --force-ter to
override (only sensible on properly segment-aligned, short-line input).

Usage
-----
    ./venv/bin/python3 mt_eval.py <openmpf.json> -r <reference.txt>

Example (this project):
    ./venv/bin/python3 mt_eval.py \
        translations/TED2020.en-pt.pt.1200.ctranslate2.json \
        -r moses/OPUS-TED2020/en-pt/TED2020.en-pt.en.1200 \
        --csv scores.csv
"""

import argparse
import csv
import json
import sys

from sacrebleu.metrics import BLEU, CHRF, TER


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_hypotheses(data):
    """Return the translation strings from an OpenMPF output dict, in order.

    Walks media[*].output.<ACTION>[*].tracks[*].trackProperties.TRANSLATION.
    Any track carrying a TRANSLATION property is collected, so this works
    regardless of how many media, actions, tracks or detections are present.
    """
    hyps = []
    for medium in data.get("media") or []:
        output = medium.get("output") or {}
        for _action_type, results in output.items():
            for result in results or []:
                for track in result.get("tracks") or []:
                    props = track.get("trackProperties") or {}
                    if "TRANSLATION" in props:
                        hyps.append(props["TRANSLATION"])
    return hyps


def read_reference(path):
    """Read a reference/source file as a list of segments (one per line)."""
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def build_streams(hyps, refs, mode):
    """Resolve scoring mode and return (mode, sys_stream, ref_stream).

    segment  -> one hypothesis per reference line (requires equal counts)
    document -> everything concatenated into a single segment per side
    auto     -> segment if counts match, else document
    """
    n_hyp, n_ref = len(hyps), len(refs)

    if mode == "auto":
        mode = "segment" if n_hyp == n_ref else "document"

    if mode == "segment":
        if n_hyp != n_ref:
            raise SystemExit(
                f"error: segment mode needs equal counts, got "
                f"{n_hyp} hypotheses vs {n_ref} reference lines.\n"
                f"       This OpenMPF output is not segment-aligned; use "
                f"--mode document (or auto)."
            )
        return mode, list(hyps), list(refs)

    # document mode: collapse each side to a single segment
    sys_doc = " ".join(h.strip() for h in hyps if h.strip())
    ref_doc = " ".join(r.strip() for r in refs if r.strip())
    return mode, [sys_doc], [ref_doc]


def max_segment_tokens(*streams):
    """Largest whitespace-token count across all segments in the given streams."""
    longest = 0
    for stream in streams:
        for seg in stream:
            longest = max(longest, len(seg.split()))
    return longest


def score(sys_stream, ref_stream, tokenize, compute_ter):
    """Compute the metrics. Returns a list of row dicts."""
    rows = []

    bleu = BLEU(tokenize=tokenize)
    r = bleu.corpus_score(sys_stream, [ref_stream])
    rows.append({
        "metric": "BLEU",
        "score": f"{r.score:.2f}",
        "detail": (
            f"{'/'.join(f'{p:.1f}' for p in r.precisions)} "
            f"(BP={r.bp:.3f} ratio={r.ratio:.3f} "
            f"hyp={r.sys_len} ref={r.ref_len})"
        ),
        "signature": bleu.get_signature().format(short=True),
    })

    chrf = CHRF()  # chrF (char n-grams, word_order=0)
    r = chrf.corpus_score(sys_stream, [ref_stream])
    rows.append({
        "metric": "chrF",
        "score": f"{r.score:.2f}",
        "detail": "character F-score (beta=2)",
        "signature": chrf.get_signature().format(short=True),
    })

    chrfpp = CHRF(word_order=2)  # chrF++
    r = chrfpp.corpus_score(sys_stream, [ref_stream])
    rows.append({
        "metric": "chrF++",
        "score": f"{r.score:.2f}",
        "detail": "chrF + word bigrams",
        "signature": chrfpp.get_signature().format(short=True),
    })

    if compute_ter:
        ter = TER()
        r = ter.corpus_score(sys_stream, [ref_stream])
        rows.append({
            "metric": "TER",
            "score": f"{r.score:.2f}",
            "detail": f"{r.num_edits} edits / {r.ref_length} ref words (lower is better)",
            "signature": ter.get_signature().format(short=True),
        })
    else:
        rows.append({
            "metric": "TER",
            "score": "skipped",
            "detail": "segment too long — would risk OOM (see --max-ter-tokens / --force-ter)",
            "signature": "",
        })

    return rows


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_table(rows, meta, stream=sys.stdout):
    """Print a metadata block and an aligned metrics table."""
    print("=" * 66, file=stream)
    print("Machine Translation Evaluation", file=stream)
    print("=" * 66, file=stream)
    for key, value in meta:
        print(f"  {key:<16}{value}", file=stream)
    print("-" * 66, file=stream)

    headers = ("Metric", "Score", "Detail")
    widths = [
        max(len(headers[0]), max(len(r["metric"]) for r in rows)),
        max(len(headers[1]), max(len(r["score"]) for r in rows)),
    ]
    print(f"  {headers[0]:<{widths[0]}}  {headers[1]:>{widths[1]}}  {headers[2]}",
          file=stream)
    print(f"  {'-' * widths[0]}  {'-' * widths[1]}  {'-' * 6}", file=stream)
    for r in rows:
        print(f"  {r['metric']:<{widths[0]}}  {r['score']:>{widths[1]}}  {r['detail']}",
              file=stream)
    print("-" * 66, file=stream)
    print("Signatures (for reproducibility):", file=stream)
    for r in rows:
        if r["signature"]:
            print(f"  {r['metric']:<8}{r['signature']}", file=stream)
    print("=" * 66, file=stream)


def write_csv(rows, path):
    """Write metric,score,detail,signature rows as CSV. '-' means stdout."""
    fields = ["metric", "score", "detail", "signature"]
    handle = sys.stdout if path == "-" else open(path, "w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if handle is not sys.stdout:
            handle.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score OpenMPF NLLB translation output against a reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  ./venv/bin/python3 mt_eval.py \\\n"
            "      translations/TED2020.en-pt.pt.1200.ctranslate2.json \\\n"
            "      -r moses/OPUS-TED2020/en-pt/TED2020.en-pt.en.1200 --csv scores.csv"
        ),
    )
    parser.add_argument("json_file", help="OpenMPF JSON output file")
    parser.add_argument("-r", "--reference", required=True,
                        help="reference translation, one segment per line")
    parser.add_argument("-m", "--mode", choices=("auto", "segment", "document"),
                        default="auto",
                        help="scoring granularity (default: auto)")
    parser.add_argument("-t", "--tokenize", default=None,
                        help="sacrebleu BLEU tokenizer (default: 13a)")
    parser.add_argument("--max-ter-tokens", type=int, default=250,
                        help="skip TER if any segment exceeds this many words "
                             "(default: 250; guards against OOM)")
    parser.add_argument("--force-ter", action="store_true",
                        help="compute TER regardless of segment length (unsafe "
                             "on long documents)")
    parser.add_argument("--csv", metavar="PATH",
                        help="also write results as CSV ('-' for stdout)")
    args = parser.parse_args(argv)

    with open(args.json_file, encoding="utf-8") as fh:
        data = json.load(fh)

    hyps = extract_hypotheses(data)
    if not hyps:
        raise SystemExit(
            f"error: no TRANSLATION tracks found in {args.json_file}. "
            f"Is this an OpenMPF translation output file?"
        )
    refs = read_reference(args.reference)

    mode, sys_stream, ref_stream = build_streams(hyps, refs, args.mode)

    longest = max_segment_tokens(sys_stream, ref_stream)
    compute_ter = args.force_ter or longest <= args.max_ter_tokens
    if not compute_ter:
        print(
            f"warning: longest segment is {longest} words (> --max-ter-tokens="
            f"{args.max_ter_tokens}); skipping TER to avoid excessive memory use. "
            f"Pass --force-ter to override.",
            file=sys.stderr,
        )

    rows = score(sys_stream, ref_stream, args.tokenize, compute_ter)

    meta = [
        ("Hypothesis:", args.json_file),
        ("Reference:", args.reference),
        ("Mode:", f"{mode}  ({len(hyps)} hypothesis segment(s), "
                  f"{len(refs)} reference line(s))"),
        ("Longest seg:", f"{longest} words"),
    ]
    if mode == "document":
        meta.append(("Note:", "output is a single blob -> scored as one "
                              "concatenated document, not line-by-line"))

    print_table(rows, meta)
    if args.csv:
        write_csv(rows, args.csv)
        if args.csv != "-":
            print(f"\nCSV written to {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
