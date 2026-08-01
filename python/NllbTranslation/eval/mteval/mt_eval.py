#!/usr/bin/env python3
"""Evaluate NLLB translation output against a reference translation.

Two subcommands:

  score    Score ONE system. Hypotheses come from an OpenMPF JSON output file
           or a plain-text file (one translation per line). Reports BLEU, chrF,
           chrF++, TER (guarded), and optionally COMET. Falls back to
           document-level scoring when the hypothesis is a single un-segmented
           blob (see below).

  compare  Score TWO systems on the same aligned reference and report a
           side-by-side table with deltas and paired-bootstrap significance.
           This is the fp16-vs-int8 comparison (Axis A, per-sentence).

Alignment note
--------------
A whole-file OpenMPF job emits a SINGLE joined TRANSLATION blob (input newlines
are consumed by the sentence splitter), which cannot be scored line-by-line.
When hypotheses != reference lines, `score` falls back to DOCUMENT mode
(concatenate both sides, one segment) and says so. The per-line driver
(nllb_eval_driver.py) produces properly aligned text files for segment mode.

TER safety guard
----------------
TER does edit distance WITH shifting (~O(n^2-n^3) memory in segment length); on
a single ~18k-word document it can OOM. TER is skipped when any segment exceeds
--max-ter-tokens words unless --force-ter is given.

COMET
-----
COMET is a neural, reference+source metric (needs --source). It requires the
`unbabel-comet` package and a model download, and is only meaningful on
segment-aligned data (not document mode).

Examples
--------
  ./venv/bin/python3 -m mteval.mt_eval score \
      translations/TED2020.en-pt.pt.1200.ctranslate2.json \
      -r moses/OPUS-TED2020/en-pt/TED2020.en-pt.en.1200

  ./venv/bin/python3 -m mteval.mt_eval compare \
      --hyp fp16=results/hyp.develop.en \
      --hyp int8=results/hyp.ctranslate2.en \
      -r eval_data/sample5000.en -s eval_data/sample5000.pt \
      --comet --bootstrap 1000 --per-segment results/segments.csv
"""

import argparse
import csv
import json
import sys

import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def extract_openmpf_hypotheses(data):
    """Translation strings from an OpenMPF output dict, in order.

    Walks media[*].output.<ACTION>[*].tracks[*].trackProperties.TRANSLATION.
    """
    hyps = []
    for medium in data.get("media") or []:
        for _action, results in (medium.get("output") or {}).items():
            for result in results or []:
                for track in result.get("tracks") or []:
                    props = track.get("trackProperties") or {}
                    if "TRANSLATION" in props:
                        hyps.append(props["TRANSLATION"])
    return hyps


def load_hypotheses(path):
    """Return a list of hypothesis segments from a .json (OpenMPF) or .txt file."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        hyps = extract_openmpf_hypotheses(data)
        if not hyps:
            raise SystemExit(f"error: no TRANSLATION tracks found in {path}")
        return hyps
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def max_segment_tokens(*streams):
    longest = 0
    for stream in streams:
        for seg in stream:
            longest = max(longest, len(seg.split()))
    return longest


def surface_metrics(sys_stream, ref_stream, tokenize, compute_ter):
    """Compute BLEU/chrF/chrF++/(TER) over aligned streams. Returns row dicts."""
    rows = []

    bleu = BLEU(tokenize=tokenize)
    r = bleu.corpus_score(sys_stream, [ref_stream])
    rows.append({"metric": "BLEU", "score": r.score,
                 "detail": (f"{'/'.join(f'{p:.1f}' for p in r.precisions)} "
                            f"(BP={r.bp:.3f} ratio={r.ratio:.3f} "
                            f"hyp={r.sys_len} ref={r.ref_len})"),
                 "signature": bleu.get_signature().format(short=True)})

    chrf = CHRF()
    r = chrf.corpus_score(sys_stream, [ref_stream])
    rows.append({"metric": "chrF", "score": r.score,
                 "detail": "character F-score (beta=2)",
                 "signature": chrf.get_signature().format(short=True)})

    chrfpp = CHRF(word_order=2)
    r = chrfpp.corpus_score(sys_stream, [ref_stream])
    rows.append({"metric": "chrF++", "score": r.score,
                 "detail": "chrF + word bigrams",
                 "signature": chrfpp.get_signature().format(short=True)})

    if compute_ter:
        ter = TER()
        r = ter.corpus_score(sys_stream, [ref_stream])
        rows.append({"metric": "TER", "score": r.score,
                     "detail": f"{r.num_edits} edits / {r.ref_length} ref words "
                               f"(lower is better)",
                     "signature": ter.get_signature().format(short=True)})
    else:
        rows.append({"metric": "TER", "score": None,
                     "detail": "skipped — segment too long (see --max-ter-tokens)",
                     "signature": ""})
    return rows


# COMET is imported lazily so the tool works without it installed.
def comet_available():
    try:
        import comet  # noqa: F401
        return True
    except Exception:
        return False


def comet_score(srcs, hyps, refs, model_name, gpus=None):
    """Return (system_score, [segment_scores]) using a COMET model.

    gpus: 0 forces CPU; >=1 uses that many GPUs (which device is chosen by
    CUDA_VISIBLE_DEVICES). None = auto (1 if a GPU is visible, else CPU).
    """
    from comet import download_model, load_from_checkpoint
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    if gpus is None:
        try:
            import torch
            gpus = 1 if torch.cuda.is_available() else 0
        except Exception:
            gpus = 0
    out = model.predict(data, batch_size=64, gpus=gpus, progress_bar=True)
    return float(out["system_score"]), [float(x) for x in out["scores"]]


# --------------------------------------------------------------------------- #
# Paired bootstrap significance (Koehn 2004)
# --------------------------------------------------------------------------- #
def _corpus_point(metric_obj, hyps, refs, idx):
    resampled_h = [hyps[i] for i in idx]
    resampled_r = [refs[i] for i in idx]
    return metric_obj.corpus_score(resampled_h, [resampled_r]).score


def paired_bootstrap(ref, hyp_a, hyp_b, n_samples, seed,
                     comet_seg_a=None, comet_seg_b=None):
    """Paired bootstrap resampling. Returns dict of metric -> stats.

    For BLEU/chrF each resample is re-scored with sacrebleu; COMET reuses
    precomputed per-segment scores (mean over the resample). Higher-is-better
    for all three here (COMET, BLEU, chrF).
    """
    rng = np.random.default_rng(seed)
    n = len(ref)
    metrics = {"BLEU": BLEU(), "chrF": CHRF()}

    # Observed full-corpus deltas (B - A).
    observed = {name: (m.corpus_score(hyp_b, [ref]).score
                       - m.corpus_score(hyp_a, [ref]).score)
                for name, m in metrics.items()}
    deltas = {name: [] for name in metrics}
    if comet_seg_a is not None:
        # COMET reported x100 to match the table's scale.
        observed["COMET"] = float((np.mean(comet_seg_b) - np.mean(comet_seg_a)) * 100)
        deltas["COMET"] = []

    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        for name, m in metrics.items():
            deltas[name].append(_corpus_point(m, hyp_b, ref, idx)
                                - _corpus_point(m, hyp_a, ref, idx))
        if comet_seg_a is not None:
            a = np.mean([comet_seg_a[i] for i in idx])
            b = np.mean([comet_seg_b[i] for i in idx])
            deltas["COMET"].append(float((b - a) * 100))

    stats = {}
    for name, dl in deltas.items():
        arr = np.array(dl)
        # Two-sided p: how often the resampled delta sign opposes the observed.
        obs = observed[name]
        if obs >= 0:
            p = 2.0 * np.mean(arr <= 0)
        else:
            p = 2.0 * np.mean(arr >= 0)
        stats[name] = {"observed_delta": obs,
                       "ci_low": float(np.percentile(arr, 2.5)),
                       "ci_high": float(np.percentile(arr, 97.5)),
                       "p_value": float(min(p, 1.0))}
    return stats


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def fmt_score(v, nd=2):
    return "skipped" if v is None else f"{v:.{nd}f}"


def print_meta(meta, stream=sys.stdout):
    print("=" * 74, file=stream)
    print("Machine Translation Evaluation", file=stream)
    print("=" * 74, file=stream)
    for k, v in meta:
        print(f"  {k:<16}{v}", file=stream)
    print("-" * 74, file=stream)


def print_score_table(rows, stream=sys.stdout):
    w = max(len("Metric"), max(len(r["metric"]) for r in rows))
    print(f"  {'Metric':<{w}}  {'Score':>8}  Detail", file=stream)
    print(f"  {'-'*w}  {'-'*8}  {'-'*6}", file=stream)
    for r in rows:
        print(f"  {r['metric']:<{w}}  {fmt_score(r['score']):>8}  {r['detail']}",
              file=stream)
    print("-" * 74, file=stream)
    print("Signatures:", file=stream)
    for r in rows:
        if r["signature"]:
            print(f"  {r['metric']:<8}{r['signature']}", file=stream)
    print("=" * 74, file=stream)


def print_compare_table(name_a, rows_a, name_b, rows_b, higher_better, stream=sys.stdout):
    by_b = {r["metric"]: r for r in rows_b}
    metrics = [r["metric"] for r in rows_a]
    w = max(8, max(len(m) for m in metrics))
    cw = max(len(name_a), len(name_b), 8)
    print(f"  {'Metric':<{w}}  {name_a:>{cw}}  {name_b:>{cw}}  {'Δ(B-A)':>9}  Better",
          file=stream)
    print(f"  {'-'*w}  {'-'*cw}  {'-'*cw}  {'-'*9}  {'-'*6}", file=stream)
    for r in rows_a:
        m = r["metric"]
        a, b = r["score"], by_b[m]["score"]
        if a is None or b is None:
            print(f"  {m:<{w}}  {fmt_score(a):>{cw}}  {fmt_score(b):>{cw}}  "
                  f"{'—':>9}  —", file=stream)
            continue
        delta = b - a
        hb = higher_better[m]
        winner = name_b if (delta > 0) == hb else name_a
        if abs(delta) < 1e-9:
            winner = "tie"
        print(f"  {m:<{w}}  {a:>{cw}.2f}  {b:>{cw}.2f}  {delta:>+9.2f}  {winner}",
              file=stream)
    print("-" * 74, file=stream)


def write_csv(rows, path, fields):
    handle = sys.stdout if path == "-" else open(path, "w", newline="", encoding="utf-8")
    try:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    finally:
        if handle is not sys.stdout:
            handle.close()


# --------------------------------------------------------------------------- #
# Subcommand: score
# --------------------------------------------------------------------------- #
def cmd_score(args):
    hyps = load_hypotheses(args.hyp)
    refs = read_lines(args.reference)

    mode = args.mode
    if mode == "auto":
        mode = "segment" if len(hyps) == len(refs) else "document"

    if mode == "segment":
        if len(hyps) != len(refs):
            raise SystemExit(f"error: segment mode needs equal counts, got "
                             f"{len(hyps)} hyp vs {len(refs)} ref")
        sys_stream, ref_stream = hyps, refs
    else:
        sys_stream = [" ".join(h.strip() for h in hyps if h.strip())]
        ref_stream = [" ".join(r.strip() for r in refs if r.strip())]

    longest = max_segment_tokens(sys_stream, ref_stream)
    compute_ter = args.force_ter or longest <= args.max_ter_tokens
    if not compute_ter:
        print(f"warning: longest segment {longest} words > --max-ter-tokens="
              f"{args.max_ter_tokens}; skipping TER (use --force-ter).",
              file=sys.stderr)

    rows = surface_metrics(sys_stream, ref_stream, args.tokenize, compute_ter)

    if args.comet:
        if mode != "segment":
            print("warning: COMET needs segment-aligned data; skipping in document mode.",
                  file=sys.stderr)
        elif not args.source:
            raise SystemExit("error: --comet requires --source")
        elif not comet_available():
            raise SystemExit("error: --comet needs the 'unbabel-comet' package "
                             "(pip install unbabel-comet)")
        else:
            srcs = read_lines(args.source)
            sysc, _ = comet_score(srcs, hyps, refs, args.comet_model, args.comet_gpus)
            rows.append({"metric": "COMET", "score": sysc * 100,
                         "detail": f"{args.comet_model} (x100)", "signature": ""})

    meta = [("System:", args.hyp), ("Reference:", args.reference),
            ("Mode:", f"{mode}  ({len(hyps)} hyp seg, {len(refs)} ref lines)"),
            ("Longest seg:", f"{longest} words")]
    if mode == "document":
        meta.append(("Note:", "single blob -> scored as one concatenated document"))
    print_meta(meta)
    print_score_table(rows)
    if args.csv:
        write_csv(rows, args.csv, ["metric", "score", "detail", "signature"])
        if args.csv != "-":
            print(f"\nCSV written to {args.csv}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Subcommand: compare
# --------------------------------------------------------------------------- #
def parse_named_hyp(values):
    out = []
    for v in values:
        if "=" not in v:
            raise SystemExit(f"--hyp expects NAME=PATH, got {v!r}")
        name, path = v.split("=", 1)
        out.append((name, path))
    return out


def cmd_compare(args):
    systems = parse_named_hyp(args.hyp)
    if len(systems) != 2:
        raise SystemExit("compare requires exactly two --hyp NAME=PATH entries")
    refs = read_lines(args.reference)
    (name_a, path_a), (name_b, path_b) = systems
    hyp_a, hyp_b = load_hypotheses(path_a), load_hypotheses(path_b)

    for nm, h in ((name_a, hyp_a), (name_b, hyp_b)):
        if len(h) != len(refs):
            raise SystemExit(f"error: {nm} has {len(h)} lines vs {len(refs)} ref "
                             f"lines — compare needs aligned per-line hypotheses.")

    longest = max_segment_tokens(hyp_a, hyp_b, refs)
    compute_ter = args.force_ter or longest <= args.max_ter_tokens
    if not compute_ter:
        print(f"warning: longest segment {longest} words; skipping TER.", file=sys.stderr)

    rows_a = surface_metrics(hyp_a, refs, args.tokenize, compute_ter)
    rows_b = surface_metrics(hyp_b, refs, args.tokenize, compute_ter)
    higher_better = {"BLEU": True, "chrF": True, "chrF++": True, "TER": False}

    comet_seg_a = comet_seg_b = None
    if args.comet:
        if not args.source:
            raise SystemExit("error: --comet requires --source")
        if not comet_available():
            raise SystemExit("error: --comet needs the 'unbabel-comet' package "
                             "(pip install unbabel-comet)")
        srcs = read_lines(args.source)
        sysc_a, comet_seg_a = comet_score(srcs, hyp_a, refs, args.comet_model, args.comet_gpus)
        sysc_b, comet_seg_b = comet_score(srcs, hyp_b, refs, args.comet_model, args.comet_gpus)
        rows_a.append({"metric": "COMET", "score": sysc_a * 100,
                       "detail": args.comet_model, "signature": ""})
        rows_b.append({"metric": "COMET", "score": sysc_b * 100,
                       "detail": args.comet_model, "signature": ""})
        higher_better["COMET"] = True

    meta = [("System A:", f"{name_a}  ({path_a})"),
            ("System B:", f"{name_b}  ({path_b})"),
            ("Reference:", args.reference),
            ("Segments:", str(len(refs)))]
    print_meta(meta)
    print_compare_table(name_a, rows_a, name_b, rows_b, higher_better)

    if args.csv:
        by_b = {r["metric"]: r for r in rows_b}
        csv_rows = []
        for r in rows_a:
            a, b = r["score"], by_b[r["metric"]]["score"]
            csv_rows.append({
                "metric": r["metric"],
                name_a: "" if a is None else round(a, 3),
                name_b: "" if b is None else round(b, 3),
                "delta_b_minus_a": "" if (a is None or b is None) else round(b - a, 3),
            })
        write_csv(csv_rows, args.csv, ["metric", name_a, name_b, "delta_b_minus_a"])

    if args.bootstrap:
        print(f"Paired bootstrap significance ({args.bootstrap} resamples, "
              f"Δ = {name_b} − {name_a}):", file=sys.stdout)
        stats = paired_bootstrap(refs, hyp_a, hyp_b, args.bootstrap, args.seed,
                                 comet_seg_a, comet_seg_b)
        print(f"  {'Metric':<8}  {'Δ':>8}  {'95% CI':>18}  {'p':>8}  sig", file=sys.stdout)
        print(f"  {'-'*8}  {'-'*8}  {'-'*18}  {'-'*8}  ---", file=sys.stdout)
        for m, s in stats.items():
            ci = f"[{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]"
            sig = "yes" if s["p_value"] < 0.05 else "no"
            print(f"  {m:<8}  {s['observed_delta']:>+8.2f}  {ci:>18}  "
                  f"{s['p_value']:>8.3f}  {sig}", file=sys.stdout)
        print("=" * 74, file=sys.stdout)

    if args.per_segment:
        write_per_segment(args.per_segment, name_a, hyp_a, name_b, hyp_b, refs,
                          args.source, comet_seg_a, comet_seg_b)
        print(f"per-segment CSV written to {args.per_segment}", file=sys.stderr)


def write_per_segment(path, name_a, hyp_a, name_b, hyp_b, refs, source_path,
                      comet_a, comet_b):
    """Per-segment sentence-chrF for each system + delta, for error analysis."""
    chrf = CHRF()
    srcs = read_lines(source_path) if source_path else [""] * len(refs)
    rows = []
    for i in range(len(refs)):
        ca = chrf.sentence_score(hyp_a[i], [refs[i]]).score
        cb = chrf.sentence_score(hyp_b[i], [refs[i]]).score
        row = {"i": i, "source": srcs[i], "reference": refs[i],
               f"hyp_{name_a}": hyp_a[i], f"hyp_{name_b}": hyp_b[i],
               f"chrf_{name_a}": round(ca, 2), f"chrf_{name_b}": round(cb, 2),
               "chrf_delta": round(cb - ca, 2)}
        if comet_a is not None:
            row[f"comet_{name_a}"] = round(comet_a[i], 4)
            row[f"comet_{name_b}"] = round(comet_b[i], 4)
            row["comet_delta"] = round(comet_b[i] - comet_a[i], 4)
        rows.append(row)
    write_csv(rows, path, list(rows[0].keys()))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common(p):
    p.add_argument("-r", "--reference", required=True, help="reference, one seg/line")
    p.add_argument("-s", "--source", help="source text (required for COMET)")
    p.add_argument("-t", "--tokenize", default=None, help="sacrebleu BLEU tokenizer")
    p.add_argument("--comet", action="store_true", help="also compute COMET")
    p.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--comet-gpus", type=int, default=None,
                   help="GPUs for COMET: 0=CPU, 1=GPU (which device via "
                        "CUDA_VISIBLE_DEVICES). Default: auto-detect.")
    p.add_argument("--max-ter-tokens", type=int, default=250)
    p.add_argument("--force-ter", action="store_true")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score", help="score one system")
    ps.add_argument("hyp", help="OpenMPF JSON or text hypothesis file")
    ps.add_argument("-m", "--mode", choices=("auto", "segment", "document"),
                    default="auto")
    ps.add_argument("--csv", metavar="PATH")
    add_common(ps)
    ps.set_defaults(func=cmd_score)

    pc = sub.add_parser("compare", help="compare two systems")
    pc.add_argument("--hyp", action="append", required=True,
                    metavar="NAME=PATH", help="repeat exactly twice")
    pc.add_argument("--bootstrap", type=int, default=0,
                    help="paired-bootstrap resamples (0=off, e.g. 1000)")
    pc.add_argument("--seed", type=int, default=12345)
    pc.add_argument("--per-segment", metavar="PATH", help="per-segment CSV out")
    pc.add_argument("--csv", metavar="PATH", help="system-level comparison CSV out")
    add_common(pc)
    pc.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
