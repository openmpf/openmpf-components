#!/usr/bin/env python3
"""Assemble the engine-vs-quantization decomposition table for one pair.

Reads the two pairwise compare CSVs and the three per-system meta files that
run_decomp.sh produced, and prints a Markdown summary: per-metric absolute
scores for HF-fp16 / CT2-fp16 / CT2-int8, the engine delta (CT2-fp16 − HF-fp16)
and quantization delta (CT2-int8 − CT2-fp16), plus throughput and speedups.

    ./venv/bin/python3 -m mteval.decomp_report results/decomp/pt-en
"""
import csv
import json
import os
import sys

METRIC_ORDER = ["BLEU", "chrF", "chrF++", "TER", "COMET"]


def read_compare_csv(path):
    """metric -> (col_a_value, col_b_value, delta) ; also returns (name_a, name_b)."""
    rows, names = {}, (None, None)
    if not os.path.exists(path):
        return rows, names
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = [c for c in r.fieldnames if c not in ("metric", "delta_b_minus_a")]
        names = (cols[0], cols[1]) if len(cols) >= 2 else (None, None)
        for row in r:
            rows[row["metric"]] = (row.get(names[0], ""), row.get(names[1], ""),
                                   row.get("delta_b_minus_a", ""))
    return rows, names


def sps(meta_path):
    try:
        with open(meta_path, encoding="utf-8") as f:
            return float(json.load(f).get("sentences_per_sec") or 0)
    except Exception:
        return None


def main():
    H = sys.argv[1] if len(sys.argv) > 1 else "."
    pair = os.path.basename(os.path.normpath(H))
    eng, _ = read_compare_csv(os.path.join(H, "decomp.engine.metrics.csv"))
    qnt, _ = read_compare_csv(os.path.join(H, "decomp.quant.metrics.csv"))

    out = [f"# Engine vs quantization decomposition — {pair}", ""]
    out += ["Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, "
            "float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.",
            "Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). "
            "Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).", ""]

    out += ["| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |",
            "|--------|---------|----------|----------|---------|--------|"]
    metrics = [m for m in METRIC_ORDER if m in eng or m in qnt]
    for m in metrics:
        hf = eng.get(m, ("", "", ""))[0]
        c16 = eng.get(m, ("", "", ""))[1] or qnt.get(m, ("", "", ""))[0]
        c8 = qnt.get(m, ("", "", ""))[1]
        d_eng = eng.get(m, ("", "", ""))[2]
        d_qnt = qnt.get(m, ("", "", ""))[2]
        out.append(f"| {m} | {hf} | {c16} | {c8} | {d_eng} | {d_qnt} |")

    # throughput
    s_hf = sps(os.path.join(H, "meta.hf-fp16.json"))
    s_16 = sps(os.path.join(H, "meta.ct2-fp16.json"))
    s_8 = sps(os.path.join(H, "meta.ct2-int8.json"))
    out += ["", "### Throughput (sentences/sec, per-line / batch 1)", ""]
    out += ["| System | sent/s | speedup vs HF-fp16 |",
            "|--------|--------|--------------------|"]
    for label, s in (("HF-fp16", s_hf), ("CT2-fp16", s_16), ("CT2-int8", s_8)):
        spd = f"{s / s_hf:.1f}x" if (s and s_hf) else "—"
        out.append(f"| {label} | {s if s is not None else '—'} | {spd} |")

    out += ["", "_Throughput here is single-sentence latency (batch 1); batched "
            "throughput widens the CTranslate2 lead further. Significance (paired "
            "bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
