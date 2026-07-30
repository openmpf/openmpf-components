#!/usr/bin/env python3
"""Combine per-pair eval outputs into results/SUMMARY.md.

Scans results/ (root + one level of subdirs) for axisA.metrics.csv (written by
`mt_eval.py compare --csv`) and axisB.{fp16,int8}.csv (from `mt_eval.py score
--csv`), and emits one comparison table across all language pairs.
"""
import csv
import glob
import os


def read_axisA(path):
    """metric -> (fp16, int8, delta) from a compare CSV (cols: metric,fp16,int8,delta_b_minus_a)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = (row.get("fp16", ""), row.get("int8", ""),
                                  row.get("delta_b_minus_a", ""))
    return out


def read_axisB_bleu(dirpath):
    """(fp16_bleu, int8_bleu) document-mode BLEU from axisB.*.csv, or ('','')."""
    vals = {}
    for m in ("fp16", "int8"):
        p = os.path.join(dirpath, f"axisB.{m}.csv")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["metric"] == "BLEU":
                        vals[m] = row["score"]
    return vals.get("fp16", ""), vals.get("int8", "")


def pair_name(metrics_path):
    d = os.path.dirname(metrics_path)
    base = os.path.basename(d)
    return "pt-en" if base == "results" else base  # root files are the pt-en run


def main():
    paths = sorted(set(glob.glob("results/axisA.metrics.csv")
                       + glob.glob("results/*/axisA.metrics.csv")))
    if not paths:
        print("no axisA.metrics.csv found under results/")
        return

    lines = ["# MT Evaluation — combined summary (fp16 vs int8)", ""]
    lines += ["Δ = int8 − fp16. Axis A = intrinsic per-sentence (beam 4 both, "
              "splitter neutralized). Axis B = as-deployed document blob.", ""]
    lines += ["| Pair | BLEU fp16 | BLEU int8 | ΔBLEU | ΔchrF | COMET fp16 | COMET int8 | ΔCOMET | AxisB ΔBLEU |",
              "|------|-----------|-----------|-------|-------|------------|------------|--------|-------------|"]

    for p in paths:
        name = pair_name(p)
        a = read_axisA(p)
        bleu = a.get("BLEU", ("", "", ""))
        chrf = a.get("chrF", ("", "", ""))
        comet = a.get("COMET", ("", "", ""))
        bf, bi = read_axisB_bleu(os.path.dirname(p))
        axisb_d = ""
        try:
            axisb_d = f"{float(bi) - float(bf):+.2f}"
        except ValueError:
            pass
        lines.append(
            f"| {name} | {bleu[0]} | {bleu[1]} | {bleu[2]} | {chrf[2]} | "
            f"{comet[0]} | {comet[1]} | {comet[2]} | {axisb_d} |"
        )

    lines += ["", "Per-pair detail: `results/<pair>/axisA.report.txt`, "
              "`axisA.segments.csv` (per-sentence + COMET), `axisB.*.report.txt`."]
    out = "results/SUMMARY.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
