#!/usr/bin/env python3
"""Combine per-pair eval outputs into results/SUMMARY.md.

Scans results/ (root + one level of subdirs) for axisA.metrics.csv (written by
`mt_eval.py compare --csv`) and axisB.{hf,ct2}.csv (from `mt_eval.py score
--csv`), and emits one comparison table across all language pairs.

The two systems are an ENGINE contrast: CT2 is the CTranslate2-branch image
(precision follows its BUILD_TYPE) and HF is the develop/Transformers image.
Runs made before that rename used fp16/int8 labels, where "int8" meant the
CTranslate2 branch; both spellings are read so older results still summarise.
"""
import csv
import glob
import os


# (current label, legacy label) -- legacy runs used precision names for what has
# always been an engine contrast.
HF_LABELS = ("hf", "fp16")
CT2_LABELS = ("ct2", "int8")


def _first(row, labels):
    """Value for whichever of `labels` the row actually carries."""
    for label in labels:
        if row.get(label):
            return row[label]
    return ""


def read_axisA(path):
    """metric -> (hf, ct2, delta) from a compare CSV, accepting either label set."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = (_first(row, HF_LABELS), _first(row, CT2_LABELS),
                                  row.get("delta_b_minus_a", ""))
    return out


def read_axisB_bleu(dirpath):
    """(hf_bleu, ct2_bleu) document-mode BLEU from axisB.*.csv, or ('','')."""
    def bleu_for(labels):
        for label in labels:
            p = os.path.join(dirpath, f"axisB.{label}.csv")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row["metric"] == "BLEU":
                            return row["score"]
        return ""
    return bleu_for(HF_LABELS), bleu_for(CT2_LABELS)


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

    lines = ["# MT Evaluation — combined summary (HF vs CT2)", ""]
    lines += ["Δ = CT2 − HF. **CT2** = the CTranslate2-branch image; its precision follows that "
              "image's `BUILD_TYPE` (gpu → float16, cpu → int8_float32) and is **not** necessarily "
              "int8. **HF** = the develop/Transformers image (facebook/nllb-200-3.3B, fp16). "
              "Earlier runs labelled these fp16/int8.", ""]
    lines += ["Axis A = intrinsic per-sentence (beam 4 both, splitter neutralized). "
              "Axis B = as-deployed document blob.", ""]
    lines += ["| Pair | BLEU HF | BLEU CT2 | ΔBLEU | ΔchrF | COMET HF | COMET CT2 | ΔCOMET | AxisB ΔBLEU |",
              "|------|---------|----------|-------|-------|----------|-----------|--------|-------------|"]

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
