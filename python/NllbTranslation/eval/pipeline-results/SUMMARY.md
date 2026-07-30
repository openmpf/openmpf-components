# MT Evaluation — combined summary (fp16 vs int8)

Δ = int8 − fp16. Axis A = intrinsic per-sentence (beam 4 both, splitter neutralized). Axis B = as-deployed document blob.

| Pair | BLEU fp16 | BLEU int8 | ΔBLEU | ΔchrF | COMET fp16 | COMET int8 | ΔCOMET | AxisB ΔBLEU |
|------|-----------|-----------|-------|-------|------------|------------|--------|-------------|
| ar-en | 40.709 | 40.836 | 0.127 | 0.051 | 84.88 | 84.901 | 0.021 | -1.00 |
| bn-en | 33.243 | 33.146 | -0.097 | -0.081 | 85.755 | 85.744 | -0.011 | +7.57 |
| de-en | 39.037 | 38.697 | -0.34 | -0.15 | 85.939 | 85.873 | -0.066 | -0.73 |
| fa-en | 36.17 | 36.189 | 0.019 | -0.021 | 84.903 | 84.927 | 0.024 | +7.22 |
| fr-en | 42.614 | 42.32 | -0.294 | -0.178 | 86.131 | 86.093 | -0.037 | -0.93 |
| pt-en | 46.044 | 45.865 | -0.179 | -0.079 | 87.538 | 87.501 | -0.036 | -1.51 |
| ru-en | 30.466 | 30.417 | -0.049 | 0.005 | 82.151 | 82.114 | -0.037 | -0.20 |
| uk-en | 33.469 | 33.355 | -0.115 | -0.023 | 83.168 | 83.152 | -0.016 | -0.62 |
| zh-en | 24.384 | 25.113 | 0.728 | 0.18 | 81.444 | 81.59 | 0.146 | -10.03 |

Per-pair detail: `results/<pair>/axisA.report.txt`, `axisA.segments.csv` (per-sentence + COMET), `axisB.*.report.txt`.
