# MT Evaluation — combined summary (HF vs CT2)

Δ = CT2 − HF. **CT2** = the CTranslate2-branch image; its precision follows that image's `BUILD_TYPE` (gpu → float16, cpu → int8_float32) and is **not** necessarily int8. **HF** = the develop/Transformers image (facebook/nllb-200-3.3B, fp16). Earlier runs labelled these fp16/int8.

Axis A = intrinsic per-sentence (beam 4 both, splitter neutralized). Axis B = as-deployed document blob.

| Pair | BLEU HF | BLEU CT2 | ΔBLEU | ΔchrF | COMET HF | COMET CT2 | ΔCOMET | AxisB ΔBLEU |
|------|---------|----------|-------|-------|----------|-----------|--------|-------------|
| ar-en | 40.709 | 40.458 | -0.251 | -0.116 | 84.88  | 84.845 | -0.034 | +2.61 |
| bn-en | 33.243 | 33.269 |  0.026 |  0.062 | 85.755 | 85.832 |  0.077 | +14.03 |
| de-en | 39.037 | 38.879 | -0.158 | -0.057 | 85.939 | 85.928 | -0.011 | +1.91 |
| fa-en | 36.17  | 36.125 | -0.045 | -0.008 | 84.903 | 84.942 |  0.039 | +10.60 |
| fr-en | 42.614 | 42.586 | -0.028 |  0.006 | 86.131 | 86.137 |  0.007 | +1.23 |
| pt-en | 46.044 | 45.982 | -0.062 | -0.028 | 87.538 | 87.54  |  0.003 | +1.39 |
| ru-en | 30.466 | 30.448 | -0.018 | -0.009 | 82.151 | 82.136 | -0.015 | +2.34 |
| uk-en | 33.469 | 33.351 | -0.119 | -0.05  | 83.168 | 83.152 | -0.016 | +2.23 |
| zh-en | 24.384 | 25.094 |  0.709 |  0.23  | 81.444 | 81.561 |  0.117 | +8.85 |

Per-pair detail: `results/<pair>/axisA.report.txt`, `axisA.segments.csv` (per-sentence + COMET), `axisB.*.report.txt`.
