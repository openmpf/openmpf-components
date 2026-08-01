# Engine vs quantization decomposition — zh-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 24.946 | 25.549 | 25.465 | 0.603 | -0.085 |
| chrF | 49.749 | 50.005 | 49.785 | 0.256 | -0.22 |
| chrF++ | 47.973 | 48.206 | 48.019 | 0.233 | -0.187 |
| TER | 69.589 | 66.487 | 67.122 | -3.102 | 0.635 |
| COMET | 81.512 | 81.639 | 81.629 | 0.127 | -0.01 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.105 | 1.0x |
| CT2-fp16 | 5.079 | 2.4x |
| CT2-int8 | 4.576 | 2.2x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
