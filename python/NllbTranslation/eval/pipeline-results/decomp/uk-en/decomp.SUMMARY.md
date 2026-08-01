# Engine vs quantization decomposition — uk-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 33.562 | 33.459 | 33.522 | -0.102 | 0.063 |
| chrF | 55.523 | 55.448 | 55.504 | -0.075 | 0.056 |
| chrF++ | 54.338 | 54.253 | 54.304 | -0.085 | 0.052 |
| TER | 53.98 | 53.929 | 54.02 | -0.051 | 0.091 |
| COMET | 83.66 | 83.604 | 83.63 | -0.056 | 0.026 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.173 | 1.0x |
| CT2-fp16 | 5.137 | 2.4x |
| CT2-int8 | 4.624 | 2.1x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
