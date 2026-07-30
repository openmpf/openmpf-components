# Engine vs quantization decomposition — pt-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 45.062 | 44.923 | 44.896 | -0.139 | -0.027 |
| chrF | 66.319 | 66.262 | 66.278 | -0.057 | 0.016 |
| chrF++ | 65.01 | 64.942 | 64.952 | -0.068 | 0.009 |
| TER |  |  |  |  |  |
| COMET | 87.348 | 87.375 | 87.366 | 0.027 | -0.008 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.057 | 1.0x |
| CT2-fp16 | 4.998 | 2.4x |
| CT2-int8 | 4.589 | 2.2x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
