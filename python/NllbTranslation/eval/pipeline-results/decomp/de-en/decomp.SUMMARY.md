# Engine vs quantization decomposition — de-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 39.038 | 38.85 | 38.854 | -0.188 | 0.004 |
| chrF | 61.02 | 61.01 | 60.994 | -0.01 | -0.015 |
| chrF++ | 59.609 | 59.582 | 59.563 | -0.027 | -0.019 |
| TER | 48.532 | 48.422 | 48.463 | -0.11 | 0.04 |
| COMET | 85.528 | 85.491 | 85.519 | -0.037 | 0.028 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.23 | 1.0x |
| CT2-fp16 | 4.801 | 2.2x |
| CT2-int8 | 4.444 | 2.0x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
