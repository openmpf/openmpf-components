# Engine vs quantization decomposition — ar-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 38.288 | 40.198 | 40.37 | 1.91 | 0.172 |
| chrF | 60.462 | 61.471 | 61.607 | 1.01 | 0.136 |
| chrF++ | 59.179 | 60.267 | 60.403 | 1.089 | 0.136 |
| TER | 51.417 | 48.314 | 48.211 | -3.103 | -0.103 |
| COMET | 83.756 | 84.471 | 84.47 | 0.715 | -0.0 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 1.226 | 1.0x |
| CT2-fp16 | 5.023 | 4.1x |
| CT2-int8 | 4.649 | 3.8x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
