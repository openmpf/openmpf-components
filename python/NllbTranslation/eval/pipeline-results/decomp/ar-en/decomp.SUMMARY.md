# Engine vs quantization decomposition — ar-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 40.236 | 40.198 | 40.37 | -0.038 | 0.172 |
| chrF | 61.479 | 61.471 | 61.607 | -0.008 | 0.136 |
| chrF++ | 60.275 | 60.267 | 60.403 | -0.007 | 0.136 |
| TER | 48.44 | 48.314 | 48.211 | -0.126 | -0.103 |
| COMET | 84.407 | 84.471 | 84.47 | 0.064 | -0.0 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.089 | 1.0x |
| CT2-fp16 | 4.977 | 2.4x |
| CT2-int8 | 4.253 | 2.0x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
