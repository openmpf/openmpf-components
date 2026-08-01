# Engine vs quantization decomposition — ru-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 30.325 | 30.475 | 30.338 | 0.15 | -0.138 |
| chrF | 53.873 | 53.986 | 53.879 | 0.113 | -0.107 |
| chrF++ | 52.324 | 52.425 | 52.363 | 0.1 | -0.062 |
| TER | 57.321 | 57.123 | 56.971 | -0.198 | -0.152 |
| COMET | 82.329 | 82.309 | 82.269 | -0.02 | -0.04 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.252 | 1.0x |
| CT2-fp16 | 5.386 | 2.4x |
| CT2-int8 | 4.498 | 2.0x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
