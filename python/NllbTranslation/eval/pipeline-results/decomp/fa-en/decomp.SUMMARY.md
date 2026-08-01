# Engine vs quantization decomposition — fa-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 36.002 | 36.109 | 36.058 | 0.107 | -0.051 |
| chrF | 58.315 | 58.426 | 58.261 | 0.11 | -0.164 |
| chrF++ | 57.129 | 57.252 | 57.086 | 0.124 | -0.166 |
| TER | 53.099 | 52.329 | 52.422 | -0.77 | 0.093 |
| COMET | 84.906 | 84.949 | 85.015 | 0.043 | 0.066 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.128 | 1.0x |
| CT2-fp16 | 5.042 | 2.4x |
| CT2-int8 | 4.569 | 2.1x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
