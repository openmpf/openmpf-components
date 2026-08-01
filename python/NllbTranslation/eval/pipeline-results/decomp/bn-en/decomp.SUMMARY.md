# Engine vs quantization decomposition — bn-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 33.356 | 33.547 | 33.619 | 0.191 | 0.072 |
| chrF | 55.365 | 55.511 | 55.539 | 0.146 | 0.027 |
| chrF++ | 54.111 | 54.282 | 54.329 | 0.171 | 0.047 |
| TER | 56.934 | 56.105 | 55.971 | -0.829 | -0.134 |
| COMET | 85.745 | 85.911 | 85.873 | 0.166 | -0.038 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.206 | 1.0x |
| CT2-fp16 | 5.179 | 2.3x |
| CT2-int8 | 4.574 | 2.1x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
