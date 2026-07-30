# Engine vs quantization decomposition — fr-en

Systems: **HF-fp16** (Transformers), **CT2-fp16** (CTranslate2, float16), **CT2-int8** (CTranslate2, int8) — all beam 4, per-sentence.
Δengine = CT2-fp16 − HF-fp16 (same precision, different engine). Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine | Δquant |
|--------|---------|----------|----------|---------|--------|
| BLEU | 42.793 | 42.529 | 42.348 | -0.264 | -0.181 |
| chrF | 63.704 | 63.541 | 63.438 | -0.163 | -0.103 |
| chrF++ | 62.471 | 62.306 | 62.212 | -0.164 | -0.095 |
| TER | 44.206 | 44.058 | 44.107 | -0.149 | 0.05 |
| COMET | 86.185 | 86.206 | 86.183 | 0.021 | -0.023 |

### Throughput (sentences/sec, per-line / batch 1)

| System | sent/s | speedup vs HF-fp16 |
|--------|--------|--------------------|
| HF-fp16 | 2.061 | 1.0x |
| CT2-fp16 | 4.945 | 2.4x |
| CT2-int8 | 4.75 | 2.3x |

_Throughput here is single-sentence latency (batch 1); batched throughput widens the CTranslate2 lead further. Significance (paired bootstrap) is in `decomp.engine.report.txt` / `decomp.quant.report.txt`._
