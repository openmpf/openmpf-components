#!/usr/bin/env python3
"""Standalone CTranslate2 translation driver — bypasses the OpenMPF component.

The ctranslate2 component ignores the job's NLLB_MODEL (it loads the default
baked model at construction and never reloads by name), so it cannot be used to
evaluate a specific converted model. This driver drives ctranslate2.Translator
directly, with the SAME tokenization the component uses (FLORES SentencePiece:
[src_lang] + pieces + ["</s>"], translate_batch beam-4 with target_prefix
[[tgt_lang]]), so results are faithful — but you choose the model and see the
actual compute_type it loaded.

Emits one translation per input line (+ a meta sidecar with throughput and the
loaded compute_type, so the model actually used is verifiable).

Runs INSIDE the ctranslate2 image (has ctranslate2 + sentencepiece + the SPM).
"""
import argparse
import json
import glob
import os
import sys
import time

import ctranslate2
import sentencepiece as spm

# SentencePiece model locations, newest first. Images built after the CTranslate2
# conversion work copy the tokenizer INTO the converted model directory
# (--copy_files); older images downloaded it separately from OpenNMT. The two
# files are byte-identical (md5 05c551ae7955b3980d5a9d044eb09d70), so either works
# -- this just has to find one.
SP_CANDIDATES = (
    "sentencepiece.bpe.model",  # relative to the model dir (converted --copy_files)
    "/models/*/sentencepiece.bpe.model",  # the image's own shipped model dir
    "/models/OpenNMT/flores200_sacrebleu_tokenizer_spm.model",  # legacy images
)


def resolve_sp_model(explicit, model_dir):
    """Locate the SentencePiece model, or fail with something actionable."""
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"--sp-model not found: {explicit}")
        return explicit
    tried = []
    for cand in SP_CANDIDATES:
        path = cand if os.path.isabs(cand) else os.path.join(model_dir, cand)
        tried.append(path)
        for match in sorted(glob.glob(path)):
            if os.path.isfile(match):
                return match
    sys.exit("No SentencePiece model found. Tried:\n  " + "\n  ".join(tried)
             + "\nPass one explicitly with --sp-model.")


def log(m):
    sys.stderr.write(m + "\n"); sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="CTranslate2 model directory")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-lang", required=True, help="ISO 639-3, e.g. por")
    ap.add_argument("--source-script", default="Latn")
    ap.add_argument("--target-lang", default="eng")
    ap.add_argument("--target-script", default="Latn")
    ap.add_argument("--sp-model", default=None,
                    help="SentencePiece model; default: look inside the model dir, "
                         "then the legacy /models/OpenNMT path")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--compute-type", default="default",
                    help="ctranslate2 compute_type (default = the model's own)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batched", action="store_true",
                    help="one big token-batched translate_batch (throughput); "
                         "default is per-line/batch-1 (single-sentence latency, resumable)")
    ap.add_argument("--max-batch-size", type=int, default=2048, help="tokens, when --batched")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--meta-out", default=None)
    args = ap.parse_args()

    src_code = f"{args.source_lang}_{args.source_script}"
    tgt_code = f"{args.target_lang}_{args.target_script}"

    with open(args.input, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if args.limit:
        lines = lines[: args.limit]
    total = len(lines)

    done = 0
    mode = "w"
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            done = sum(1 for _ in f)
        mode = "a"
        log(f"resuming: {done} lines already translated")

    sp_path = resolve_sp_model(args.sp_model, args.model)
    print(f'tokenizer: {sp_path}', flush=True)
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    log(f"loading model {args.model} (device={args.device}, compute_type={args.compute_type})...")
    t0 = time.time()
    translator = ctranslate2.Translator(args.model, device=args.device,
                                        compute_type=args.compute_type)
    actual_ct = translator.compute_type
    log(f"model ready in {time.time()-t0:.1f}s; actual compute_type = {actual_ct}")
    log(f"src={src_code} tgt={tgt_code} beam={args.beam} batched={args.batched}")

    def encode(text_list):
        pieces = sp.encode_as_pieces(text_list)
        return [[src_code] + p + ["</s>"] for p in pieces]

    def decode_hyp(res):
        toks = [x for x in res.hypotheses[0] if x != tgt_code]
        return sp.decode(toks).replace("\r", " ").replace("\n", " ").strip()

    todo = lines[done:]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t_start = time.time()
    out_chars = 0

    with open(args.output, mode, encoding="utf-8") as out:
        if args.batched:
            res = translator.translate_batch(
                encode(todo), batch_type="tokens", max_batch_size=args.max_batch_size,
                beam_size=args.beam, target_prefix=[[tgt_code]] * len(todo))
            for r in res:
                h = decode_hyp(r); out.write(h + "\n"); out_chars += len(h)
            out.flush()
        else:  # per-line (batch 1) — resumable, single-sentence latency
            for i, text in enumerate(todo):
                r = translator.translate_batch(encode([text]), beam_size=args.beam,
                                               target_prefix=[[tgt_code]])[0]
                h = decode_hyp(r); out.write(h + "\n"); out.flush(); out_chars += len(h)
                n = done + i + 1
                if n % args.progress_every == 0 or n == total:
                    rate = (i + 1) / max(time.time() - t_start, 1e-6)
                    log(f"  {n}/{total}  ({rate:.1f} sent/s)")

    elapsed = time.time() - t_start
    n_run = total - done
    sps = round(n_run / elapsed, 3) if elapsed > 0 else 0
    log(f"done: {total} lines in {elapsed/60:.1f} min ({sps} sent/s), compute_type={actual_ct}")

    if args.meta_out:
        with open(args.meta_out, "w", encoding="utf-8") as f:
            json.dump({
                "driver": "ct2_driver", "model": args.model,
                "requested_compute_type": args.compute_type,
                "actual_compute_type": actual_ct,
                "source": src_code, "target": tgt_code, "beam": args.beam,
                "batched": args.batched, "n_lines": total, "lines_this_run": n_run,
                "elapsed_sec": round(elapsed, 1), "sentences_per_sec": sps,
                "output_chars": out_chars,
                "chars_per_sec": round(out_chars / elapsed, 1) if elapsed > 0 else 0,
            }, f, indent=2)
        log(f"wrote metadata: {args.meta_out}")


if __name__ == "__main__":
    main()
