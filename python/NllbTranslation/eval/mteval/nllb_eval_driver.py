#!/usr/bin/env python3
"""Generate aligned NLLB translations, ONE per input line.

Runs INSIDE an OpenMPF NllbTranslation container (develop = facebook fp16, or
ctranslate2 = int8). It instantiates the component once (model loads once),
then feeds each source line as its own single-detection generic feed-forward
job, so the output is exactly 1 translation per input line — the alignment the
whole-file path destroys.

This is the "intrinsic" (Axis A) hypothesis generator: because each OPUS line
is already a single sentence, the component's sentence splitter is a no-op,
which neutralizes the sat-3l-sm vs wtp-bert-mini difference between branches
and isolates the real variable (fp16 vs int8).

Language-pair-agnostic: pass ISO 639-3 language + ISO 15924 script for source
and target.

Invoked (from the host) roughly as:
    docker run --rm --gpus device=0 -v /home/regexer/src/mt-eval:/eval \
        --entrypoint /opt/mpf/plugin-venv/bin/python \
        openmpf_nllb_translation:develop \
        /eval/mteval/nllb_eval_driver.py \
            --input  /eval/eval_data/sample5000.pt \
            --output /eval/results/hyp.develop.en \
            --source-lang por --source-script Latn \
            --target-lang eng --target-script Latn \
            --num-beams 4
"""
import argparse
import json
import os
import sys
import time

import mpf_component_api as mpf
from nllb_component import NllbTranslationComponent


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def build_props(args):
    """Sparse job-properties dict; component fills the rest from its defaults."""
    props = {
        "DEFAULT_SOURCE_LANGUAGE": args.source_lang,
        "DEFAULT_SOURCE_SCRIPT": args.source_script,
        "TARGET_LANGUAGE": args.target_lang,
        "TARGET_SCRIPT": args.target_script,
    }
    for kv in args.prop or []:
        if "=" not in kv:
            raise SystemExit(f"--prop expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        props[k] = v
    return props


def apply_beam(component, num_beams):
    """Harmonize decoding across engines.

    HF (facebook) model exposes generation_config; the default ships without
    num_beams (=> greedy). Setting it here makes fp16 use beam search to match
    the int8 branch (which hardcodes beam_size=4). The ctranslate2 Translator
    has no generation_config, so it is left at its built-in beam_size.
    """
    if not num_beams:
        return "unchanged (no --num-beams)"
    model = getattr(component, "_model", None)
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.num_beams = num_beams
        return f"HF generation_config.num_beams = {num_beams}"
    return (f"ctranslate2 backend: generation_config absent; relying on the "
            f"component's built-in beam_size (expected {num_beams})")


def translate_one(component, text, props, data_uri):
    ff_track = mpf.GenericTrack(-1, dict(TEXT=text))
    job = mpf.GenericJob("eval", data_uri, props, {}, ff_track)
    tracks = component.get_detections_from_generic(job)
    out = tracks[0].detection_properties.get("TRANSLATION", "")
    # Keep strictly one physical line per input line.
    return out.replace("\r", " ").replace("\n", " ").strip()


def _clean(s):
    return s.replace("\r", " ").replace("\n", " ").strip()


class HFBatcher:
    """Faithful batched decode for the HF (transformers) backend.

    Reproduces the component's per-sentence generate() call — same
    forced_bos_token_id, same max_length, same tokenizer (with src_lang) and
    generation_config (beam) — but runs many sentences per generate() call.
    Only sentences under a conservative token cap are batched here; longer ones
    (which the component would sentence-split) are routed back through the
    component so its splitting behavior stays exact. Validated to match the
    per-sentence path before use.
    """

    def __init__(self, component, props, token_cap):
        import torch
        from nllb_component import JobConfig
        from nllb_component.nllb_translation_component import should_translate
        self.torch = torch
        self.should_translate = should_translate
        self.component = component
        self.model = component._model
        self.tok = component._tokenizer          # loaded by the warmup call
        cfg = JobConfig(props, {})
        self.max_length = cfg.nllb_token_limit
        self.forced_bos = self.tok.encode(cfg.translate_to_language)[1]
        self.token_cap = token_cap

    def n_tokens(self, text):
        return len(self.tok(text)["input_ids"])

    def is_simple(self, text):
        """True if the component would NOT sentence-split this (single generate)."""
        return self.n_tokens(text) <= self.token_cap

    def translate_batch(self, sentences):
        enc = self.tok(sentences, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            gen = self.model.generate(**enc, forced_bos_token_id=self.forced_bos,
                                      max_length=self.max_length)
        return [_clean(t) for t in self.tok.batch_decode(gen, skip_special_tokens=True)]


def _progress(written, start, total, t_start):
    rate = (written - start) / max(time.time() - t_start, 1e-6)
    eta = (total - written) / max(rate, 1e-6)
    log(f"  {written}/{total}  ({rate:.1f} lines/s, ETA {eta/60:.1f} min)")


def run_per_line(component, lines, done, total, props, args, out, t_start):
    chars = 0
    for i in range(done, total):
        t = translate_one(component, lines[i], props, args.input)
        out.write(t + "\n")
        out.flush()
        chars += len(t)
        n = i + 1
        if n % args.progress_every == 0 or n == total:
            _progress(n, done, total, t_start)
    return chars


def run_batched(component, lines, done, total, props, args, out, t_start):
    if done >= total:
        return 0
    written = done
    chars = [0]

    # Warm up on the first pending line via the component: this loads the
    # tokenizer with the correct src_lang and gives a faithful first output.
    _warm = translate_one(component, lines[done], props, args.input)
    out.write(_warm + "\n")
    out.flush()
    written += 1
    chars[0] += len(_warm)

    batcher = HFBatcher(component, props, args.batch_token_cap)
    buf = []
    n_flushes = [0]

    def flush():
        nonlocal written
        if not buf:
            return
        for t in batcher.translate_batch(buf):
            out.write(t + "\n")
            written += 1
            chars[0] += len(t)
        buf.clear()
        out.flush()
        n_flushes[0] += 1
        if args.gpu_empty_cache_every and n_flushes[0] % args.gpu_empty_cache_every == 0:
            batcher.torch.cuda.empty_cache()
        _progress(written, done, total, t_start)

    for i in range(done + 1, total):
        line = lines[i]
        if not batcher.should_translate(line):
            flush()
            _t = _clean(line)
            out.write(_t + "\n"); out.flush(); written += 1; chars[0] += len(_t)
        elif batcher.is_simple(line):
            buf.append(line)
            if len(buf) >= args.batch:
                flush()
        else:  # long sentence: let the component sentence-split it exactly
            flush()
            _t = translate_one(component, line, props, args.input)
            out.write(_t + "\n"); out.flush(); written += 1; chars[0] += len(_t)
    flush()
    if written != total:
        _progress(written, done, total, t_start)
    return chars[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source file, one sentence per line")
    ap.add_argument("--output", required=True, help="translations out, one per line")
    ap.add_argument("--source-lang", required=True, help="ISO 639-3, e.g. por")
    ap.add_argument("--source-script", default="Latn", help="ISO 15924, e.g. Latn")
    ap.add_argument("--target-lang", default="eng")
    ap.add_argument("--target-script", default="Latn")
    ap.add_argument("--num-beams", type=int, default=None,
                    help="harmonize HF decoding to this beam size (int8 is already 4)")
    ap.add_argument("--prop", action="append",
                    help="extra job property KEY=VALUE (repeatable)")
    ap.add_argument("--batch", type=int, default=None,
                    help="HF backend only: batch this many sentences per generate() "
                         "call (big fp16 speedup). Ignored for the ctranslate2 backend.")
    ap.add_argument("--batch-token-cap", type=int, default=120,
                    help="only batch sentences with <= this many tokens; longer ones "
                         "route through the component (which sentence-splits them)")
    ap.add_argument("--gpu-empty-cache-every", type=int, default=0,
                    help="call torch.cuda.empty_cache() every N batches to fight "
                         "fragmentation on long runs (0 = never)")
    ap.add_argument("--limit", type=int, default=None, help="only first N lines (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="skip lines already present in --output")
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--meta-out", default=None, help="write a JSON run-metadata sidecar")
    args = ap.parse_args()

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

    props = build_props(args)
    log(f"loading component (this loads the model once)...")
    t0 = time.time()
    component = NllbTranslationComponent()
    log(f"component ready in {time.time() - t0:.1f}s")
    log("beam: " + apply_beam(component, args.num_beams))
    log(f"props: {json.dumps(props, ensure_ascii=False)}")

    use_batch = bool(args.batch) and getattr(
        getattr(component, "_model", None), "generation_config", None) is not None
    if args.batch and not use_batch:
        log("note: --batch requested but backend has no generation_config "
            "(ctranslate2?); using per-line path.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t_start = time.time()
    with open(args.output, mode, encoding="utf-8") as out:
        if use_batch:
            log(f"batched HF decode: batch={args.batch}, token_cap={args.batch_token_cap}")
            out_chars = run_batched(component, lines, done, total, props, args, out, t_start)
        else:
            out_chars = run_per_line(component, lines, done, total, props, args, out, t_start)

    elapsed = time.time() - t_start
    lines_this_run = total - done
    sps = round(lines_this_run / elapsed, 3) if elapsed > 0 else 0
    cps = round((out_chars or 0) / elapsed, 1) if elapsed > 0 else 0
    log(f"done: {total} lines in {elapsed/60:.1f} min "
        f"({sps} sent/s, {cps} out-chars/s)")

    if args.meta_out:
        meta = {
            "input": args.input,
            "output": args.output,
            "source_lang": args.source_lang,
            "source_script": args.source_script,
            "target_lang": args.target_lang,
            "target_script": args.target_script,
            "num_beams_requested": args.num_beams,
            "job_properties": props,
            "n_lines": total,
            "lines_this_run": lines_this_run,
            "elapsed_sec": round(elapsed, 1),
            "sentences_per_sec": sps,
            "output_chars": out_chars,
            "chars_per_sec": cps,
            "batch": args.batch,
        }
        with open(args.meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log(f"wrote metadata: {args.meta_out}")


if __name__ == "__main__":
    main()
