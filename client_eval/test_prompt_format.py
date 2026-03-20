"""Quick test to compare prompt formats on 8 samples.

Tests:
  A) NEW format (default MLX processor):
     "Given the following audio context: <SpeechHere>\n\nText instruction: {instr}"
  B) OLD format (HF eval_hf_asr.py):
     "Instruction: {instr} \nFollow the text instruction based on the following audio: <SpeechHere>"

Usage:
    python client_eval/test_prompt_format.py --model-size 3b --n 8
    python client_eval/test_prompt_format.py --model-size 10b-4bit --n 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from text_normalizer import preprocess_text_asr

_MODEL_DIRS = {
    "10b":      "meralion-mlx-2-10b/models/2-10b-mlx",
    "10b-4bit": "meralion-mlx-2-10b/models/2-10b-mlx-4bit",
    "3b":       "meralion-mlx-2-3b/models/2-3b-mlx",
}
_PKG_DIRS = {
    "10b":      "meralion-mlx-2-10b",
    "10b-4bit": "meralion-mlx-2-10b",
    "3b":       "meralion-mlx-2-3b",
}

PROMPT_NEW = "Given the following audio context: <SpeechHere>\n\nText instruction: Please transcribe this speech."
PROMPT_OLD = "Instruction: Please transcribe this speech. \nFollow the text instruction based on the following audio: <SpeechHere>"


def _lev(a, b):
    if not a: return len(b)
    if not b: return len(a)
    p = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1):
            c.append(min(p[j]+1, c[-1]+1, p[j-1]+(0 if ca==cb else 1)))
        p = c
    return p[-1]


def wer(refs, preds, normalizer=preprocess_text_asr):
    errs = words = 0
    for r, p in zip(refs, preds):
        rt = normalizer(r).split()
        pt = normalizer(p).split()
        errs += _lev(rt, pt)
        words += len(rt)
    return errs / words if words else 0.0


def run_format(model, processor_cls, model_dir, prompt_text, audio_arrays, SPEECH_TOKEN_IDX):
    """Run inference using a custom prompt text (bypassing processor's default format)."""
    import mlx.core as mx
    from scripts.inference import _infer_segment, build_merged_embeddings
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    speech_token_index = SPEECH_TOKEN_IDX
    conversation = [{"role": "user", "content": prompt_text}]
    chat_text = model.processor.tokenizer.apply_chat_template(
        conversation=[conversation], tokenize=False, add_generation_prompt=True
    )
    if isinstance(chat_text, list):
        chat_text = chat_text[0]

    preds = []
    for audio_arr in audio_arrays:
        mel_features, num_chunks = model.processor.prepare_audio(audio_array=audio_arr, max_duration=None)
        mel_mx = mx.array(mel_features)

        # Tokenize with SPEECH_TOKEN expansion
        encoded = model.processor.tokenizer(chat_text, return_tensors="np")
        input_ids_raw = encoded["input_ids"]

        # expand speech token
        expanded = []
        for tok in input_ids_raw[0]:
            if tok == speech_token_index:
                expanded.extend([speech_token_index] * (100 * num_chunks))
            else:
                expanded.append(int(tok))
        input_ids = mx.array(np.array([expanded], dtype=np.int32))

        # encode speech
        chunk_embeds = []
        for i in range(num_chunks):
            chunk_mel = mel_mx[i:i+1]
            enc_out = model.encoder(chunk_mel)
            enc_out = model.ln_speech(enc_out)
            chunk_speech = model.adaptor(enc_out)
            chunk_embeds.append(chunk_speech)
        speech_embeds = mx.concatenate(chunk_embeds, axis=1)
        mx.eval(speech_embeds)

        merged = build_merged_embeddings(model.decoder, input_ids, speech_embeds, speech_token_index)
        mx.eval(merged)

        prompt_tokens = input_ids[0]
        embeddings_2d = merged[0]
        eos_tokens = {1, 107}
        generated = []
        for token, _ in generate_step(
            prompt=prompt_tokens, model=model.decoder, max_tokens=512,
            sampler=None, input_embeddings=embeddings_2d,
        ):
            tid = int(token)
            if tid in eos_tokens: break
            generated.append(tid)
        pred = model.processor.decode(generated)
        preds.append(pred)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="3b", choices=list(_MODEL_DIRS))
    ap.add_argument("--dataset", default="idpc_short_ASR_v2")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    model_dir = REPO_ROOT / _MODEL_DIRS[args.model_size]
    pkg_dir = REPO_ROOT / _PKG_DIRS[args.model_size]
    pkg_str = str(pkg_dir)
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)

    from scripts.inference import load_model
    print(f"Loading {args.model_size} model ...")
    model = load_model(model_dir, verbose=True)

    data_path = REPO_ROOT / "private_data" / args.dataset
    data = load_from_disk(str(data_path))
    if isinstance(data, dict):
        data = data[next(iter(data))]
    data = data.select(range(min(args.n, len(data))))

    audio_arrays, references = [], []
    for sample in data:
        ctx = sample["context"]
        audio = ctx["audio"] if isinstance(ctx, dict) else ctx
        arr = np.asarray(audio["array"], dtype=np.float32)
        audio_arrays.append(arr)
        ans = sample["answer"]
        references.append(str(ans["text"] if isinstance(ans, dict) else ans))

    SPEECH_IDX = model.processor.speech_token_index

    print(f"\n=== Format A (NEW - MLX default) ===")
    print(f"  {PROMPT_NEW[:80]}")
    t0 = time.time()
    preds_new = run_format(model, None, model_dir, PROMPT_NEW, audio_arrays, SPEECH_IDX)
    wer_new = wer(references, preds_new)
    print(f"  WER={wer_new:.4f}  ({time.time()-t0:.0f}s)")
    for r, p in zip(references[:3], preds_new[:3]):
        print(f"  REF: {r[:80]}")
        print(f"  PRD: {p[:80]}")
        print()

    print(f"\n=== Format B (OLD - HF eval_hf_asr.py) ===")
    print(f"  {PROMPT_OLD[:80]}")
    t0 = time.time()
    preds_old = run_format(model, None, model_dir, PROMPT_OLD, audio_arrays, SPEECH_IDX)
    wer_old = wer(references, preds_old)
    print(f"  WER={wer_old:.4f}  ({time.time()-t0:.0f}s)")
    for r, p in zip(references[:3], preds_old[:3]):
        print(f"  REF: {r[:80]}")
        print(f"  PRD: {p[:80]}")
        print()

    print(f"\nSummary ({args.n} samples, {args.dataset}):")
    print(f"  NEW format WER: {wer_new:.4f}")
    print(f"  OLD format WER: {wer_old:.4f}")
    winner = "NEW" if wer_new < wer_old else "OLD"
    print(f"  Best: {winner} format")


if __name__ == "__main__":
    main()
