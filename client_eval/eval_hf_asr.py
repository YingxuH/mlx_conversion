"""ASR WER evaluation using HuggingFace inference (no flash-attn, no vLLM).

Reuses datasets, normalizers, and WER thresholds from
tests/test_asr_transcription_eval.py but runs inference via
transformers AutoModelForSpeechSeq2Seq instead of a vLLM OpenAI server.

Usage:
    # Full eval, all datasets
    CUDA_VISIBLE_DEVICES=1 python client_eval/eval_hf_asr.py

    # Quick smoke test (16 samples per dataset)
    CUDA_VISIBLE_DEVICES=1 python client_eval/eval_hf_asr.py --max-samples 16

    # Specific datasets
    CUDA_VISIBLE_DEVICES=1 python client_eval/eval_hf_asr.py --datasets ytb_asr_batch1 ytb_asr_batch2

    # Custom attn backend
    CUDA_VISIBLE_DEVICES=1 python client_eval/eval_hf_asr.py --attn-implementation eager
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# ---------------------------------------------------------------------------
# Add tests/ to sys.path so we can reuse the Audiobench normalizers
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from text_normalizer import (
    preprocess_text_asr,
    preprocess_text_asr_code_switch_chinese,
    preprocess_text_asr_malay,
    preprocess_text_asr_tamil,
)

# ---------------------------------------------------------------------------
# Constants (mirrored from test_asr_transcription_eval.py)
# ---------------------------------------------------------------------------
DEFAULT_PRIVATE_DATA_ROOT = "/home/yingxu/private_data"
DEFAULT_MODEL_PATH = "/home/yingxu/workspace/MERaLiON_local/MERaLiON-2-10B"
ASR_CHUNK_SECONDS = 30.0

PROMPT_TEXT = "Please transcribe this speech."
PROMPT_TEMPLATE = (
    "Instruction: {text_input} \n"
    "Follow the text instruction based on the following audio: <SpeechHere>"
)

DEFAULT_DATASETS = [
    "idpc_short_ASR_v2",
    "ste_test3",
    "ytb_asr_batch1",
    "ytb_asr_batch2",
    "ytb_asr_batch3_chinese",
    "ytb_asr_batch3_malay",
    "ytb_asr_batch3_tamil_v2",
]

DEFAULT_DATASETS_WER = {
    "idpc_short_ASR_v2": 0.16,
    "ste_test3": 0.15,
    "ytb_asr_batch1": 0.11,
    "ytb_asr_batch2": 0.12,
    "ytb_asr_batch3_chinese": 0.17,
    "ytb_asr_batch3_malay": 0.18,
    "ytb_asr_batch3_tamil_v2": 0.35,
}

_DATASET_NORMALIZER = {
    "idpc_short_ASR_v2": preprocess_text_asr,
    "ste_test3": preprocess_text_asr,
    "ytb_asr_batch1": preprocess_text_asr,
    "ytb_asr_batch2": preprocess_text_asr,
    "ytb_asr_batch3_chinese": preprocess_text_asr_code_switch_chinese,
    "ytb_asr_batch3_malay": preprocess_text_asr_malay,
    "ytb_asr_batch3_tamil_v2": preprocess_text_asr_tamil,
}

_DATASET_PATH_OVERRIDES = {
    "ytb_asr_batch3_tamil_v2": "ytb_asr_batch3_tamil_filtered",
}

# ---------------------------------------------------------------------------
# WER helpers (copied from test_asr_transcription_eval.py)
# ---------------------------------------------------------------------------

def _normalize_text_fallback(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\[|\(|\{|\<)[^\(\)\[\]\{\}\<\>]*(\]|\)|\}|\>)", " ", text)
    text = re.sub(r"[^\w\s\u4e00-\u9fff\u0E00-\u0E7F\u0B80-\u0BFF]", " ", text)
    text = re.sub(r"\b(uh|umm|um|er|ah)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_for_wer(text: str, dataset_name: str = "") -> list[str]:
    normalizer = _DATASET_NORMALIZER.get(dataset_name, _normalize_text_fallback)
    normalized = normalizer(text)
    if not normalized:
        return []
    return normalized.split()


def _levenshtein_distance(reference: list[str], prediction: list[str]) -> int:
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)
    prev = list(range(len(prediction) + 1))
    for i, ref_tok in enumerate(reference, start=1):
        curr = [i]
        for j, pred_tok in enumerate(prediction, start=1):
            sub_cost = 0 if ref_tok == pred_tok else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + sub_cost))
        prev = curr
    return prev[-1]


def _compute_dataset_wer(
    references: Iterable[str], predictions: Iterable[str], dataset_name: str = ""
) -> float:
    total_errors = 0
    total_ref_tokens = 0
    for ref, pred in zip(references, predictions):
        ref_tokens = _tokenize_for_wer(ref, dataset_name)
        pred_tokens = _tokenize_for_wer(pred, dataset_name)
        total_errors += _levenshtein_distance(ref_tokens, pred_tokens)
        total_ref_tokens += len(ref_tokens)
    if total_ref_tokens == 0:
        return 0.0
    return total_errors / total_ref_tokens


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio_array(audio) -> tuple[np.ndarray, int]:
    """Load audio into mono float32 array + sample_rate."""
    if isinstance(audio, str):
        array, sr = sf.read(str(audio), dtype="float32")
    elif isinstance(audio, dict) and "array" in audio:
        array = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", 16000))
    elif isinstance(audio, dict) and "path" in audio:
        array, sr = sf.read(str(audio["path"]), dtype="float32")
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)!r}")
    if array.ndim > 1:
        array = np.mean(array, axis=1, dtype=np.float32)
    return np.asarray(array, dtype=np.float32), int(sr)


def _chunk_audio(audio, chunk_seconds: float = ASR_CHUNK_SECONDS) -> list[np.ndarray]:
    """Split long audio into fixed-duration chunks, return list of arrays at 16kHz."""
    array, sr = _load_audio_array(audio)
    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
    chunk_samples = max(1, int(chunk_seconds * 16000))
    if len(array) <= chunk_samples:
        return [array]
    return [array[start:start + chunk_samples] for start in range(0, len(array), chunk_samples)]


def _extract_audio_and_reference(sample: dict) -> tuple:
    context = sample["context"]
    answer = sample["answer"]
    if isinstance(context, dict) and "audio" in context:
        audio = context["audio"]
    else:
        audio = context
    if isinstance(answer, dict) and "text" in answer:
        reference = answer["text"]
    else:
        reference = answer
    return audio, str(reference)


# ---------------------------------------------------------------------------
# HuggingFace batch inference
# ---------------------------------------------------------------------------

def transcribe_batch(
    model,
    processor,
    audio_arrays: list[np.ndarray],
    device: str,
    max_new_tokens: int = 1024,
) -> list[str]:
    """Transcribe a batch of audio arrays using HF model.generate()."""
    model_dtype = next(model.parameters()).dtype

    prompt = PROMPT_TEMPLATE.format(text_input=PROMPT_TEXT)
    conversation = [[{"role": "user", "content": prompt}]] * len(audio_arrays)
    chat_prompt = processor.tokenizer.apply_chat_template(
        conversation=conversation, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=chat_prompt, audios=audio_arrays)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
            if value.dtype == torch.float32 and model_dtype != torch.float32:
                inputs[key] = inputs[key].to(model_dtype)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = outputs[:, inputs["input_ids"].size(1):]
    responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [r.strip() for r in responses]


def transcribe_dataset(
    model,
    processor,
    samples: list[dict],
    device: str,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    """Transcribe all samples, chunking long audio and batching inference."""
    # Flatten: each sample may produce multiple chunks
    all_chunks = []       # (sample_idx, chunk_array)
    references = []

    for i, sample in enumerate(samples):
        audio, ref = _extract_audio_and_reference(sample)
        chunks = _chunk_audio(audio)
        for chunk in chunks:
            all_chunks.append((i, chunk))
        references.append(ref)

    # Run inference in batches
    chunk_predictions = [""] * len(all_chunks)
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_chunks))
        batch_arrays = [all_chunks[j][1] for j in range(start, end)]

        preds = transcribe_batch(model, processor, batch_arrays, device)
        for j, pred in enumerate(preds):
            chunk_predictions[start + j] = pred

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  batch {batch_idx + 1}/{total_batches}")

    # Reassemble: join chunk predictions per sample
    sample_predictions = [""] * len(samples)
    for chunk_idx, (sample_idx, _) in enumerate(all_chunks):
        if sample_predictions[sample_idx]:
            sample_predictions[sample_idx] += " " + chunk_predictions[chunk_idx]
        else:
            sample_predictions[sample_idx] = chunk_predictions[chunk_idx]

    return [p.strip() for p in sample_predictions], references


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HF ASR WER evaluation (no flash-attn)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=DEFAULT_PRIVATE_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto-detect from GPU memory")
    parser.add_argument("--attn-implementation", default=None,
                        choices=["sdpa", "eager", "flash_attention_2"],
                        help="Attention backend (default: auto-detect)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (default: bfloat16)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect batch size from free GPU memory
    if args.batch_size == 0:
        if device == "cuda":
            free_mem_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            # bf16/fp16: model ~20GB, fp32: model ~40GB
            model_gb = 40 if args.dtype == "float32" else 20
            overhead_gb = model_gb + 5
            per_sample_gb = 4 if args.dtype == "float32" else 2
            available_for_batch = free_mem_gb - overhead_gb
            args.batch_size = max(1, min(8, int(available_for_batch / per_sample_gb)))
            print(f"GPU free memory: {free_mem_gb:.1f} GB, model ~{model_gb} GB -> auto batch_size={args.batch_size}")
        else:
            args.batch_size = 1

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.model_path}")
    print(f"  attn_implementation={args.attn_implementation}, dtype={args.dtype}, batch_size={args.batch_size}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    load_kwargs = dict(
        use_safetensors=True,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path, **load_kwargs
    ).to(device)
    model.eval()

    print(f"Model loaded. attn_implementation={model.config._attn_implementation}\n")

    results = {}
    all_pass = True

    for dataset_name in args.datasets:
        dir_name = _DATASET_PATH_OVERRIDES.get(dataset_name, dataset_name)
        dataset_path = Path(args.data_root) / dir_name
        if not dataset_path.exists():
            print(f"SKIP {dataset_name}: path not found ({dataset_path})")
            continue

        raw_data = load_from_disk(str(dataset_path))
        if isinstance(raw_data, dict):
            split_name = "test" if "test" in raw_data else next(iter(raw_data.keys()))
            data = raw_data[split_name]
        else:
            data = raw_data
        assert isinstance(data, Dataset)

        if args.max_samples > 0 and len(data) > args.max_samples:
            data = data.select(range(args.max_samples))

        n = len(data)
        print(f"--- {dataset_name} (n={n}) ---")
        t0 = time.time()

        predictions, references = transcribe_dataset(
            model, processor, list(data), device, args.batch_size
        )
        elapsed = time.time() - t0

        wer = _compute_dataset_wer(references, predictions, dataset_name)
        threshold = DEFAULT_DATASETS_WER.get(dataset_name, 1.0)
        passed = wer < threshold
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        results[dataset_name] = {
            "n": n,
            "wer": wer,
            "threshold": threshold,
            "status": status,
            "elapsed": elapsed,
        }
        print(f"  WER={wer:.4f} (threshold={threshold:.2f}) [{status}] ({elapsed:.1f}s)\n")

    # Summary
    print("=" * 60)
    print(f"{'Dataset':<30} {'N':>5} {'WER':>8} {'Thresh':>8} {'Status':>6}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<30} {r['n']:>5} {r['wer']:>8.4f} {r['threshold']:>8.2f} {r['status']:>6}")
    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"attn_implementation={args.attn_implementation}, dtype={args.dtype}, batch_size={args.batch_size}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
