"""MLX ASR WER evaluation on Apple Silicon.

Mirrors eval_hf_asr.py but drives inference through the MLX pipeline
(meralion-mlx-2-10b or meralion-mlx-2-3b) instead of HF/CUDA.

Usage:
    # Full eval, 10b model (default)
    python client_eval/eval_mlx_asr.py

    # Use 3b model
    python client_eval/eval_mlx_asr.py --model-size 3b

    # Specific datasets
    python client_eval/eval_mlx_asr.py --datasets ytb_asr_batch1 ytb_asr_batch2

    # Quick smoke test (first 8 samples per dataset)
    python client_eval/eval_mlx_asr.py --max-samples 8

    # Custom model dir
    python client_eval/eval_mlx_asr.py --model-dir /path/to/model-mlx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# text_normalizer lives at repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_normalizer import (
    preprocess_text_asr,
    preprocess_text_asr_code_switch_chinese,
    preprocess_text_asr_malay,
    preprocess_text_asr_tamil,
)

# ---------------------------------------------------------------------------
# Dataset config (same as eval_hf_asr.py, minus cantonese/ste_test3)
# ---------------------------------------------------------------------------
DEFAULT_PRIVATE_DATA_ROOT = REPO_ROOT / "private_data"

DEFAULT_DATASETS = [
    "idpc_short_ASR_v2",
    "ytb_asr_batch1",
    "ytb_asr_batch2",
    "ytb_asr_batch3_chinese",
    "ytb_asr_batch3_malay",
    "ytb_asr_batch3_tamil_v2",
]

DEFAULT_DATASETS_WER = {
    "idpc_short_ASR_v2": 0.16,
    "ytb_asr_batch1":    0.11,
    "ytb_asr_batch2":    0.12,
    "ytb_asr_batch3_chinese":   0.22,
    "ytb_asr_batch3_malay":     0.22,
    "ytb_asr_batch3_tamil_v2":  0.50,
}

_DATASET_NORMALIZER = {
    "idpc_short_ASR_v2":        preprocess_text_asr,
    "ytb_asr_batch1":           preprocess_text_asr,
    "ytb_asr_batch2":           preprocess_text_asr,
    "ytb_asr_batch3_chinese":   preprocess_text_asr_code_switch_chinese,
    "ytb_asr_batch3_malay":     preprocess_text_asr_malay,
    "ytb_asr_batch3_tamil_v2":  preprocess_text_asr_tamil,
}

_DATASET_PATH_OVERRIDES = {
    "ytb_asr_batch3_tamil_v2": "ytb_asr_batch3_tamil_filtered",
}

# Default model dirs relative to repo root
_DEFAULT_MODEL_DIRS = {
    "10b":      "meralion-mlx-2-10b/models/2-10b-mlx",
    "10b-4bit": "meralion-mlx-2-10b/models/2-10b-mlx-4bit",
    "10b-8bit": "meralion-mlx-2-10b/models/2-10b-mlx-8bit",
    "3b":       "meralion-mlx-2-3b/models/2-3b-mlx",
}

# ---------------------------------------------------------------------------
# WER helpers (identical logic to eval_hf_asr.py)
# ---------------------------------------------------------------------------

def _tokenize_for_wer(text: str, dataset_name: str = "") -> list[str]:
    normalizer = _DATASET_NORMALIZER.get(dataset_name)
    if normalizer is None:
        normalized = text.lower()
    else:
        normalized = normalizer(text)
    return normalized.split() if normalized else []


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
    references: list[str], predictions: list[str], dataset_name: str = ""
) -> float:
    total_errors = 0
    total_ref_tokens = 0
    for ref, pred in zip(references, predictions):
        ref_tokens = _tokenize_for_wer(ref, dataset_name)
        pred_tokens = _tokenize_for_wer(pred, dataset_name)
        total_errors += _levenshtein_distance(ref_tokens, pred_tokens)
        total_ref_tokens += len(ref_tokens)
    return total_errors / total_ref_tokens if total_ref_tokens > 0 else 0.0


# ---------------------------------------------------------------------------
# Audio / dataset helpers
# ---------------------------------------------------------------------------

def _extract_audio_and_reference(sample: dict) -> tuple[np.ndarray, str]:
    """Extract 16kHz float32 audio array and reference text from a dataset sample."""
    context = sample["context"]
    answer = sample["answer"]

    audio = context["audio"] if isinstance(context, dict) and "audio" in context else context
    reference = answer["text"] if isinstance(answer, dict) and "text" in answer else answer

    # audio is an HF Audio dict: {"array": ..., "sampling_rate": ...}
    if isinstance(audio, dict) and "array" in audio:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", 16000))
    else:
        raise TypeError(f"Unexpected audio type: {type(audio)}")

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)

    return arr.astype(np.float32), str(reference)


# ---------------------------------------------------------------------------
# MLX model loading and inference
# ---------------------------------------------------------------------------

def load_mlx_model(model_dir: Path, package_dir: Path):
    """Load MLX model components. Returns a LoadedModel namedtuple."""
    # Add the package root to sys.path so meralion_mlx and scripts are importable
    pkg_str = str(package_dir)
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)

    from scripts.inference import load_model
    print(f"Loading MLX model from {model_dir} ...")
    t0 = time.time()
    model = load_model(model_dir, verbose=True)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return model


_CHUNK_SAMPLES = 30 * 16000      # 480000 samples = 30s at 16kHz
_MIN_LAST_CHUNK = 10 * 16000     # 160000 samples = 10s — min size of last chunk
_NO_REPEAT_NGRAM_SIZE = 6        # Match generation_config.json


def _make_no_repeat_ngram_sampler(ngram_size: int):
    """Create a greedy sampler that blocks repeated n-grams.

    Instead of modifying logits (which breaks MLX's async GPU pipeline),
    this sampler does greedy argmax and falls back to the next-best token
    if the top choice would create a repeated n-gram.

    Returns a sampler compatible with mlx-lm's generate_step(sampler=...).
    """
    import mlx.core as mx

    # Map (ngram_size-1) prefix -> set of tokens that followed
    prefix_to_next: dict[tuple[int, ...], set[int]] = {}
    id_list: list[int] = []

    def _register(token: int):
        id_list.append(token)
        if len(id_list) >= ngram_size:
            prefix = tuple(id_list[-ngram_size:-1])
            if prefix not in prefix_to_next:
                prefix_to_next[prefix] = set()
            prefix_to_next[prefix].add(id_list[-1])

    def sampler(logits: mx.array) -> mx.array:
        # Greedy: get top token
        flat = logits.reshape(-1) if logits.ndim == 2 else logits

        token = mx.argmax(flat)
        tid = int(token)

        # Check if this token creates a banned ngram
        if len(id_list) >= ngram_size - 1:
            ctx = tuple(id_list[-(ngram_size - 1):])
            banned = prefix_to_next.get(ctx)
            if banned and tid in banned:
                # Top token is banned — find next best that isn't banned
                sorted_ids = mx.argsort(flat)[::-1]
                for candidate in sorted_ids:
                    cid = int(candidate)
                    if cid not in banned:
                        tid = cid
                        token = candidate
                        break

        _register(tid)
        # Return shape expected by generate_step
        return token.reshape(logits.shape[:-1]) if logits.ndim == 2 else token

    return sampler


def _infer_one_chunk(
    model,
    audio_chunk: np.ndarray,
    max_new_tokens: int,
) -> str:
    """Run ASR inference on an audio chunk.

    Uses prepare_audio(max_duration=None) so chunks slightly longer than 30s
    (from merging a short tail with the previous segment) are handled as
    multi-chunk within a single inference call.
    """
    import mlx.core as mx
    from scripts.inference import build_merged_embeddings
    from meralion_mlx.processor import get_task_prompt
    from mlx_lm.generate import generate_step

    mel_features, num_chunks = model.processor.prepare_audio(
        audio_array=audio_chunk, max_duration=None
    )
    mel_mx = mx.array(mel_features)
    text_inputs = model.processor.prepare_text(
        get_task_prompt("asr"), num_chunks=num_chunks
    )
    input_ids = mx.array(text_inputs["input_ids"])

    chunk_embeds = []
    for i in range(num_chunks):
        enc_out = model.encoder(mel_mx[i:i+1])
        enc_out = model.ln_speech(enc_out)
        chunk_embeds.append(model.adaptor(enc_out))
    speech_embeds = mx.concatenate(chunk_embeds, axis=1)
    mx.eval(speech_embeds)

    merged_emb = build_merged_embeddings(
        model.decoder, input_ids, speech_embeds, model.processor.speech_token_index
    )
    mx.eval(merged_emb)

    eos_tokens = {1, 107}
    generated = []
    ngram_sampler = _make_no_repeat_ngram_sampler(_NO_REPEAT_NGRAM_SIZE)
    for token, _ in generate_step(
        prompt=input_ids[0], model=model.decoder,
        max_tokens=max_new_tokens, sampler=ngram_sampler,
        input_embeddings=merged_emb[0],
    ):
        tid = int(token)
        if tid in eos_tokens:
            break
        generated.append(tid)

    return model.processor.decode(generated)


def infer_one(
    model,
    audio_array: np.ndarray,
    max_new_tokens: int = 512,
) -> str:
    """Run ASR inference on a single audio array.

    Strategy (matches HF eval_hf_asr.py chunking, with short-tail fix):
    1. Split audio at 30s boundaries.
    2. If the last chunk is shorter than 10s, merge it with the preceding
       segment to avoid hallucinations from mostly-silent short inputs.
    3. Process each segment independently and concatenate the outputs.
    Merged segments >30s are handled as multi-chunk by prepare_audio internally.
    """
    if len(audio_array) <= _CHUNK_SAMPLES:
        return _infer_one_chunk(model, audio_array, max_new_tokens)

    # Build 30s-boundary split
    segments = [
        audio_array[start : start + _CHUNK_SAMPLES]
        for start in range(0, len(audio_array), _CHUNK_SAMPLES)
    ]

    # Merge short last segment into the previous one to prevent hallucination
    if len(segments) > 1 and len(segments[-1]) < _MIN_LAST_CHUNK:
        segments = segments[:-2] + [np.concatenate([segments[-2], segments[-1]])]

    parts = [_infer_one_chunk(model, seg, max_new_tokens) for seg in segments]
    return " ".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLX ASR WER evaluation (Apple Silicon)")
    parser.add_argument("--model-size", choices=["10b", "10b-4bit", "10b-8bit", "3b"],
                        default="10b-4bit",
                        help="Which MLX model to use (default: 10b-4bit)")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Override model dir (takes precedence over --model-size)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_PRIVATE_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save per-sample JSONL outputs (one file per dataset)")
    args = parser.parse_args()

    # Resolve model dir and package dir
    if args.model_dir is not None:
        model_dir = args.model_dir
        # Infer package dir from model_dir location (models/X-mlx -> package root)
        package_dir = model_dir.parent.parent
    else:
        rel_path = _DEFAULT_MODEL_DIRS[args.model_size]
        model_dir = REPO_ROOT / rel_path
        # package dir is always the versioned package (strip quant suffix for dir name)
        pkg_name = args.model_size.replace("-4bit", "").replace("-8bit", "")
        package_dir = REPO_ROOT / f"meralion-mlx-2-{pkg_name}"

    if not model_dir.exists():
        print(f"Error: model dir not found: {model_dir}")
        sys.exit(1)

    model = load_mlx_model(model_dir, package_dir)

    # Per-sample output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "eval_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Per-sample outputs: {output_dir}\n")

    results = {}
    all_pass = True

    for dataset_name in args.datasets:
        dir_name = _DATASET_PATH_OVERRIDES.get(dataset_name, dataset_name)
        dataset_path = args.data_root / dir_name
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

        predictions: list[str] = []
        references: list[str] = []

        jsonl_path = output_dir / f"{dataset_name}.jsonl"
        jsonl_fh = open(jsonl_path, "w")

        for i, sample in enumerate(data):
            audio_arr, ref = _extract_audio_and_reference(sample)
            sample_t0 = time.time()
            pred = infer_one(model, audio_arr, max_new_tokens=args.max_new_tokens)
            sample_time = time.time() - sample_t0
            predictions.append(pred)
            references.append(ref)

            # Per-sample WER
            normalizer = _DATASET_NORMALIZER.get(dataset_name)
            norm_ref = normalizer(ref) if normalizer else ref.lower()
            norm_pred = normalizer(pred) if normalizer else pred.lower()
            ref_tokens = norm_ref.split() if norm_ref else []
            pred_tokens = norm_pred.split() if norm_pred else []
            sample_dist = _levenshtein_distance(ref_tokens, pred_tokens)
            sample_wer = sample_dist / len(ref_tokens) if ref_tokens else 0.0

            # Write per-sample result immediately
            record = {
                "idx": i,
                "wer": round(sample_wer, 4),
                "edit_dist": sample_dist,
                "ref_words": len(ref_tokens),
                "pred_words": len(pred_tokens),
                "audio_secs": round(len(audio_arr) / 16000, 1),
                "infer_secs": round(sample_time, 1),
                "reference": ref,
                "reference_normalized": norm_ref,
                "prediction": pred,
                "prediction_normalized": norm_pred,
            }
            jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_fh.flush()

            if (i + 1) % 20 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{n}] {elapsed:.0f}s elapsed")

        jsonl_fh.close()
        print(f"  Saved: {jsonl_path}")

        elapsed = time.time() - t0
        wer = _compute_dataset_wer(references, predictions, dataset_name)
        threshold = DEFAULT_DATASETS_WER.get(dataset_name, 1.0)
        passed = wer <= threshold
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        results[dataset_name] = {"n": n, "wer": wer, "threshold": threshold,
                                 "status": status, "elapsed": elapsed}
        print(f"  WER={wer:.4f} (threshold={threshold:.2f}) [{status}] ({elapsed:.0f}s)\n")

    # Summary table
    print("=" * 65)
    print(f"{'Dataset':<30} {'N':>5} {'WER':>8} {'Thresh':>8} {'Status':>6}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<30} {r['n']:>5} {r['wer']:>8.4f} {r['threshold']:>8.2f} {r['status']:>6}")
    print("=" * 65)
    print(f"Model: {model_dir}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
