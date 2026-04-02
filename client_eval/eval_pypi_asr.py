"""ASR WER evaluation using the mlx-meralion PyPI package.

Tests that `pip install mlx-meralion` produces correct ASR with n-gram blocking.

Usage:
    python client_eval/eval_pypi_asr.py --model MERaLiON/MERaLiON-2-3B-MLX \
        --datasets ytb_asr_batch1 ytb_asr_batch3_chinese

    python client_eval/eval_pypi_asr.py --model MERaLiON/MERaLiON-2-10B-MLX \
        --datasets ytb_asr_batch1 ytb_asr_batch3_chinese
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

# text_normalizer lives at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_normalizer import (
    preprocess_text_asr,
    preprocess_text_asr_code_switch_chinese,
    preprocess_text_asr_malay,
    preprocess_text_asr_tamil,
)

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
DEFAULT_PRIVATE_DATA_ROOT = REPO_ROOT / "private_data"

DEFAULT_DATASETS = [
    "ytb_asr_batch1",
    "ytb_asr_batch3_chinese",
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

# ---------------------------------------------------------------------------
# WER helpers
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
# Audio helpers
# ---------------------------------------------------------------------------

def _extract_audio_and_reference(sample: dict) -> tuple[np.ndarray, str]:
    context = sample["context"]
    answer = sample["answer"]

    audio = context["audio"] if isinstance(context, dict) and "audio" in context else context
    reference = answer["text"] if isinstance(answer, dict) and "text" in answer else answer

    if isinstance(audio, dict) and "array" in audio:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", 16000))
    else:
        raise TypeError(f"Unexpected audio type: {type(audio)}")

    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)

    return arr.astype(np.float32), str(reference)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASR WER evaluation using mlx-meralion PyPI package"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace repo ID (e.g. MERaLiON/MERaLiON-2-3B-MLX)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_PRIVATE_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    # Import from PyPI package
    from mlx_meralion import load_model, transcribe

    print(f"Loading model: {args.model}")
    t0 = time.time()
    model = load_model(args.model, verbose=True)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "eval_outputs_pypi"
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

            # Use PyPI package's transcribe() — has n-gram blocking + smart chunking
            pred = transcribe(
                model, audio_arr,
                task="asr",
                max_new_tokens=args.max_new_tokens,
                verbose=False,
            )

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
    print(f"Model: {args.model}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
