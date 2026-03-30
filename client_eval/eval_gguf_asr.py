"""GGUF ASR WER evaluation via llama.cpp (llama-mtmd-cli).

Usage:
    python client_eval/eval_gguf_asr.py \
        --decoder /tmp/meralion-decoder-q8_0.gguf \
        --mmproj /tmp/meralion-mmproj-f16.gguf

    # Quick smoke test
    python client_eval/eval_gguf_asr.py --max-samples 8 \
        --decoder /tmp/meralion-decoder-q8_0.gguf \
        --mmproj /tmp/meralion-mmproj-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_normalizer import (
    preprocess_text_asr,
    preprocess_text_asr_code_switch_chinese,
    preprocess_text_asr_malay,
    preprocess_text_asr_tamil,
)

# Same dataset config as eval_mlx_asr.py
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

LLAMA_CPP_DIR = Path("/Users/heyx/Documents/llama.cpp")
MTMD_CLI = LLAMA_CPP_DIR / "build/bin/llama-mtmd-cli"

PROMPT_TEMPLATE = (
    "Instruction: Please transcribe this speech. \n"
    "Follow the text instruction based on the following audio: <__media__>"
)


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


_CHUNK_SAMPLES = 30 * 16000      # 480000 samples = 30s at 16kHz
_MIN_LAST_CHUNK = 10 * 16000     # 160000 samples = 10s — min size of last chunk


def _infer_chunk_gguf(
    decoder_path: Path,
    mmproj_path: Path,
    audio_chunk: np.ndarray,
    max_tokens: int = 512,
    ngl: int = 99,
) -> str:
    """Run inference on a single audio chunk via llama-mtmd-cli."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_chunk, 16000)
        wav_path = f.name

    try:
        result = subprocess.run(
            [
                str(MTMD_CLI),
                "-m", str(decoder_path),
                "--mmproj", str(mmproj_path),
                "--audio", wav_path,
                "-p", PROMPT_TEMPLATE,
                "-ngl", str(ngl),
                "-n", str(max_tokens),
                "--no-warmup",
                "--no-perf",
                # DRY sampling to prevent n-gram repetition
                "--dry-multiplier", "1.0",
                "--dry-allowed-length", "1",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        lines = result.stdout.strip().split("\n")
        text_lines = []
        for line in lines:
            if any(line.startswith(p) for p in [
                "WARN:", "LOG:", "main:", "llama_", "ggml_", "clip_",
                "load_", "warmup:", "encoding", "decoding", "audio",
            ]):
                continue
            text_lines.append(line)

        return "\n".join(text_lines).strip()
    except subprocess.TimeoutExpired:
        return ""
    finally:
        Path(wav_path).unlink(missing_ok=True)


def infer_one_gguf(
    decoder_path: Path,
    mmproj_path: Path,
    audio_array: np.ndarray,
    max_tokens: int = 512,
    ngl: int = 99,
) -> str:
    """Run ASR inference with smart chunking (matching MLX/HF eval strategy).

    Strategy:
    1. Audio ≤30s: single chunk
    2. Audio >30s: split at 30s boundaries, merge last if <10s,
       process each independently, concatenate text
    """
    if len(audio_array) <= _CHUNK_SAMPLES:
        return _infer_chunk_gguf(decoder_path, mmproj_path, audio_array, max_tokens, ngl)

    # Split at 30s boundaries
    segments = [
        audio_array[start : start + _CHUNK_SAMPLES]
        for start in range(0, len(audio_array), _CHUNK_SAMPLES)
    ]

    # Merge short last segment into previous to prevent hallucination
    if len(segments) > 1 and len(segments[-1]) < _MIN_LAST_CHUNK:
        segments = segments[:-2] + [np.concatenate([segments[-2], segments[-1]])]

    parts = [_infer_chunk_gguf(decoder_path, mmproj_path, seg, max_tokens, ngl) for seg in segments]
    return " ".join(p.strip() for p in parts if p.strip())


def main():
    parser = argparse.ArgumentParser(description="GGUF ASR WER evaluation via llama.cpp")
    parser.add_argument("--decoder", type=Path, required=True, help="Path to decoder GGUF")
    parser.add_argument("--mmproj", type=Path, required=True, help="Path to mmproj GGUF")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_PRIVATE_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or REPO_ROOT / "eval_outputs_gguf"
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
        total_errors = 0
        total_ref_tokens = 0

        jsonl_path = output_dir / f"{dataset_name}.jsonl"
        jsonl_fh = open(jsonl_path, "w")

        for i, sample in enumerate(data):
            audio_arr, ref = _extract_audio_and_reference(sample)
            sample_t0 = time.time()
            pred = infer_one_gguf(
                args.decoder, args.mmproj, audio_arr,
                max_tokens=args.max_tokens,
            )
            sample_time = time.time() - sample_t0
            predictions.append(pred)
            references.append(ref)

            normalizer = _DATASET_NORMALIZER.get(dataset_name)
            norm_ref = normalizer(ref) if normalizer else ref.lower()
            norm_pred = normalizer(pred) if normalizer else pred.lower()
            ref_tokens = norm_ref.split() if norm_ref else []
            pred_tokens = norm_pred.split() if norm_pred else []
            sample_dist = _levenshtein_distance(ref_tokens, pred_tokens)
            sample_wer = sample_dist / len(ref_tokens) if ref_tokens else 0.0
            total_errors += sample_dist
            total_ref_tokens += len(ref_tokens)

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

            if (i + 1) % 10 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                running_wer = total_errors / total_ref_tokens if total_ref_tokens else 0
                print(f"  [{i + 1}/{n}] {elapsed:.0f}s elapsed, WER={running_wer:.4f}")

        jsonl_fh.close()
        print(f"  Saved: {jsonl_path}")

        elapsed = time.time() - t0
        wer = total_errors / total_ref_tokens if total_ref_tokens else 0.0
        threshold = DEFAULT_DATASETS_WER.get(dataset_name, 1.0)
        passed = wer <= threshold
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        results[dataset_name] = {"n": n, "wer": wer, "threshold": threshold,
                                 "status": status, "elapsed": elapsed}
        print(f"  WER={wer:.4f} (threshold={threshold:.2f}) [{status}] ({elapsed:.0f}s)\n")

    print("=" * 65)
    print(f"{'Dataset':<30} {'N':>5} {'WER':>8} {'Thresh':>8} {'Status':>6}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<30} {r['n']:>5} {r['wer']:>8.4f} {r['threshold']:>8.2f} {r['status']:>6}")
    print("=" * 65)
    print(f"Decoder: {args.decoder}")
    print(f"mmproj: {args.mmproj}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
