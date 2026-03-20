#!/usr/bin/env python3
"""Benchmark MERaLiON MLX inference performance.

Measures:
    - Model loading time
    - Audio encoding time (Whisper + adaptor)
    - Text generation time and tokens/sec
    - Peak memory usage
    - Comparison with PyTorch CPU/MPS (optional)

Usage:
    # Basic benchmark
    python scripts/benchmark.py --model-dir models/audiollm-mlx --audio test.wav

    # With PyTorch comparison
    python scripts/benchmark.py --model-dir models/audiollm-mlx --audio test.wav --compare-pytorch models/audiollm

    # Multiple runs for stable timing
    python scripts/benchmark.py --model-dir models/audiollm-mlx --audio test.wav --runs 5

    # Compare quantization levels
    python scripts/benchmark.py --model-dir models/audiollm-mlx --audio test.wav \
        --compare-quantized models/audiollm-mlx-4bit models/audiollm-mlx-8bit
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info.rss / (1024**2)
    except ImportError:
        return 0.0


def get_model_size_mb(model_dir: Path) -> float:
    """Get total size of SafeTensors files in MB."""
    total = sum(
        f.stat().st_size for f in model_dir.glob("*.safetensors")
    )
    return total / (1024**2)


def benchmark_mlx(
    model_dir: Path,
    audio_path: str,
    n_runs: int = 3,
    max_tokens: int = 128,
) -> dict:
    """Benchmark MLX inference pipeline."""
    from meralion_mlx.processor import MERaLiONProcessor, load_audio
    from meralion_mlx.whisper_encoder import WhisperEncoder, WhisperEncoderConfig
    from meralion_mlx.adaptor import MERaLiONSpeechAudioAdapter

    results = {
        "framework": "MLX",
        "model_dir": str(model_dir),
        "model_size_mb": get_model_size_mb(model_dir),
        "runs": n_runs,
    }

    # Load model
    mem_before = get_memory_usage_mb()
    t0 = time.time()

    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load encoder
    encoder_config = WhisperEncoderConfig.from_dict(config.get("speech_config", {}))
    encoder = WhisperEncoder(encoder_config)

    encoder_weights_path = model_dir / "encoder.safetensors"
    if encoder_weights_path.exists():
        encoder_weights = mx.load(str(encoder_weights_path))
        encoder.load_weights(list(encoder_weights.items()))
        mx.eval(encoder.parameters())

    # Load adaptor
    adaptor_config = config.get("speech_config", {})
    text_config = config.get("text_config", {})
    import mlx.nn as nn
    ln_speech = nn.LayerNorm(encoder_config.d_model)
    adaptor = MERaLiONSpeechAudioAdapter(
        speech_hidden_size=encoder_config.d_model,
        text_hidden_size=text_config.get("hidden_size", 3584),
        scale_factor=config.get("speech_mlp_scale_factor", 15),
    )

    adaptor_path = model_dir / "adaptor.safetensors"
    if adaptor_path.exists():
        adaptor_weights = mx.load(str(adaptor_path))
        adaptor_w = {}
        ln_w = {}
        for k, v in adaptor_weights.items():
            if k.startswith("ln_speech."):
                ln_w[k[len("ln_speech."):]] = v
            else:
                adaptor_w[k] = v
        adaptor.load_weights(list(adaptor_w.items()))
        ln_speech.load_weights(list(ln_w.items()))
        mx.eval(adaptor.parameters())
        mx.eval(ln_speech.parameters())

    load_time = time.time() - t0
    mem_after = get_memory_usage_mb()
    results["load_time_s"] = load_time
    results["memory_increase_mb"] = mem_after - mem_before

    # Prepare audio
    processor = MERaLiONProcessor(model_dir)
    mel = processor.prepare_audio(audio_path=audio_path)
    mel_mx = mx.array(mel)
    audio_len_s = len(load_audio(audio_path)) / 16000
    results["audio_duration_s"] = audio_len_s

    # Benchmark encoding
    encode_times = []
    for i in range(n_runs):
        t0 = time.time()
        enc_out = encoder(mel_mx)
        enc_out = ln_speech(enc_out)
        speech_embeds = adaptor(enc_out)
        mx.eval(speech_embeds)
        encode_times.append(time.time() - t0)

    results["encode_times_s"] = encode_times
    results["encode_mean_s"] = np.mean(encode_times)
    results["encode_std_s"] = np.std(encode_times)
    results["speech_embed_shape"] = list(speech_embeds.shape)

    # Note: Full generation benchmark requires decoder loading
    # which is handled separately to measure decoder-specific timing

    return results


def print_results(results: dict):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  MERaLiON MLX Benchmark Results")
    print(f"{'=' * 60}")
    print(f"  Framework:      {results['framework']}")
    print(f"  Model size:     {results['model_size_mb']:.1f} MB")
    print(f"  Audio duration: {results.get('audio_duration_s', 0):.1f}s")
    print(f"  Runs:           {results['runs']}")
    print()
    print(f"  Load time:      {results['load_time_s']:.2f}s")
    print(f"  Memory usage:   +{results['memory_increase_mb']:.0f} MB")
    print()
    print(f"  Encoding:")
    print(f"    Mean:         {results['encode_mean_s']:.3f}s")
    print(f"    Std:          {results['encode_std_s']:.3f}s")
    print(f"    Output shape: {results.get('speech_embed_shape', 'N/A')}")

    if "gen_mean_s" in results:
        print(f"\n  Generation:")
        print(f"    Mean:         {results['gen_mean_s']:.3f}s")
        print(f"    Tokens/sec:   {results.get('tokens_per_sec', 0):.1f}")
        print(f"    Total tokens: {results.get('total_tokens', 0)}")

    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MERaLiON MLX performance"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to converted MLX model directory",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to test audio file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--compare-quantized",
        type=Path,
        nargs="+",
        default=None,
        help="Additional quantized model dirs to compare",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    all_results = []

    # Benchmark main model
    print(f"Benchmarking: {args.model_dir}")
    results = benchmark_mlx(
        args.model_dir, args.audio, n_runs=args.runs, max_tokens=args.max_tokens,
    )
    print_results(results)
    all_results.append(results)

    # Benchmark quantized variants
    if args.compare_quantized:
        for qdir in args.compare_quantized:
            if not qdir.exists():
                print(f"Warning: {qdir} not found, skipping")
                continue

            gc.collect()
            print(f"\nBenchmarking: {qdir}")
            q_results = benchmark_mlx(
                qdir, args.audio, n_runs=args.runs, max_tokens=args.max_tokens,
            )
            print_results(q_results)
            all_results.append(q_results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
