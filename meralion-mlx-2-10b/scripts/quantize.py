#!/usr/bin/env python3
"""Quantize MERaLiON MLX model for reduced memory footprint.

Applies MLX's affine quantization (4-bit or 8-bit) to model components.
The decoder benefits most from quantization (largest component).

Usage:
    # 4-bit quantization (default, ~75% size reduction)
    python scripts/quantize.py --model-dir models/audiollm-mlx --output-dir models/audiollm-mlx-4bit

    # 8-bit quantization (~50% size reduction)
    python scripts/quantize.py --model-dir models/audiollm-mlx --output-dir models/audiollm-mlx-8bit --bits 8

    # Quantize only decoder (encoder keeps full precision for accuracy)
    python scripts/quantize.py --model-dir models/audiollm-mlx --output-dir models/audiollm-mlx-4bit --decoder-only

    # Custom group size
    python scripts/quantize.py --model-dir models/audiollm-mlx --output-dir models/audiollm-mlx-4bit --group-size 32
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def quantize_weights(
    weights: dict[str, mx.array],
    bits: int = 4,
    group_size: int = 64,
    skip_patterns: list[str] | None = None,
) -> dict[str, mx.array]:
    """Quantize weight tensors using MLX affine quantization.

    Only quantizes 2D weight matrices (Linear layers). Biases, norms,
    embeddings, and 1D tensors are kept at full precision.

    Args:
        weights: Dict of weight name → mx.array
        bits: Quantization bits (4 or 8)
        group_size: Group size for affine quantization
        skip_patterns: List of key patterns to skip (keep full precision)

    Returns:
        Dict of quantized weights (includes quantized data, scales, biases)
    """
    skip_patterns = skip_patterns or []
    quantized = {}
    n_quantized = 0
    n_skipped = 0
    original_bytes = 0
    quantized_bytes = 0

    for key, value in weights.items():
        original_bytes += value.nbytes

        # Skip non-weight tensors and small tensors
        should_skip = (
            value.ndim != 2
            or "norm" in key.lower()
            or "embed" in key.lower()
            or "bias" in key
            or value.shape[0] < group_size
            or value.shape[1] < group_size
            or any(pat in key for pat in skip_patterns)
        )

        if should_skip:
            quantized[key] = value
            quantized_bytes += value.nbytes
            n_skipped += 1
            continue

        # Quantize the weight
        q_weight, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
        quantized[key] = q_weight
        quantized[f"{key}_scales"] = scales
        quantized[f"{key}_biases"] = biases
        quantized_bytes += q_weight.nbytes + scales.nbytes + biases.nbytes
        n_quantized += 1

    compression = (1 - quantized_bytes / original_bytes) * 100 if original_bytes > 0 else 0
    print(f"  Quantized {n_quantized} tensors, skipped {n_skipped}")
    print(f"  Size: {original_bytes / 1024**2:.1f}MB → {quantized_bytes / 1024**2:.1f}MB "
          f"({compression:.1f}% reduction)")

    return quantized


def save_quantized_weights(
    weights: dict[str, mx.array],
    output_path: Path,
):
    """Save quantized weights as SafeTensors."""
    from safetensors.numpy import save_file

    np_weights = {}
    for key, value in weights.items():
        if value.dtype == mx.bfloat16:
            value = value.astype(mx.float16)
        np_weights[key] = np.array(value)

    save_file(np_weights, str(output_path))
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"  Saved {output_path.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize MERaLiON MLX model"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to converted MLX model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--decoder-only",
        action="store_true",
        help="Only quantize decoder (keep encoder at full precision)",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"Quantizing MERaLiON model ({args.bits}-bit, group_size={args.group_size})")
    print(f"  Source: {args.model_dir}")
    print(f"  Output: {args.output_dir}")

    # Process encoder
    encoder_path = args.model_dir / "encoder.safetensors"
    if encoder_path.exists():
        if args.decoder_only:
            print("\n[Encoder] Copying at full precision...")
            shutil.copy2(encoder_path, args.output_dir / "encoder.safetensors")
        else:
            print("\n[Encoder] Quantizing...")
            encoder_weights = mx.load(str(encoder_path))
            q_encoder = quantize_weights(
                encoder_weights, bits=args.bits, group_size=args.group_size,
            )
            save_quantized_weights(q_encoder, args.output_dir / "encoder.safetensors")

    # Process adaptor (always keep full precision — small and accuracy-sensitive)
    adaptor_path = args.model_dir / "adaptor.safetensors"
    if adaptor_path.exists():
        print("\n[Adaptor] Copying at full precision (small, accuracy-sensitive)...")
        shutil.copy2(adaptor_path, args.output_dir / "adaptor.safetensors")

    # Process decoder (largest component — biggest benefit from quantization)
    print("\n[Decoder] Quantizing...")
    decoder_files = sorted(args.model_dir.glob("decoder*.safetensors"))
    if not decoder_files:
        decoder_files = [args.model_dir / "decoder.safetensors"]

    for df in decoder_files:
        if not df.exists():
            continue
        if df.name.endswith(".index.json"):
            continue
        print(f"  Processing {df.name}...")
        decoder_weights = mx.load(str(df))
        q_decoder = quantize_weights(
            decoder_weights, bits=args.bits, group_size=args.group_size,
        )
        out_name = df.name
        save_quantized_weights(q_decoder, args.output_dir / out_name)

    # Copy config and tokenizer files
    print("\n[Config] Copying configuration files...")
    for pattern in ["*.json", "tokenizer.model", "special_tokens_map.json"]:
        for f in args.model_dir.glob(pattern):
            if f.name not in ("model.safetensors.index.json",):
                shutil.copy2(f, args.output_dir / f.name)

    # Save quantization metadata
    quant_config = {
        "bits": args.bits,
        "group_size": args.group_size,
        "decoder_only": args.decoder_only,
        "source": str(args.model_dir),
    }
    with open(args.output_dir / "quantization_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nQuantization complete in {total_time:.1f}s")

    # Print size comparison
    src_size = sum(f.stat().st_size for f in args.model_dir.glob("*.safetensors"))
    dst_size = sum(f.stat().st_size for f in args.output_dir.glob("*.safetensors"))
    print(f"  Original: {src_size / 1024**3:.2f} GB")
    print(f"  Quantized: {dst_size / 1024**3:.2f} GB")
    print(f"  Reduction: {(1 - dst_size / src_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
