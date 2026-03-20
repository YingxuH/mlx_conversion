#!/usr/bin/env python3
"""Convert MERaLiON weights from HuggingFace format to MLX format.

This script:
1. Loads the full MERaLiON SafeTensors weights
2. Partitions them into 3 components (encoder, adaptor, decoder)
3. Remaps weight keys to match MLX module names
4. Saves each component as separate SafeTensors files

Usage:
    python scripts/convert.py --model-dir models/audiollm --output-dir models/audiollm-mlx
    python scripts/convert.py --model-dir models/2-3b --output-dir models/2-3b-mlx
    python scripts/convert.py --model-dir models/audiollm --output-dir models/audiollm-mlx --component encoder
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meralion_mlx.model import (
    load_config,
    load_weights,
    partition_weights,
    print_weight_summary,
    remap_whisper_keys,
    remap_adaptor_keys,
    save_component_weights,
)


def convert_encoder(
    encoder_weights: dict[str, mx.array],
    output_dir: Path,
    config: dict,
):
    """Convert and save Whisper encoder weights."""
    print("\n[Encoder] Converting Whisper encoder...")
    remapped = remap_whisper_keys(encoder_weights)
    print_weight_summary("Whisper Encoder", remapped)
    save_component_weights(remapped, output_dir / "encoder.safetensors")

    # Save encoder config
    speech_config = config.get("speech_config", {})
    with open(output_dir / "encoder_config.json", "w") as f:
        json.dump(speech_config, f, indent=2)
    print("  Saved encoder_config.json")


def convert_adaptor(
    adaptor_weights: dict[str, mx.array],
    ln_weights: dict[str, mx.array],
    output_dir: Path,
    config: dict,
):
    """Convert and save MLP adaptor weights (including speech LayerNorm)."""
    print("\n[Adaptor] Converting MLP adaptor...")

    remapped = remap_adaptor_keys(adaptor_weights)

    # Include ln_speech weights with prefix
    for key, value in ln_weights.items():
        remapped[f"ln_speech.{key}"] = value

    print_weight_summary("MLP Adaptor + LN", remapped)
    save_component_weights(remapped, output_dir / "adaptor.safetensors")

    # Save adaptor config
    adaptor_config = {
        "speech_hidden_size": config.get("speech_config", {}).get("d_model", 1280),
        "text_hidden_size": config.get("text_config", {}).get("hidden_size", 3584),
        "scale_factor": config.get("speech_mlp_scale_factor", 15),
    }
    with open(output_dir / "adaptor_config.json", "w") as f:
        json.dump(adaptor_config, f, indent=2)
    print("  Saved adaptor_config.json")


def convert_decoder(
    decoder_weights: dict[str, mx.array],
    output_dir: Path,
    config: dict,
):
    """Convert and save Gemma2/Llama decoder weights.

    The decoder weights are saved in a format compatible with mlx-lm,
    which can load standard Gemma2/Llama weights.
    """
    print("\n[Decoder] Converting text decoder...")

    # Strip 'model.' prefix if present (HF Gemma2 nests under model.)
    remapped = {}
    for key, value in decoder_weights.items():
        new_key = key
        # Keep the weights as-is for now — mlx-lm loading will handle remapping
        remapped[new_key] = value

    print_weight_summary("Text Decoder", remapped)

    # Save decoder weights — may need multiple shards for large models
    total_bytes = sum(v.nbytes for v in remapped.values())
    if total_bytes > 4 * 1024**3:  # > 4GB → shard
        shard_size = 4 * 1024**3
        shard_idx = 0
        current_shard = {}
        current_bytes = 0
        weight_map = {}

        for key, value in remapped.items():
            if current_bytes + value.nbytes > shard_size and current_shard:
                shard_name = f"decoder-{shard_idx:05d}.safetensors"
                save_component_weights(current_shard, output_dir / shard_name)
                shard_idx += 1
                current_shard = {}
                current_bytes = 0

            current_shard[key] = value
            current_bytes += value.nbytes
            shard_name = f"decoder-{shard_idx:05d}.safetensors"
            weight_map[key] = shard_name

        if current_shard:
            shard_name = f"decoder-{shard_idx:05d}.safetensors"
            save_component_weights(current_shard, output_dir / shard_name)

        # Save index
        with open(output_dir / "decoder.safetensors.index.json", "w") as f:
            json.dump({"weight_map": weight_map}, f, indent=2)
        print(f"  Saved {shard_idx + 1} decoder shards")
    else:
        save_component_weights(remapped, output_dir / "decoder.safetensors")

    # Save decoder config
    text_config = config.get("text_config", {})
    text_config["model_type"] = text_config.get("model_type", "gemma2")
    with open(output_dir / "decoder_config.json", "w") as f:
        json.dump(text_config, f, indent=2)
    print("  Saved decoder_config.json")


def copy_tokenizer(model_dir: Path, output_dir: Path):
    """Copy tokenizer files to output directory."""
    print("\n[Tokenizer] Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
    ]
    copied = 0
    for fname in tokenizer_files:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            copied += 1

    print(f"  Copied {copied} tokenizer/config files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MERaLiON weights to MLX format"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to downloaded MERaLiON model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for converted MLX weights",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["encoder", "adaptor", "decoder", "all"],
        default="all",
        help="Convert only a specific component (default: all)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect weights without converting",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"Loading config from {args.model_dir}...")
    config = load_config(args.model_dir)
    print(f"  Model type: {config.get('model_type', 'unknown')}")
    print(f"  Speech token index: {config.get('speech_token_index', 'N/A')}")
    print(f"  Scale factor: {config.get('speech_mlp_scale_factor', 'N/A')}")

    # Save full config to output
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load all weights
    print(f"\nLoading weights from {args.model_dir}...")
    t0 = time.time()
    weights = load_weights(args.model_dir)
    t1 = time.time()
    print(f"  Loaded {len(weights)} tensors in {t1 - t0:.1f}s")

    # Partition into components
    print("\nPartitioning weights...")
    encoder_w, ln_w, adaptor_w, decoder_w = partition_weights(weights)
    print(f"  Encoder:  {len(encoder_w)} tensors")
    print(f"  LN:       {len(ln_w)} tensors")
    print(f"  Adaptor:  {len(adaptor_w)} tensors")
    print(f"  Decoder:  {len(decoder_w)} tensors")

    if args.inspect_only:
        print("\n--- Detailed Weight Inspection ---")
        print_weight_summary("Encoder", encoder_w)
        print_weight_summary("LayerNorm", ln_w)
        print_weight_summary("Adaptor", adaptor_w)
        print_weight_summary("Decoder", decoder_w)
        print("\n[Inspect only — no files written]")
        return

    # Convert components
    if args.component in ("encoder", "all"):
        convert_encoder(encoder_w, args.output_dir, config)

    if args.component in ("adaptor", "all"):
        convert_adaptor(adaptor_w, ln_w, args.output_dir, config)

    if args.component in ("decoder", "all"):
        convert_decoder(decoder_w, args.output_dir, config)

    # Copy tokenizer
    if args.component == "all":
        copy_tokenizer(args.model_dir, args.output_dir)

    t2 = time.time()
    print(f"\nConversion complete in {t2 - t0:.1f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
