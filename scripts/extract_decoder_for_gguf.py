#!/usr/bin/env python3
"""Extract MERaLiON's text decoder weights for GGUF conversion.

Creates a directory with Gemma2-compatible weights and config by:
1. Stripping the 'text_decoder.' prefix from weight keys
2. Copying text_config as the main config
3. Copying tokenizer files

The output directory can be directly converted with llama.cpp's convert_hf_to_gguf.py.

Usage:
    python scripts/extract_decoder_for_gguf.py \
        --model-dir models/2-10b-hf \
        --output-dir models/2-10b-decoder
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description="Extract MERaLiON decoder for GGUF conversion")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to HF MERaLiON model")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for Gemma2 decoder")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and extract text_config as main config
    with open(args.model_dir / "config.json") as f:
        config = json.load(f)

    text_config = config["text_config"]
    text_config["architectures"] = ["Gemma2ForCausalLM"]
    text_config["model_type"] = "gemma2"

    with open(args.output_dir / "config.json", "w") as f:
        json.dump(text_config, f, indent=2)
    print(f"Wrote config.json (Gemma2)")

    # Copy tokenizer files
    for pattern in ["tokenizer*", "special_tokens_map.json", "generation_config.json"]:
        for src in args.model_dir.glob(pattern):
            shutil.copy2(src, args.output_dir / src.name)
    print("Copied tokenizer files")

    # Load weight index to find all shards
    index_path = args.model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    # Extract text_decoder.* weights with prefix stripped
    new_weight_map = {}
    shard_idx = 0
    current_shard = {}
    current_size = 0
    max_shard_size = 4 * 1024 ** 3  # 4GB per shard

    for shard_file in shard_files:
        print(f"Processing {shard_file}...")
        weights = load_file(str(args.model_dir / shard_file))

        for key, tensor in weights.items():
            if not key.startswith("text_decoder."):
                continue
            new_key = key.removeprefix("text_decoder.")
            current_shard[new_key] = tensor
            current_size += tensor.numel() * tensor.element_size()

            if current_size >= max_shard_size:
                out_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                save_file(current_shard, str(args.output_dir / out_name))
                for k in current_shard:
                    new_weight_map[k] = out_name
                print(f"  Saved {out_name} ({current_size / 1024**3:.1f} GB)")
                current_shard = {}
                current_size = 0
                shard_idx += 1

    # Save remaining
    if current_shard:
        out_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(current_shard, str(args.output_dir / out_name))
        for k in current_shard:
            new_weight_map[k] = out_name
        print(f"  Saved {out_name} ({current_size / 1024**3:.1f} GB)")
        shard_idx += 1

    # Fix shard names with total count
    total_shards = shard_idx
    final_weight_map = {}
    for key, old_name in new_weight_map.items():
        new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
        final_weight_map[key] = new_name

    # Rename files
    for i in range(total_shards):
        old = args.output_dir / f"model-{i:05d}-of-XXXXX.safetensors"
        new = args.output_dir / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old.rename(new)

    # Write index
    index_out = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in load_file(str(args.output_dir / f"model-00000-of-{total_shards:05d}.safetensors")).values()) * total_shards},
        "weight_map": dict(sorted(final_weight_map.items())),
    }
    with open(args.output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_out, f, indent=2)

    print(f"\nDone! Extracted {len(final_weight_map)} tensors in {total_shards} shard(s)")
    print(f"Output: {args.output_dir}")
    print(f"\nTo convert to GGUF:")
    print(f"  python convert_hf_to_gguf.py {args.output_dir} --outfile meralion-decoder.gguf --outtype q8_0")


if __name__ == "__main__":
    main()
