#!/usr/bin/env python3
"""Quantize MERaLiON decoder weights to mlx-lm compatible 4-bit format.

Unlike quantize.py (which uses `weight_scales`/`weight_biases`), this script
uses the mlx-lm convention of `.scales`/`.biases` (without the `weight_` prefix)
and writes a `quantization` block to decoder_config.json so that mlx-lm's
`load()` correctly converts Linear → QuantizedLinear.

Processes decoder shards one at a time to keep peak memory low (~4GB/shard).

Usage:
    python scripts/quantize_for_mlxlm.py \
        --model-dir models/2-10b-mlx \
        --output-dir models/2-10b-mlx-4bit \
        [--bits 4] [--group-size 64]
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def quantize_shard(
    weights: dict,
    bits: int,
    group_size: int,
) -> tuple[dict, dict]:
    """Quantize a shard's weights in-place (mlx-lm key format).

    Returns (quantized_weights, weight_map) where weight_map maps each key
    to the current shard filename (filled in by the caller).

    mlx-lm expects:
        {module_path}.weight  → quantized int4/8 matrix
        {module_path}.scales  → float16 scales
        {module_path}.biases  → float16 biases  (only if mode uses biases)

    where module_path = weight_key[:-len('.weight')]
    """
    out = {}
    n_quantized = n_skipped = 0
    orig_bytes = quant_bytes = 0

    for key, value in weights.items():
        orig_bytes += value.nbytes

        should_skip = (
            value.ndim != 2
            or "norm" in key.lower()
            or "embed" in key.lower()
            or not key.endswith(".weight")
            or value.shape[0] < group_size
            or value.shape[1] < group_size
        )

        if should_skip:
            out[key] = value
            quant_bytes += value.nbytes
            n_skipped += 1
            continue

        # Quantize; mx.quantize returns (quantized, scales[, biases])
        results = mx.quantize(value, group_size=group_size, bits=bits)
        q_weight = results[0]
        scales = results[1]
        biases = results[2] if len(results) > 2 else None

        module_path = key[: -len(".weight")]  # strip ".weight"
        out[key] = q_weight
        out[f"{module_path}.scales"] = scales
        if biases is not None:
            out[f"{module_path}.biases"] = biases

        quant_bytes += q_weight.nbytes + scales.nbytes + (biases.nbytes if biases is not None else 0)
        n_quantized += 1

    reduction = (1 - quant_bytes / orig_bytes) * 100 if orig_bytes else 0
    print(f"    quantized={n_quantized}  skipped={n_skipped}  "
          f"{orig_bytes/1024**2:.0f}MB → {quant_bytes/1024**2:.0f}MB  ({reduction:.0f}% saved)")
    return out


def save_shard(weights: dict, path: Path) -> None:
    """Save quantized weights as a SafeTensors file (via numpy)."""
    from safetensors.numpy import save_file

    np_w = {}
    for k, v in weights.items():
        # safetensors-numpy doesn't support bfloat16 or uint32 directly
        if v.dtype == mx.bfloat16:
            v = v.astype(mx.float16)
        np_w[k] = np.array(v)
    save_file(np_w, str(path))
    print(f"    saved  {path.name}  ({path.stat().st_size / 1024**2:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Quantize MERaLiON decoder for mlx-lm")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    if not args.model_dir.exists():
        print(f"ERROR: model dir not found: {args.model_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Quantizing {args.bits}-bit (group_size={args.group_size})")
    print(f"  source : {args.model_dir}")
    print(f"  output : {args.output_dir}")

    # ------------------------------------------------------------------
    # Copy non-decoder files unchanged (encoder, adaptor, config, tokenizer)
    # ------------------------------------------------------------------
    print("\n[Step 1] Copying non-decoder files ...")
    for pattern in [
        "encoder.safetensors", "adaptor.safetensors",
        "*.json", "tokenizer.model",
    ]:
        for f in args.model_dir.glob(pattern):
            if "decoder" not in f.name:
                shutil.copy2(f, args.output_dir / f.name)
    print("  done")

    # ------------------------------------------------------------------
    # Update decoder_config.json with quantization info so mlx-lm
    # knows to convert Linear → QuantizedLinear during load.
    # ------------------------------------------------------------------
    dec_config_src = args.model_dir / "decoder_config.json"
    dec_config_dst = args.output_dir / "decoder_config.json"
    with open(dec_config_src) as fh:
        dec_cfg = json.load(fh)
    dec_cfg["quantization"] = {"bits": args.bits, "group_size": args.group_size}
    with open(dec_config_dst, "w") as fh:
        json.dump(dec_cfg, fh, indent=2)
    print(f"\n[Step 2] Updated decoder_config.json with quantization metadata")

    # ------------------------------------------------------------------
    # Quantize decoder shards one at a time
    # ------------------------------------------------------------------
    decoder_shards = sorted(args.model_dir.glob("decoder-*.safetensors"))
    if not decoder_shards:
        decoder_shards = [args.model_dir / "decoder.safetensors"]
    decoder_shards = [p for p in decoder_shards if p.exists()]

    print(f"\n[Step 3] Quantizing {len(decoder_shards)} decoder shard(s) ...")
    all_keys_to_shard: dict[str, str] = {}  # key → shard filename

    for shard_path in decoder_shards:
        print(f"\n  Shard: {shard_path.name}")
        weights = mx.load(str(shard_path))
        q_weights = quantize_shard(weights, bits=args.bits, group_size=args.group_size)
        # Force evaluation before saving so memory is freed after save
        mx.eval(*q_weights.values())

        out_name = shard_path.name  # keep same filename
        out_path = args.output_dir / out_name
        save_shard(q_weights, out_path)

        for k in q_weights:
            all_keys_to_shard[k] = out_name

        del weights, q_weights

    # ------------------------------------------------------------------
    # Write decoder shard index
    # ------------------------------------------------------------------
    if len(decoder_shards) > 1:
        index = {"metadata": {"format": "mlx"}, "weight_map": dict(sorted(all_keys_to_shard.items()))}
        idx_path = args.output_dir / "decoder.safetensors.index.json"
        with open(idx_path, "w") as fh:
            json.dump(index, fh, indent=2)
        print(f"\n  Wrote index: {idx_path.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    src_bytes = sum(p.stat().st_size for p in decoder_shards)
    dst_bytes = sum(p.stat().st_size for p in args.output_dir.glob("decoder-*.safetensors"))
    if dst_bytes == 0:
        dst_bytes = sum(p.stat().st_size for p in args.output_dir.glob("decoder.safetensors"))
    print(f"\nDecoder: {src_bytes/1024**3:.2f} GB → {dst_bytes/1024**3:.2f} GB "
          f"({(1 - dst_bytes/src_bytes)*100:.0f}% reduction)")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
