#!/usr/bin/env python3
"""Download MERaLiON model variants from HuggingFace.

Usage:
    python scripts/download.py --model audiollm         # Primary 10B AudioLLM
    python scripts/download.py --model 2-3b             # Smaller 3B variant
    python scripts/download.py --model 2-10b            # MERaLiON-2 10B
    python scripts/download.py --model 2-10b-asr        # ASR-optimized 10B
    python scripts/download.py --model 3-10b-preview     # Latest preview
    python scripts/download.py --model speech-encoder-v1 # Speech encoder only
    python scripts/download.py --model speech-encoder-2  # Expanded speech encoder
    python scripts/download.py --model all               # Download all variants
    python scripts/download.py --list                    # List available models
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi


MODEL_REGISTRY = {
    "audiollm": {
        "repo_id": "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION",
        "description": "Primary 10B AudioLLM (Whisper-v2 + SEA-LION/Gemma2-9B)",
        "size_gb": 19.9,
    },
    "2-3b": {
        "repo_id": "MERaLiON/MERaLiON-2-3B",
        "description": "Smaller 3B variant (Whisper-v3 + Gemma2)",
        "size_gb": 6.98,
    },
    "2-10b": {
        "repo_id": "MERaLiON/MERaLiON-2-10B",
        "description": "MERaLiON-2 10B (extended audio, 300s)",
        "size_gb": 19.9,
    },
    "2-10b-asr": {
        "repo_id": "MERaLiON/MERaLiON-2-10B-ASR",
        "description": "ASR-optimized 10B variant",
        "size_gb": 19.9,
    },
    "3-10b-preview": {
        "repo_id": "MERaLiON/MERaLiON-3-10B-preview",
        "description": "Latest preview 10B model",
        "size_gb": 19.9,
    },
    "speech-encoder-v1": {
        "repo_id": "MERaLiON/MERaLiON-SpeechEncoder-v1",
        "description": "Speech encoder foundation model (630M)",
        "size_gb": 1.2,
    },
    "speech-encoder-2": {
        "repo_id": "MERaLiON/MERaLiON-SpeechEncoder-2",
        "description": "Expanded multilingual speech encoder (600M)",
        "size_gb": 1.2,
    },
}


def list_models():
    """Print available model variants."""
    print("\nAvailable MERaLiON models:\n")
    print(f"{'Key':<22} {'Size':>7}  Description")
    print("-" * 75)
    for key, info in MODEL_REGISTRY.items():
        print(f"  {key:<20} {info['size_gb']:>5.1f}GB  {info['description']}")
    total = sum(m["size_gb"] for m in MODEL_REGISTRY.values())
    print(f"\n  {'all':<20} {total:>5.1f}GB  Download all variants")
    print()


def download_model(
    model_key: str,
    output_dir: Path,
    token: str | None = None,
    resume: bool = True,
):
    """Download a single model variant."""
    info = MODEL_REGISTRY[model_key]
    repo_id = info["repo_id"]
    local_dir = output_dir / model_key

    print(f"\nDownloading {repo_id} (~{info['size_gb']}GB)")
    print(f"  → {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        resume_download=resume,
        ignore_patterns=["*.md", "*.png", "*.pdf", "*.gitattributes"],
    )

    print(f"  Done: {repo_id}")
    return local_dir


def inspect_model(model_dir: Path):
    """Print summary of downloaded model files."""
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    py_files = sorted(model_dir.glob("*.py"))
    config_files = sorted(model_dir.glob("*.json"))

    print(f"\n  Model directory: {model_dir}")
    print(f"  SafeTensors shards: {len(safetensor_files)}")
    for f in safetensor_files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"    {f.name}: {size_gb:.2f} GB")

    total_gb = sum(f.stat().st_size for f in safetensor_files) / (1024**3)
    print(f"  Total model size: {total_gb:.2f} GB")

    if py_files:
        print(f"  Custom code files: {', '.join(f.name for f in py_files)}")
    if config_files:
        print(f"  Config files: {', '.join(f.name for f in config_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download MERaLiON models from HuggingFace"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Model variant to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory (default: ./models)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect already-downloaded model files",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model:
        parser.print_help()
        print("\nError: --model is required (or use --list to see options)")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models_to_download = list(MODEL_REGISTRY.keys())
    else:
        models_to_download = [args.model]

    for model_key in models_to_download:
        model_dir = download_model(
            model_key, args.output_dir, token=args.token
        )

        if args.inspect:
            inspect_model(model_dir)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
