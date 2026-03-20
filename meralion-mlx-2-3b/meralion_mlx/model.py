"""Unified MERaLiON MLX model.

Combines Whisper encoder + MLP adaptor + Gemma2 decoder into a single
inference pipeline running natively on Apple Silicon via MLX.

Architecture:
    Audio → [Whisper Encoder] → [LayerNorm] → [MLP Adaptor] → speech_embeds
    Text  → [Token Embedding] → text_embeds
    Merge: replace speech token positions with speech_embeds
    Combined → [Gemma2 Decoder] → logits → tokens → text
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .whisper_encoder import WhisperEncoder, WhisperEncoderConfig
from .adaptor import MERaLiONSpeechAudioAdapter


class MERaLiONMLX(nn.Module):
    """MERaLiON AudioLLM running on MLX.

    This is the main inference class. It loads converted MLX weights
    and runs the full audio→text pipeline on Apple Silicon.
    """

    def __init__(self, config: dict):
        super().__init__()

        speech_config = config.get("speech_config", {})
        text_config = config.get("text_config", {})

        self.speech_token_index = config.get("speech_token_index", 255999)
        self.scale_factor = config.get("speech_mlp_scale_factor", 15)

        # Component 1: Whisper encoder
        whisper_config = WhisperEncoderConfig.from_dict(speech_config)
        self.speech_encoder = WhisperEncoder(whisper_config)

        # LayerNorm on speech encoder output
        self.ln_speech = nn.LayerNorm(whisper_config.d_model)

        # Component 2: MLP adaptor
        text_hidden_size = text_config.get("hidden_size", 3584)
        self.speech_audio_adapter = MERaLiONSpeechAudioAdapter(
            speech_hidden_size=whisper_config.d_model,
            text_hidden_size=text_hidden_size,
            scale_factor=self.scale_factor,
        )

        # Component 3: Gemma2 text decoder
        # We store the text model config for loading
        self._text_config = text_config
        self._text_hidden_size = text_hidden_size
        self._vocab_size = text_config.get("vocab_size", 256000)

        # The text decoder will be loaded separately since it uses
        # the full Gemma2 architecture which is complex.
        # We store it as a generic module container.
        self.text_decoder = None  # Set via load_text_decoder()

    def encode_speech(self, input_features: mx.array) -> mx.array:
        """Encode audio features through Whisper + adaptor.

        Args:
            input_features: (batch, n_mels, time) log-Mel spectrogram

        Returns:
            (batch, 100, text_hidden_size) speech embeddings
        """
        # Whisper encoder
        encoder_output = self.speech_encoder(input_features)

        # LayerNorm
        encoder_output = self.ln_speech(encoder_output)

        # MLP adaptor (compress + project)
        speech_embeds = self.speech_audio_adapter(encoder_output)

        return speech_embeds

    def merge_embeddings(
        self,
        input_ids: mx.array,
        speech_embeds: mx.array,
        text_embed_fn,
    ) -> mx.array:
        """Merge speech embeddings into text embedding sequence.

        Replaces positions where input_ids == speech_token_index with
        the corresponding speech embeddings.

        Args:
            input_ids: (batch, seq_len) token IDs
            speech_embeds: (batch, n_speech_tokens, hidden_size)
            text_embed_fn: Function to embed text tokens → (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size) merged embeddings
        """
        # Get text embeddings for all tokens
        text_embeds = text_embed_fn(input_ids)

        B, S, H = text_embeds.shape

        # For each batch item, find speech token positions and replace
        for b in range(B):
            # Find positions of speech tokens
            speech_mask = (input_ids[b] == self.speech_token_index)
            speech_positions = mx.where(speech_mask)[0]

            if speech_positions.size == 0:
                continue

            n_speech = min(speech_positions.size, speech_embeds.shape[1])

            # Replace speech token embeddings with actual speech embeddings
            # We do this by constructing the full embedding tensor
            indices = speech_positions[:n_speech]
            for i in range(n_speech):
                pos = int(indices[i])
                text_embeds = text_embeds.at[b, pos].add(
                    speech_embeds[b, i] - text_embeds[b, pos]
                )

        return text_embeds


def load_config(model_dir: str | Path) -> dict:
    """Load MERaLiON config.json."""
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def load_weights(model_dir: str | Path) -> dict[str, mx.array]:
    """Load all SafeTensors shards from model directory.

    Handles multi-shard SafeTensors files (model-00001-of-00004.safetensors, etc.)

    Args:
        model_dir: Path to model directory with .safetensors files

    Returns:
        Dict mapping weight names to MLX arrays
    """
    model_dir = Path(model_dir)
    weights = {}

    # Check for index file first
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(
            f.name for f in model_dir.glob("*.safetensors")
        )

    for shard_name in shard_files:
        shard_path = model_dir / shard_name
        print(f"  Loading {shard_name}...")
        shard_weights = mx.load(str(shard_path))
        weights.update(shard_weights)

    return weights


def partition_weights(
    weights: dict[str, mx.array],
) -> tuple[dict, dict, dict, dict]:
    """Partition MERaLiON weights into component groups.

    MERaLiON weight keys follow these prefixes:
        - speech_encoder.* → Whisper encoder
        - ln_speech.* → Speech LayerNorm
        - speech_audio_adapter.* → MLP adaptor
        - text_decoder.* → Gemma2 decoder

    Returns:
        (encoder_weights, ln_weights, adaptor_weights, decoder_weights)
    """
    encoder_weights = {}
    ln_weights = {}
    adaptor_weights = {}
    decoder_weights = {}

    for key, value in weights.items():
        if key.startswith("speech_encoder."):
            # Strip prefix for component loading
            new_key = key[len("speech_encoder."):]
            encoder_weights[new_key] = value
        elif key.startswith("ln_speech."):
            new_key = key[len("ln_speech."):]
            ln_weights[new_key] = value
        elif key.startswith("speech_audio_adapter."):
            new_key = key[len("speech_audio_adapter."):]
            adaptor_weights[new_key] = value
        elif key.startswith("text_decoder."):
            new_key = key[len("text_decoder."):]
            decoder_weights[new_key] = value
        else:
            # Unknown prefix — likely belongs to decoder
            decoder_weights[key] = value

    return encoder_weights, ln_weights, adaptor_weights, decoder_weights


def remap_whisper_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap HuggingFace Whisper weight keys to our MLX Whisper format.

    HF format: encoder.layers.0.self_attn.q_proj.weight
    MLX format: layers.0.self_attn.q_proj.weight

    Also handles Conv1d weight transposition (HF uses [out, in, kernel],
    MLX Conv1d expects [out, kernel, in]).
    """
    remapped = {}

    for key, value in weights.items():
        # Strip 'encoder.' prefix if present (HF Whisper nests under encoder.)
        new_key = key
        if new_key.startswith("encoder."):
            new_key = new_key[len("encoder."):]

        # Handle embed_positions — HF stores as (max_len, d_model)
        if "embed_positions" in new_key:
            new_key = "embed_positions"
            remapped[new_key] = value
            continue

        # Conv1d weight transposition: HF (out_ch, in_ch, kernel) → MLX (out_ch, kernel, in_ch)
        if ("conv1.weight" in new_key or "conv2.weight" in new_key) and value.ndim == 3:
            value = mx.transpose(value, axes=(0, 2, 1))

        remapped[new_key] = value

    return remapped


def remap_adaptor_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap adaptor weight keys from HF to MLX format.

    nn.Sequential in MLX uses .layers. prefix for indexed children:
        HF: mlp_adapter.0.weight → MLX: mlp_adapter.layers.0.weight
        HF: speech_llm_proj.0.weight → MLX: speech_llm_proj.layers.0.weight

    Direct nn.Linear attributes (gate_proj, pool_proj, out_proj) need no remapping.
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key

        # Remap Sequential children: prefix.N.param → prefix.layers.N.param
        # Match patterns like "mlp_adapter.0.weight" or "speech_llm_proj.2.bias"
        for prefix in ("mlp_adapter", "speech_llm_proj"):
            if key.startswith(f"{prefix}.") and not key.startswith(f"{prefix}.layers."):
                suffix = key[len(f"{prefix}."):]
                # Only remap if suffix starts with a digit (Sequential index)
                if suffix and suffix[0].isdigit():
                    new_key = f"{prefix}.layers.{suffix}"
                break

        remapped[new_key] = value

    return remapped


def save_component_weights(
    weights: dict[str, mx.array],
    output_path: str | Path,
):
    """Save component weights as SafeTensors."""
    from safetensors.numpy import save_file

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert mx arrays to numpy for safetensors
    # numpy doesn't support bfloat16 — cast to float16 first
    np_weights = {}
    for key, value in weights.items():
        if value.dtype == mx.bfloat16:
            value = value.astype(mx.float16)
        np_weights[key] = np.array(value)

    save_file(np_weights, str(output_path))
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"  Saved {output_path.name} ({size_mb:.1f} MB, {len(np_weights)} tensors)")


def print_weight_summary(name: str, weights: dict[str, mx.array]):
    """Print summary of weights in a component."""
    total_params = sum(v.size for v in weights.values())
    total_bytes = sum(v.nbytes for v in weights.values())
    print(f"  {name}: {len(weights)} tensors, {total_params:,} params, {total_bytes / 1024**2:.1f} MB")
    for key, value in sorted(weights.items())[:5]:
        print(f"    {key}: {value.shape} {value.dtype}")
    if len(weights) > 5:
        print(f"    ... and {len(weights) - 5} more")
