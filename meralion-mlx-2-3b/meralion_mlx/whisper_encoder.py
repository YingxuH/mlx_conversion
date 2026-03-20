"""MLX Whisper encoder for MERaLiON.

Implements the Whisper encoder architecture in MLX, matching the
WhisperEncoder from HuggingFace transformers used by MERaLiON.
Based on the mlx-examples whisper implementation.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class WhisperEncoderConfig:
    """Whisper encoder configuration matching MERaLiON's speech_config."""

    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    activation_function: str = "gelu"
    dropout: float = 0.0
    scale_embedding: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "WhisperEncoderConfig":
        """Create config from MERaLiON's speech_config dict."""
        return cls(
            d_model=d.get("d_model", 1280),
            encoder_layers=d.get("encoder_layers", 32),
            encoder_attention_heads=d.get("encoder_attention_heads", 20),
            encoder_ffn_dim=d.get("encoder_ffn_dim", 5120),
            num_mel_bins=d.get("num_mel_bins", 80),
            max_source_positions=d.get("max_source_positions", 1500),
            activation_function=d.get("activation_function", "gelu"),
            dropout=d.get("dropout", 0.0),
            scale_embedding=d.get("scale_embedding", False),
        )


class WhisperAttention(nn.Module):
    """Multi-head attention for Whisper encoder."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(out)


class WhisperEncoderLayer(nn.Module):
    """Single Whisper encoder layer."""

    def __init__(self, config: WhisperEncoderConfig):
        super().__init__()
        self.self_attn = WhisperAttention(config.d_model, config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)

        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention with residual
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        x = residual + x

        return x


class WhisperEncoder(nn.Module):
    """Full Whisper encoder matching MERaLiON's speech_encoder.

    Architecture:
        1. Conv1d feature extraction (2 layers)
        2. Sinusoidal positional embedding
        3. N transformer encoder layers
        4. Final layer norm
    """

    def __init__(self, config: WhisperEncoderConfig):
        super().__init__()
        self.config = config

        # Conv feature extractor: 2 Conv1d layers
        # Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)
        # Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv1d(
            in_channels=config.num_mel_bins,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Positional embedding (sinusoidal, frozen)
        self.embed_positions = self._sinusoidal_embeddings(
            config.max_source_positions, config.d_model
        )

        # Encoder layers
        self.layers = [
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ]

        self.layer_norm = nn.LayerNorm(config.d_model)

    @staticmethod
    def _sinusoidal_embeddings(max_len: int, d_model: int) -> mx.array:
        """Create sinusoidal positional embeddings (non-trainable)."""
        position = mx.arange(max_len).reshape(-1, 1)
        div_term = mx.exp(
            mx.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        embeddings = mx.zeros((max_len, d_model))
        # sin for even indices, cos for odd indices
        sin_vals = mx.sin(position * div_term)
        cos_vals = mx.cos(position * div_term)
        # Interleave sin and cos
        embeddings = mx.concatenate(
            [sin_vals.reshape(max_len, -1, 1), cos_vals.reshape(max_len, -1, 1)],
            axis=-1,
        ).reshape(max_len, d_model)
        return embeddings

    def __call__(self, input_features: mx.array) -> mx.array:
        """Encode audio features.

        Args:
            input_features: (batch, num_mel_bins, time) log-Mel spectrogram

        Returns:
            (batch, seq_len, d_model) encoder hidden states
        """
        # MLX Conv1d expects (batch, seq_len, channels) — transpose from (B, C, T) to (B, T, C)
        x = input_features.transpose(0, 2, 1)

        # Conv feature extraction
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        # x is now (B, T/2, d_model) — add positional embeddings
        # Truncate to max_source_positions if conv output is slightly longer
        seq_len = x.shape[1]
        max_pos = self.embed_positions.shape[0]
        if seq_len > max_pos:
            x = x[:, :max_pos, :]
            seq_len = max_pos
        x = x + self.embed_positions[:seq_len]

        # Encoder layers
        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)
        return x
