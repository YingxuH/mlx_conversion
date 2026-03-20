"""Unit tests for MERaLiON MLX components.

Tests verify:
    1. Whisper encoder architecture matches expected shapes
    2. MLP adaptor compresses timesteps correctly
    3. Processor handles audio and text correctly
    4. Weight loading and partitioning works
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from meralion_mlx.whisper_encoder import WhisperEncoder, WhisperEncoderConfig
from meralion_mlx.adaptor import MERaLiONSpeechAudioAdapter
from meralion_mlx.model import partition_weights, remap_whisper_keys, remap_adaptor_keys


class TestWhisperEncoder:
    """Test Whisper encoder shapes and forward pass."""

    def setup_method(self):
        self.config = WhisperEncoderConfig(
            d_model=128,  # Small for testing
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=512,
            num_mel_bins=80,
            max_source_positions=1500,
        )
        self.encoder = WhisperEncoder(self.config)

    def test_output_shape(self):
        """Encoder output should be (batch, seq_len, d_model)."""
        # Input: (batch, n_mels, time)
        x = mx.random.normal((1, 80, 3000))
        out = self.encoder(x)
        mx.eval(out)

        assert out.ndim == 3
        assert out.shape[0] == 1  # batch
        assert out.shape[2] == 128  # d_model
        # Time dim is halved by stride-2 conv
        assert out.shape[1] == 1500

    def test_batch_processing(self):
        """Encoder should handle batch > 1."""
        x = mx.random.normal((2, 80, 3000))
        out = self.encoder(x)
        mx.eval(out)

        assert out.shape[0] == 2

    def test_positional_embeddings_shape(self):
        """Positional embeddings should be (max_positions, d_model)."""
        assert self.encoder.embed_positions.shape == (1500, 128)


class TestMERaLiONAdapter:
    """Test MLP adaptor timestep compression and projection."""

    def setup_method(self):
        self.adaptor = MERaLiONSpeechAudioAdapter(
            speech_hidden_size=1280,
            text_hidden_size=3584,
            scale_factor=15,
        )

    def test_output_shape(self):
        """Adaptor should compress 1500 → 100 timesteps and project to 3584."""
        x = mx.random.normal((1, 1500, 1280))
        out = self.adaptor(x)
        mx.eval(out)

        assert out.shape == (1, 100, 3584)

    def test_batch_processing(self):
        """Adaptor should handle batch > 1."""
        x = mx.random.normal((2, 1500, 1280))
        out = self.adaptor(x)
        mx.eval(out)

        assert out.shape == (2, 100, 3584)

    def test_truncation(self):
        """Adaptor should truncate input to multiple of scale_factor."""
        # 1507 → truncated to 1500 (floor to multiple of 15)
        x = mx.random.normal((1, 1507, 1280))
        out = self.adaptor(x)
        mx.eval(out)

        assert out.shape == (1, 100, 3584)

    def test_small_config(self):
        """Test with smaller dimensions for speed."""
        adaptor = MERaLiONSpeechAudioAdapter(
            speech_hidden_size=64,
            text_hidden_size=128,
            scale_factor=5,
        )
        x = mx.random.normal((1, 50, 64))
        out = adaptor(x)
        mx.eval(out)

        assert out.shape == (1, 10, 128)


class TestWeightPartitioning:
    """Test weight key partitioning logic."""

    def test_partition_by_prefix(self):
        """Weights should be split by prefix."""
        weights = {
            "speech_encoder.layers.0.weight": mx.zeros((10,)),
            "speech_encoder.layers.0.bias": mx.zeros((10,)),
            "ln_speech.weight": mx.zeros((10,)),
            "ln_speech.bias": mx.zeros((10,)),
            "speech_audio_adapter.mlp_adapter.0.weight": mx.zeros((10,)),
            "text_decoder.model.layers.0.weight": mx.zeros((10,)),
        }

        enc, ln, adapt, dec = partition_weights(weights)

        assert len(enc) == 2
        assert len(ln) == 2
        assert len(adapt) == 1
        assert len(dec) == 1

        # Check prefix stripping
        assert "layers.0.weight" in enc
        assert "weight" in ln
        assert "mlp_adapter.0.weight" in adapt
        assert "model.layers.0.weight" in dec

    def test_unknown_prefix_goes_to_decoder(self):
        """Weights without known prefix should go to decoder."""
        weights = {
            "some_unknown_key": mx.zeros((10,)),
        }
        _, _, _, dec = partition_weights(weights)
        assert "some_unknown_key" in dec


class TestWhisperKeyRemapping:
    """Test Whisper weight key remapping."""

    def test_strip_encoder_prefix(self):
        """Should strip 'encoder.' prefix from keys."""
        weights = {
            "encoder.layers.0.self_attn.q_proj.weight": mx.zeros((10, 10)),
        }
        remapped = remap_whisper_keys(weights)
        assert "layers.0.self_attn.q_proj.weight" in remapped

    def test_conv_transposition(self):
        """Conv1d weights should be transposed from HF to MLX format."""
        # HF: (out_channels, in_channels, kernel_size)
        weights = {
            "conv1.weight": mx.random.normal((1280, 80, 3)),
        }
        remapped = remap_whisper_keys(weights)

        # MLX: (out_channels, kernel_size, in_channels)
        assert remapped["conv1.weight"].shape == (1280, 3, 80)


class TestAdaptorKeyRemapping:
    """Test adaptor weight key remapping."""

    def test_sequential_prefix(self):
        """nn.Sequential keys should get 'layers.' prefix."""
        weights = {
            "mlp_adapter.0.weight": mx.zeros((10, 10)),
            "mlp_adapter.2.weight": mx.zeros((10, 10)),
            "speech_llm_proj.weight": mx.zeros((10, 10)),
        }
        remapped = remap_adaptor_keys(weights)

        assert "mlp_adapter.layers.0.weight" in remapped
        assert "mlp_adapter.layers.2.weight" in remapped
        assert "speech_llm_proj.weight" in remapped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
