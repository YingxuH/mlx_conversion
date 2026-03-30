# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains MLX-native implementations of [MERaLiON AudioLLM](https://huggingface.co/MERaLiON) — A*STAR's multimodal audio-language model — optimized for Apple Silicon. Two packages are provided: `meralion-mlx-2-3b` and `meralion-mlx-2-10b`. They share identical code structure; only the bundled model weights differ.

## Setup

Each package has its own virtual environment:

```bash
cd meralion-mlx-2-3b   # or meralion-mlx-2-10b
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv/` and installs the package in editable mode (`pip install -e .`). Requires macOS Apple Silicon and Python 3.10+.

## Running Tests

```bash
cd meralion-mlx-2-3b
source .venv/bin/activate
pytest tests/test_components.py -v

# Run a single test class
pytest tests/test_components.py::TestWhisperEncoder -v

# Run a single test
pytest tests/test_components.py::TestMERaLiONAdapter::test_output_shape -v
```

## Inference

```bash
python scripts/inference.py --model-dir models/2-3b-mlx --audio example_25s.wav --task asr
```

If a raw HuggingFace model directory is passed instead of a converted MLX dir, inference auto-converts it first (detected by presence of `encoder_config.json`).

## Model Preparation Scripts

```bash
# Download models from HuggingFace
python scripts/download.py --list
python scripts/download.py --model 2-10b --output-dir models

# Convert raw HF weights to MLX component format
python scripts/convert.py --model-dir models/2-3b --output-dir models/2-3b-mlx

# Quantize (recommended for 10B models)
python scripts/quantize.py --model-dir models/2-3b-mlx --output-dir models/2-3b-mlx-4bit
python scripts/quantize.py --model-dir models/2-3b-mlx --output-dir models/2-3b-mlx-8bit --bits 8
```

## Architecture

The model is split into three separately-stored components:

```
Audio (16kHz WAV/MP3/FLAC)
  └─> librosa → 30s chunks → HF WhisperFeatureExtractor → (N_chunks, 80, 3000) mel spectrograms
        └─> WhisperEncoder (encoder.safetensors) → (N_chunks, 1500, 1280)
              └─> LayerNorm (ln_speech) → MERaLiONSpeechAudioAdapter (adaptor.safetensors)
                    └─> (N_chunks, 100, hidden_size) speech_embeds
                          └─> merged into text token sequence at <SpeechHere> positions
                                └─> Gemma2 decoder (decoder-*.safetensors via mlx-lm) → text
```

**Key implementation details:**

- **`meralion_mlx/whisper_encoder.py`** — Custom MLX Whisper encoder. Conv1d weights are transposed from HF format `(out, in, kernel)` → MLX format `(out, kernel, in)` during conversion (`remap_whisper_keys`).
- **`meralion_mlx/adaptor.py`** — Two variants: `MERaLiONSpeechAudioAdapter` (v1, simple MLP) and `MERaLiONSpeechAudioAdapterLarge` (v2, gated MLP). The variant is auto-detected from weight keys at load time (presence of `gate_proj`).
- **`meralion_mlx/model.py`** — Weight loading utilities: `partition_weights` splits the monolithic HF checkpoint by key prefix; `remap_whisper_keys` and `remap_adaptor_keys` handle key format differences between HF and MLX.
- **`meralion_mlx/processor.py`** — `MERaLiONProcessor` handles audio and text prep. Uses HF `WhisperFeatureExtractor` for exact mel spectrogram match with training. Prompt format matches HF training: `"Instruction: {task} \nFollow the text instruction based on the following audio: <SpeechHere>"`. Expands the single `<SpeechHere>` token (index 255999) to `100 × num_chunks` copies in the input_ids before embedding.
- **`scripts/inference.py`** — Main pipeline. Patches the mlx-lm Gemma2 model at the class level (`patch_decoder_for_embeddings`) to accept `input_embeddings` in `generate_step`, bypassing `embed_tokens` for the merged speech+text embeddings. Long audio is split into `--segment-length` second segments (default 300s), each processed independently.

**Converted model directory layout** (what `encoder_config.json` signals):
```
models/2-3b-mlx/
  encoder_config.json       # Whisper encoder hyperparams
  encoder.safetensors       # Whisper encoder weights
  adaptor_config.json       # Adaptor dims + scale_factor
  adaptor.safetensors       # MLP adaptor + ln_speech weights
  decoder_config.json       # Gemma2 config (used by mlx-lm)
  decoder-00000.safetensors # Gemma2 decoder weights (sharded if >4GB)
  decoder.safetensors.index.json  # Present if sharded
  config.json               # Full MERaLiON config
  tokenizer.*               # HF tokenizer files
```

The `scripts/inference.py::load_decoder` function creates a `models/2-3b-mlx/decoder/` subdirectory with symlinks and copies to satisfy mlx-lm's expected layout.

## ASR Evaluation

```bash
# Full eval, 10B 8-bit model (recommended)
python client_eval/eval_mlx_asr.py --model-size 10b-8bit

# Quick smoke test
python client_eval/eval_mlx_asr.py --model-size 10b-8bit --max-samples 8

# Specific datasets
python client_eval/eval_mlx_asr.py --model-size 10b-8bit --datasets ytb_asr_batch2 ytb_asr_batch3_chinese
```

Per-sample JSONL outputs are saved to `eval_outputs/`. See `RESULTS.md` for full evaluation findings.

**Critical implementation notes:**
- **Prompt format**: Must use `"Instruction: {task} \nFollow ... <SpeechHere>"` (set in `processor.py`). The old `"Given the following audio context: <SpeechHere>..."` format gives worse WER.
- **No-repeat n-gram**: Must be implemented as a custom **sampler**, not a logits processor. Any Python-side `mx.array` manipulation in the `generate_step` hot loop (including `np.array(logits)` or `logits[idx] = val`) breaks MLX's async GPU pipeline, causing 100-1000x slowdowns.
- **Smart chunking**: Long audio split at 30s boundaries; last segment merged into previous if <10s to prevent hallucination on mostly-silent padded chunks.

## HuggingFace Model Repos

- 10B 8-bit MLX: https://huggingface.co/YingxuHe/mlx_test_10b
- 3B fp16 MLX: https://huggingface.co/YingxuHe/mlx_test_3b

## Supported Tasks

| Flag | Description |
|------|-------------|
| `--task asr` | Speech-to-text transcription |
| `--task translate_zh/id/ms/ta` | Translation to Chinese/Indonesian/Malay/Tamil |
| `--task sqa --question "..."` | Spoken question answering |
| `--task summarize` | Dialogue summarization |
| `--task paralinguistics` | Speaker characteristic analysis |
| `--instruction "..."` | Custom free-form instruction |
