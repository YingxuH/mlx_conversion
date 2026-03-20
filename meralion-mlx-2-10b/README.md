# MERaLiON-MLX

Run [A*STAR's MERaLiON AudioLLM](https://huggingface.co/MERaLiON) natively on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

This package includes the pre-converted MLX model weights for MERaLiON-2-3B. Additional models can be downloaded and converted using the included scripts.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 16GB+ unified memory

## Setup

```bash
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
```

## Inference

### Speech-to-Text (ASR)

```bash
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio your_audio.wav \
  --task asr
```

### Translate to Chinese

```bash
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio your_audio.wav \
  --task translate_zh
```

### Spoken Question Answering

```bash
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio your_audio.wav \
  --task sqa \
  --question "What is the speaker talking about?"
```

### Dialogue Summarization

```bash
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio your_audio.wav \
  --task summarize
```

### Speaker Analysis (Paralinguistics)

```bash
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio your_audio.wav \
  --task paralinguistics
```

### Custom Instruction

```bash
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio your_audio.wav \
  --instruction "List the key points discussed."
```

### Long Audio

Audio of any length is automatically split into segments (default: 300s each).
Each segment's text is printed as it completes, followed by the combined full transcript.

```bash
# Transcribe a 30-minute recording
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio long_recording.wav \
  --task asr

# Use shorter segments for faster per-segment output
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio long_recording.wav \
  --task asr \
  --segment-length 60

# Only show the combined transcript (suppress per-segment output)
python scripts/inference.py \
  --model-dir models/2-3b-mlx \
  --audio long_recording.wav \
  --task asr \
  --no-show-segments
```

## Supported Tasks

| Task | Flag | Description |
|------|------|-------------|
| ASR | `--task asr` | Speech-to-text transcription |
| Translation | `--task translate_zh/id/ms/ta` | Speech translation to Chinese, Indonesian, Malay, Tamil |
| Spoken QA | `--task sqa --question "..."` | Answer questions about audio content |
| Summarization | `--task summarize` | Summarize spoken dialogue |
| Instruction | `--task instruction` | Follow speech instructions |
| Paralinguistics | `--task paralinguistics` | Analyze speaker characteristics |

## Other Models

The 2-3B model is included. To use other MERaLiON variants, download and convert them:

### Download

```bash
# List available models
python scripts/download.py --list

# Download a model
python scripts/download.py --model audiollm --output-dir models
python scripts/download.py --model 2-10b --output-dir models
```

### Convert to MLX Format

```bash
python scripts/convert.py --model-dir models/audiollm --output-dir models/audiollm-mlx
```

### Quantize (Optional)

Quantization reduces memory usage significantly — recommended for 10B models.

```bash
# 4-bit (~75% size reduction)
python scripts/quantize.py --model-dir models/audiollm-mlx --output-dir models/audiollm-mlx-4bit

# 8-bit (~50% size reduction)
python scripts/quantize.py --model-dir models/audiollm-mlx --output-dir models/audiollm-mlx-8bit --bits 8
```

### Available Models

| Key | Model | Params | Size | Description |
|-----|-------|--------|------|-------------|
| `audiollm` | MERaLiON-AudioLLM-Whisper-SEA-LION | 10B | ~20GB | Primary 6-task AudioLLM |
| `2-3b` | MERaLiON-2-3B | 3B | ~7GB | Smaller variant (included) |
| `2-10b` | MERaLiON-2-10B | 10B | ~20GB | Extended audio (300s) |
| `2-10b-asr` | MERaLiON-2-10B-ASR | 10B | ~20GB | ASR-optimized |
| `3-10b-preview` | MERaLiON-3-10B-preview | 10B | ~20GB | Latest preview |

### Memory Requirements

| Format | 10B Model | 3B Model |
|--------|-----------|----------|
| BF16 (full) | ~20GB | ~6GB |
| 8-bit | ~10GB | ~3GB |
| 4-bit | ~5GB | ~1.5GB |

## Architecture

```
Audio (16kHz) -> [Whisper Encoder] -> [LayerNorm] -> [MLP Adaptor] -> speech_embeds (100 x 3584)
                                                                            |
Text prompt  -> [Tokenizer] -> [Token Embedding] -> merge at <SpeechHere> positions
                                                            |
                                                 [Gemma2 Decoder] -> Generated text
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-tokens` | 256 | Maximum tokens to generate (per segment) |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `--max-audio-duration` | None | Truncate audio to N seconds (no limit by default) |
| `--segment-length` | 300.0 | Max seconds per inference segment |
| `--show-segments` | on | Print each segment's text as it completes |
| `--show-full` | on | Print combined transcript after all segments |

Use `--no-show-segments` or `--no-show-full` to suppress the corresponding output.

## Audio Input

- Formats: WAV, MP3, FLAC (anything librosa supports)
- Automatically resampled to 16kHz mono
- Any duration (automatically segmented for long audio)

## Troubleshooting

**`ModuleNotFoundError: No module named 'mlx'`**
You're not in the virtual environment. Run `source .venv/bin/activate` first.

**Intel Mac / non-macOS**
MLX only runs on Apple Silicon. There is no workaround for Intel Macs.

**Slow first run**
The first inference compiles MLX graphs and is slower than subsequent runs. This is normal.

**Out of memory**
The 3B model needs ~8GB. Close other large apps and try again. If using an 8GB Mac, add `--max-audio-duration 60` to limit input length.

**Segment boundary artifacts**
Long audio is split at segment boundaries (default 300s). Words at boundaries may occasionally be cut. Use `--segment-length 30` for shorter segments if needed.

## License

This toolkit is MIT licensed. The MERaLiON models are under the [MERaLiON Public License](https://huggingface.co/MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION/blob/main/MERaLiON-Public-Licence-v1.pdf).

## Acknowledgments

- [A*STAR I2R](https://www.a-star.edu.sg/i2r) — MERaLiON models
- [AI Singapore](https://aisingapore.org/) — SEA-LION language model
- [Apple MLX](https://github.com/ml-explore/mlx) — ML framework for Apple Silicon
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech encoder architecture
