#!/usr/bin/env python3
"""Run MERaLiON inference on Apple Silicon via MLX.

Usage:
    # Transcribe audio (ASR)
    python scripts/inference.py --model-dir models/audiollm-mlx --audio test.wav --task asr

    # Translate to Chinese
    python scripts/inference.py --model-dir models/audiollm-mlx --audio test.wav --task translate_zh

    # Spoken question answering
    python scripts/inference.py --model-dir models/audiollm-mlx --audio test.wav --task sqa --question "What is the speaker talking about?"

    # Custom instruction
    python scripts/inference.py --model-dir models/audiollm-mlx --audio test.wav --instruction "Summarize this in one sentence."

    # Paralinguistics
    python scripts/inference.py --model-dir models/audiollm-mlx --audio test.wav --task paralinguistics

    # Long audio (auto-segmented)
    python scripts/inference.py --model-dir models/audiollm-mlx --audio long_recording.wav --task asr
"""

import argparse
import json
import sys
import time
from collections import namedtuple
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from meralion_mlx.processor import MERaLiONProcessor, get_task_prompt, load_audio, SAMPLE_RATE
from meralion_mlx.whisper_encoder import WhisperEncoder, WhisperEncoderConfig
from meralion_mlx.adaptor import create_adapter

# ---------------------------------------------------------------------------
# N-gram blocking sampler
# ---------------------------------------------------------------------------

NO_REPEAT_NGRAM_SIZE = 6


def make_no_repeat_ngram_sampler(ngram_size: int = NO_REPEAT_NGRAM_SIZE):
    """Create a greedy sampler that blocks repeated n-grams.

    Instead of modifying logits (which breaks MLX's async GPU pipeline),
    this sampler does greedy argmax and falls back to the next-best token
    if the top choice would create a repeated n-gram.

    Returns a sampler compatible with mlx-lm's generate_step(sampler=...).
    """
    prefix_to_next: dict[tuple[int, ...], set[int]] = {}
    id_list: list[int] = []

    def _register(token: int):
        id_list.append(token)
        if len(id_list) >= ngram_size:
            prefix = tuple(id_list[-ngram_size:-1])
            if prefix not in prefix_to_next:
                prefix_to_next[prefix] = set()
            prefix_to_next[prefix].add(id_list[-1])

    def sampler(logits: mx.array) -> mx.array:
        flat = logits.reshape(-1) if logits.ndim == 2 else logits

        token = mx.argmax(flat)
        tid = int(token)

        # Check if this token creates a banned ngram
        if len(id_list) >= ngram_size - 1:
            ctx = tuple(id_list[-(ngram_size - 1):])
            banned = prefix_to_next.get(ctx)
            if banned and tid in banned:
                sorted_ids = mx.argsort(flat)[::-1]
                for candidate in sorted_ids:
                    cid = int(candidate)
                    if cid not in banned:
                        tid = cid
                        token = candidate
                        break

        _register(tid)
        return token.reshape(logits.shape[:-1]) if logits.ndim == 2 else token

    return sampler


def _wrap_sampler_with_ngram_blocking(base_sampler, ngram_size: int = NO_REPEAT_NGRAM_SIZE):
    """Wrap any sampler (including temperature>0) with n-gram blocking.

    For temperature=0 (base_sampler is None), uses greedy with blocking.
    For temperature>0, runs the base sampler then checks/rejects banned n-grams.
    """
    if base_sampler is None:
        return make_no_repeat_ngram_sampler(ngram_size)

    prefix_to_next: dict[tuple[int, ...], set[int]] = {}
    id_list: list[int] = []

    def _register(token: int):
        id_list.append(token)
        if len(id_list) >= ngram_size:
            prefix = tuple(id_list[-ngram_size:-1])
            if prefix not in prefix_to_next:
                prefix_to_next[prefix] = set()
            prefix_to_next[prefix].add(id_list[-1])

    def wrapped_sampler(logits: mx.array) -> mx.array:
        token = base_sampler(logits)
        tid = int(token.reshape(-1)[0]) if token.ndim > 0 else int(token)

        if len(id_list) >= ngram_size - 1:
            ctx = tuple(id_list[-(ngram_size - 1):])
            banned = prefix_to_next.get(ctx)
            if banned and tid in banned:
                flat = logits.reshape(-1) if logits.ndim == 2 else logits
                sorted_ids = mx.argsort(flat)[::-1]
                for candidate in sorted_ids:
                    cid = int(candidate)
                    if cid not in banned:
                        tid = cid
                        token = candidate.reshape(token.shape)
                        break

        _register(tid)
        return token

    return wrapped_sampler


def is_converted_dir(model_dir: Path) -> bool:
    """Check if model_dir contains converted MLX weights."""
    return (model_dir / "encoder_config.json").exists()


def is_raw_hf_dir(model_dir: Path) -> bool:
    """Check if model_dir contains raw HuggingFace model files."""
    return (
        (model_dir / "config.json").exists()
        and any(model_dir.glob("model-*.safetensors"))
    )


def auto_convert(model_dir: Path, verbose: bool = True) -> Path:
    """Auto-convert raw HF model to MLX format if needed.

    Returns the path to the converted model directory.
    """
    if is_converted_dir(model_dir):
        return model_dir

    if not is_raw_hf_dir(model_dir):
        print(f"Error: {model_dir} is neither a converted MLX dir nor a raw HF dir.")
        print("Expected either encoder_config.json (converted) or model-*.safetensors (raw).")
        sys.exit(1)

    # Auto-convert to sibling directory
    converted_dir = model_dir.parent / f"{model_dir.name}-mlx"
    if is_converted_dir(converted_dir):
        if verbose:
            print(f"Using existing converted model: {converted_dir}")
        return converted_dir

    if verbose:
        print(f"Raw HF model detected. Auto-converting to {converted_dir}...")

    from meralion_mlx.model import (
        load_config, load_weights, partition_weights,
        remap_whisper_keys, remap_adaptor_keys, save_component_weights,
    )
    from scripts.convert import (
        convert_encoder, convert_adaptor, convert_decoder, copy_tokenizer,
    )

    converted_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(model_dir)

    # Save full config
    with open(converted_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    weights = load_weights(model_dir)
    encoder_w, ln_w, adaptor_w, decoder_w = partition_weights(weights)

    convert_encoder(encoder_w, converted_dir, config)
    convert_adaptor(adaptor_w, ln_w, converted_dir, config)
    convert_decoder(decoder_w, converted_dir, config)
    copy_tokenizer(model_dir, converted_dir)

    if verbose:
        print(f"Auto-conversion complete: {converted_dir}\n")

    return converted_dir


def load_encoder(model_dir: Path) -> WhisperEncoder:
    """Load converted Whisper encoder."""
    config_path = model_dir / "encoder_config.json"
    weights_path = model_dir / "encoder.safetensors"

    with open(config_path) as f:
        config_dict = json.load(f)

    config = WhisperEncoderConfig.from_dict(config_dict)
    encoder = WhisperEncoder(config)

    weights = mx.load(str(weights_path))
    encoder.load_weights(list(weights.items()))
    mx.eval(encoder.parameters())

    return encoder


def detect_adaptor_variant(adaptor_weights: dict) -> str:
    """Detect whether weights are v1 (simple) or v2 (gated) adaptor."""
    keys = set(adaptor_weights.keys())
    if any("gate_proj" in k for k in keys):
        return "v2"
    return "v1"


def load_adaptor(model_dir: Path) -> tuple:
    """Load converted MLP adaptor and LayerNorm.

    Auto-detects v1 vs v2 adaptor architecture from weight keys.
    """
    import mlx.nn as nn

    config_path = model_dir / "adaptor_config.json"
    weights_path = model_dir / "adaptor.safetensors"

    with open(config_path) as f:
        config = json.load(f)

    all_weights = mx.load(str(weights_path))

    # Separate ln_speech weights from adaptor weights
    adaptor_weights = {}
    ln_weights = {}
    for key, value in all_weights.items():
        if key.startswith("ln_speech."):
            ln_weights[key[len("ln_speech."):]] = value
        else:
            adaptor_weights[key] = value

    # Auto-detect variant from weight keys
    variant = detect_adaptor_variant(adaptor_weights)
    adaptor = create_adapter(
        variant=variant,
        speech_hidden_size=config["speech_hidden_size"],
        text_hidden_size=config["text_hidden_size"],
        scale_factor=config["scale_factor"],
    )

    ln_speech = nn.LayerNorm(config["speech_hidden_size"])

    adaptor.load_weights(list(adaptor_weights.items()))
    ln_speech.load_weights(list(ln_weights.items()))
    mx.eval(adaptor.parameters())
    mx.eval(ln_speech.parameters())

    return adaptor, ln_speech


def load_decoder(model_dir: Path):
    """Load text decoder using mlx-lm.

    This leverages mlx-lm's Gemma2 model loading, which handles
    the full transformer decoder architecture.

    Returns:
        (model, tokenizer) from mlx-lm
    """
    try:
        from mlx_lm import load as mlx_lm_load

        decoder_config_path = model_dir / "decoder_config.json"
        if not decoder_config_path.exists():
            raise FileNotFoundError(
                f"decoder_config.json not found in {model_dir}. "
                "Run convert.py first."
            )

        # mlx-lm expects a directory with config.json + weights
        # Create a temporary symlink structure if needed
        decoder_dir = model_dir / "decoder"
        decoder_dir.mkdir(exist_ok=True)

        # Copy decoder config as the main config for mlx-lm
        import shutil

        shutil.copy2(decoder_config_path, decoder_dir / "config.json")

        # Symlink decoder weights
        for f in model_dir.glob("decoder*.safetensors"):
            target = decoder_dir / f.name.replace("decoder-", "model-")
            if not target.exists():
                target.symlink_to(f.resolve())

        # Copy tokenizer files
        for f in model_dir.glob("tokenizer*"):
            target = decoder_dir / f.name
            if not target.exists():
                shutil.copy2(f, target)

        for f in model_dir.glob("special_tokens*"):
            target = decoder_dir / f.name
            if not target.exists():
                shutil.copy2(f, target)

        model, tokenizer = mlx_lm_load(str(decoder_dir))
        return model, tokenizer

    except ImportError:
        print("Error: mlx-lm is required for decoder loading.")
        print("Install with: pip install mlx-lm")
        sys.exit(1)


def patch_decoder_for_embeddings(decoder_model):
    """Patch mlx-lm Gemma2 model to accept input_embeddings parameter.

    This enables use with mlx_lm.generate.generate_step(), which passes
    input_embeddings as a keyword argument. The patch adds a code path
    that uses provided embeddings instead of calling embed_tokens, while
    preserving ALL model internals: sqrt scaling, mask creation, attention
    softcapping, layer iteration, final logit softcapping.
    """
    from mlx_lm.models.base import create_attention_mask

    inner = decoder_model.model  # GemmaModel instance

    def patched_inner_call(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size ** 0.5)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0], return_array=True)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)

    def patched_outer_call(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        out = self.model.embed_tokens.as_linear(out)
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
        return out

    # Patch at the CLASS level — Python's () operator uses type(obj).__call__
    type(inner).__call__ = patched_inner_call
    type(decoder_model).__call__ = patched_outer_call


def build_merged_embeddings(
    decoder_model,
    input_ids: mx.array,
    speech_embeds: mx.array,
    speech_token_index: int,
) -> mx.array:
    """Build merged text+speech embeddings (UNscaled).

    Returns UNscaled embeddings because the model's __call__ applies
    sqrt(hidden_size) scaling internally after embed_tokens.

    Args:
        decoder_model: mlx-lm model with embed_tokens
        input_ids: (B, S) token IDs with speech_token_index at speech positions
        speech_embeds: (B, N_speech, H) speech embeddings from adaptor
        speech_token_index: token ID marking speech positions

    Returns:
        (B, S, H) merged embeddings ready for model forward pass
    """
    embed_fn = decoder_model.model.embed_tokens
    text_embeds = embed_fn(input_ids)  # (B, S, H) — UNscaled from embedding table

    B, S, H = text_embeds.shape

    for b in range(B):
        speech_idx = 0
        for s in range(S):
            if int(input_ids[b, s]) == speech_token_index and speech_idx < speech_embeds.shape[1]:
                text_embeds = text_embeds.at[b, s].add(
                    speech_embeds[b, speech_idx] - text_embeds[b, s]
                )
                speech_idx += 1

    return text_embeds


# ---------------------------------------------------------------------------
# Model loading and segment-level inference
# ---------------------------------------------------------------------------

LoadedModel = namedtuple("LoadedModel", [
    "encoder", "adaptor", "ln_speech", "decoder", "processor",
])


def load_model(model_dir: Path, verbose: bool = True) -> LoadedModel:
    """Load all model components once.

    Returns a LoadedModel namedtuple that can be reused across segments.
    """
    model_dir = auto_convert(model_dir, verbose=verbose)

    if verbose:
        print("Loading model components...")

    t0 = time.time()
    encoder = load_encoder(model_dir)
    if verbose:
        print(f"  Encoder loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    adaptor, ln_speech = load_adaptor(model_dir)
    if verbose:
        print(f"  Adaptor loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    decoder, _ = load_decoder(model_dir)
    if verbose:
        print(f"  Decoder loaded in {time.time() - t0:.1f}s")

    patch_decoder_for_embeddings(decoder)
    processor = MERaLiONProcessor(model_dir)

    return LoadedModel(encoder, adaptor, ln_speech, decoder, processor)


def _infer_segment(
    model: LoadedModel,
    audio_array: np.ndarray,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    verbose: bool = True,
) -> str:
    """Run inference on a single audio segment.

    The audio_array is split into 30s chunks internally by prepare_audio().
    Each chunk produces 100 speech tokens. Up to 10 chunks (300s) per segment.
    """
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    # Prepare audio (chunked into 30s pieces internally)
    mel_features, num_chunks = model.processor.prepare_audio(
        audio_array=audio_array, max_duration=None,
    )
    mel_mx = mx.array(mel_features)
    if verbose:
        print(f"  Mel: {mel_features.shape} ({num_chunks} chunk{'s' if num_chunks > 1 else ''})")

    # Prepare text (expands <SpeechHere> to 100 * num_chunks positions)
    text_inputs = model.processor.prepare_text(instruction, num_chunks=num_chunks)
    input_ids = mx.array(text_inputs["input_ids"])

    # Encode speech — process each chunk through encoder → adaptor
    t0 = time.time()
    chunk_embeds = []
    for i in range(num_chunks):
        chunk_mel = mel_mx[i : i + 1]  # (1, n_mels, 3000)
        enc_out = model.encoder(chunk_mel)
        enc_out = model.ln_speech(enc_out)
        chunk_speech = model.adaptor(enc_out)  # (1, 100, H)
        chunk_embeds.append(chunk_speech)
    speech_embeds = mx.concatenate(chunk_embeds, axis=1)  # (1, 100*num_chunks, H)
    mx.eval(speech_embeds)
    if verbose:
        print(f"  Encoded {num_chunks} chunk(s) in {time.time() - t0:.2f}s")

    # Build merged embeddings (UNscaled — model applies sqrt internally)
    t0 = time.time()
    merged_embeds = build_merged_embeddings(
        model.decoder, input_ids, speech_embeds, model.processor.speech_token_index,
    )
    mx.eval(merged_embeds)

    # Generate using mlx-lm's native generate_step with input_embeddings
    prompt_tokens = input_ids[0]  # (S,) — 1D for generate_step
    embeddings_2d = merged_embeds[0]  # (S, H) — 2D for generate_step

    # N-gram blocking always enabled (matches HF generation_config.json)
    base_sampler = make_sampler(temp=temperature) if temperature > 0 else None
    sampler = _wrap_sampler_with_ngram_blocking(base_sampler, NO_REPEAT_NGRAM_SIZE)

    # EOS tokens for Gemma2: <eos>=1, <end_of_turn>=107
    eos_tokens = {1, 107}
    if hasattr(model.processor.tokenizer, 'eos_token_id') and model.processor.tokenizer.eos_token_id is not None:
        eos_tokens.add(model.processor.tokenizer.eos_token_id)

    generated_tokens = []
    for token, logprobs in generate_step(
        prompt=prompt_tokens,
        model=model.decoder,
        max_tokens=max_new_tokens,
        sampler=sampler,
        input_embeddings=embeddings_2d,
    ):
        token_id = token.item() if hasattr(token, 'item') else int(token)
        if token_id in eos_tokens:
            break
        generated_tokens.append(token_id)

    gen_time = time.time() - t0
    response = model.processor.decode(generated_tokens)

    if verbose:
        n_tokens = len(generated_tokens)
        tps = n_tokens / gen_time if gen_time > 0 else 0
        print(f"  Generated {n_tokens} tokens in {gen_time:.2f}s ({tps:.1f} tok/s)")

    return response


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def run_inference(
    model_dir: Path,
    audio_path: str,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    max_audio_duration: float | None = None,
    segment_length: float = 300.0,
    show_segments: bool = True,
    show_full: bool = True,
    verbose: bool = True,
) -> str:
    """Run full MERaLiON inference pipeline with automatic segmentation.

    For audio longer than segment_length, the audio is split into
    independent segments. Each segment is processed through the full
    encode-merge-generate pipeline. Model components are loaded once
    and reused across all segments.

    Args:
        model_dir: Path to converted MLX model directory
        audio_path: Path to input audio file
        instruction: Text instruction for the model
        max_new_tokens: Maximum generated tokens (per segment)
        temperature: Sampling temperature
        max_audio_duration: Truncate audio to this many seconds (None = no limit)
        segment_length: Maximum seconds per inference segment (default: 300)
        show_segments: Print per-segment text as each completes
        show_full: Print combined text after all segments complete
        verbose: Print timing information

    Returns:
        Tuple of (generated_text, num_segments)
    """
    t_start = time.time()

    # Load model components once
    model = load_model(model_dir, verbose=verbose)

    # Load full audio
    if verbose:
        print(f"\nProcessing audio: {audio_path}")
    audio = load_audio(audio_path)
    total_duration = len(audio) / SAMPLE_RATE

    # Truncate if max_audio_duration is set
    if max_audio_duration is not None:
        max_samples = int(max_audio_duration * SAMPLE_RATE)
        audio = audio[:max_samples]
        total_duration = len(audio) / SAMPLE_RATE

    if verbose:
        print(f"  Duration: {_format_time(total_duration)} ({total_duration:.1f}s)")

    # Split audio into segments
    segment_samples = int(segment_length * SAMPLE_RATE)
    num_segments = max(1, -(-len(audio) // segment_samples))  # ceil division

    if num_segments == 1:
        # Single segment — identical to original behavior
        response = _infer_segment(
            model, audio, instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        if verbose:
            print(f"  Total pipeline: {time.time() - t_start:.2f}s")
        return response, 1

    # Multi-segment processing
    if verbose:
        print(f"  Splitting into {num_segments} segments"
              f" (up to {_format_time(segment_length)} each)")

    results = []
    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_samples
        seg_end = min(seg_start + segment_samples, len(audio))
        seg_audio = audio[seg_start:seg_end]

        start_time = seg_start / SAMPLE_RATE
        end_time = seg_end / SAMPLE_RATE
        header = (f"[Segment {seg_idx + 1}/{num_segments}"
                  f" | {_format_time(start_time)}\u2013{_format_time(end_time)}]")

        if verbose or show_segments:
            print(f"\n{header}")

        text = _infer_segment(
            model, seg_audio, instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        results.append(text)

        if show_segments:
            print(text)

    combined = " ".join(results)

    if verbose:
        print(f"\n  Total pipeline: {time.time() - t_start:.2f}s"
              f" ({num_segments} segments)")

    if show_full:
        print(f"\n{'=' * 60}")
        print(f"Full transcript:\n{combined}")
        print(f"{'=' * 60}")

    return combined, num_segments


def main():
    parser = argparse.ArgumentParser(
        description="MERaLiON inference on Apple Silicon via MLX"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to converted MLX model directory",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file (WAV, MP3, FLAC, etc.)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Predefined task: asr, translate_zh, translate_id, sqa, summarize, "
        "instruction, paralinguistics",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Custom text instruction (overrides --task)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for SQA task",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per segment (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy, default: 0.0)",
    )
    parser.add_argument(
        "--max-audio-duration",
        type=float,
        default=None,
        help="Truncate audio to this many seconds (default: no limit)",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=300.0,
        help="Maximum seconds per inference segment (default: 300). "
        "Audio longer than this is split into multiple segments.",
    )
    parser.add_argument(
        "--show-segments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each segment's text as it completes (default: on)",
    )
    parser.add_argument(
        "--show-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print combined full text after all segments (default: on)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Validate segment-length
    if args.segment_length <= 0:
        print("Error: --segment-length must be > 0")
        sys.exit(1)

    # Determine instruction
    if args.instruction:
        instruction = args.instruction
    elif args.task:
        kwargs = {}
        if args.task == "sqa":
            if not args.question:
                print("Error: --question is required for SQA task")
                sys.exit(1)
            kwargs["question"] = args.question
        instruction = get_task_prompt(args.task, **kwargs)
    else:
        instruction = get_task_prompt("asr")  # Default to ASR

    if not args.quiet:
        print(f"Task instruction: {instruction}")

    # Run inference
    response, num_segments = run_inference(
        model_dir=args.model_dir,
        audio_path=args.audio,
        instruction=instruction,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        max_audio_duration=args.max_audio_duration,
        segment_length=args.segment_length,
        show_segments=args.show_segments,
        show_full=args.show_full,
        verbose=not args.quiet,
    )

    # For single-segment audio, print the response here.
    # Multi-segment output is handled inside run_inference.
    if num_segments == 1:
        print(f"\n{'=' * 60}")
        print(f"Response:\n{response}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
