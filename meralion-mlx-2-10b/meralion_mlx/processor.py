"""Audio processor for MERaLiON MLX.

Handles:
    1. Audio loading and resampling to 16kHz mono
    2. Log-Mel spectrogram extraction (matching WhisperFeatureExtractor)
    3. Text tokenization with speech token expansion
    4. Chat template formatting
"""

import json
from pathlib import Path

import numpy as np

# Whisper-compatible mel filterbank and spectrogram
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000

SPEECH_TOKEN = "<SpeechHere>"
SPEECH_TOKEN_INDEX = 255999
FIXED_SPEECH_LENGTH = 100


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file and resample to target rate.

    Args:
        path: Path to audio file
        sr: Target sample rate (default: 16000)

    Returns:
        1D float32 numpy array of audio samples
    """
    import librosa

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Compute Whisper-compatible mel filterbank.

    Uses the same mel scale as OpenAI Whisper.
    """
    import librosa

    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float32)


def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = N_MELS,
    padding: int = 0,
) -> np.ndarray:
    """Compute log-Mel spectrogram matching Whisper's preprocessing.

    Args:
        audio: 1D float32 array of audio samples at 16kHz
        n_mels: Number of mel bins (default: 80)
        padding: Padding samples to add

    Returns:
        (n_mels, time) float32 log-mel spectrogram
    """
    import librosa

    if padding > 0:
        audio = np.pad(audio, (0, padding))

    # STFT with Whisper's parameters
    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    magnitudes = np.abs(stft) ** 2

    # Mel filterbank
    filters = mel_filterbank(n_mels=n_mels)
    mel_spec = filters @ magnitudes

    # Log scale (clamp to avoid log(0))
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.astype(np.float32)


class MERaLiONProcessor:
    """Combined audio + text processor for MERaLiON inference.

    Handles:
    - Audio: load → resample → log-Mel spectrogram
    - Text: tokenize with speech token expansion
    - Chat template: format prompts for SEA-LION decoder
    """

    def __init__(self, model_dir: str | Path):
        """Load tokenizer and config from model directory.

        Args:
            model_dir: Path to downloaded MERaLiON model directory
        """
        model_dir = Path(model_dir)

        # Load tokenizer via transformers (for chat template support)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=False
        )

        # Load WhisperFeatureExtractor for audio preprocessing
        # This ensures exact match with model training pipeline
        from transformers import WhisperFeatureExtractor

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            str(model_dir)
        )

        # Load config for speech token index
        config_path = model_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        self.speech_token_index = config.get("speech_token_index", SPEECH_TOKEN_INDEX)
        self.fixed_speech_length = FIXED_SPEECH_LENGTH
        self.n_mels = config.get("speech_config", {}).get("num_mel_bins", N_MELS)

    def prepare_audio(
        self,
        audio_path: str | None = None,
        audio_array: np.ndarray | None = None,
        max_duration: float | None = 30.0,
    ) -> tuple[np.ndarray, int]:
        """Process audio to log-Mel spectrogram using HF WhisperFeatureExtractor.

        Long audio is split into 30-second chunks (matching MERaLiON-2's
        chunked encoding). Each chunk is independently feature-extracted
        and padded to 3000 frames.

        Args:
            audio_path: Path to audio file
            audio_array: Pre-loaded audio array (16kHz, mono)
            max_duration: Maximum audio duration in seconds (None for no limit)

        Returns:
            Tuple of:
                (num_chunks, n_mels, 3000) float32 log-Mel spectrograms
                num_chunks: number of 30s chunks
        """
        if audio_array is None:
            if audio_path is None:
                raise ValueError("Provide either audio_path or audio_array")
            audio_array = load_audio(audio_path)

        # Truncate if needed
        if max_duration is not None:
            max_samples = int(max_duration * SAMPLE_RATE)
            audio_array = audio_array[:max_samples]

        # Split into 30s chunks (matching MERaLiON-2's chunked encoding)
        chunk_samples = CHUNK_LENGTH * SAMPLE_RATE  # 480000
        num_chunks = max(1, -(-len(audio_array) // chunk_samples))  # ceil division
        chunks = []
        for i in range(num_chunks):
            chunk = audio_array[i * chunk_samples : (i + 1) * chunk_samples]
            chunks.append(chunk)

        # Use HF WhisperFeatureExtractor for exact match with training pipeline
        features = self.feature_extractor(
            chunks,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            return_attention_mask=True,
            padding="max_length",
            do_normalize=True,
        )

        return features["input_features"], num_chunks  # (num_chunks, n_mels, 3000)

    def prepare_text(
        self,
        instruction: str,
        system_prompt: str | None = None,
        num_chunks: int = 1,
    ) -> dict:
        """Format text prompt with speech token expansion.

        The MERaLiON prompt format (matches HF training):
            "Instruction: {instruction} \\n
             Follow the text instruction based on the following audio: <SpeechHere>"

        The <SpeechHere> token gets expanded to `fixed_speech_length * num_chunks`
        (100 per chunk) copies of speech_token_index (255999).

        Args:
            instruction: User's text instruction (e.g., "Transcribe this speech")
            system_prompt: Optional system prompt

        Returns:
            Dict with 'input_ids' and 'attention_mask' as numpy arrays
        """
        prompt = (
            f"Instruction: {instruction} \n"
            f"Follow the text instruction based on the following audio: {SPEECH_TOKEN}"
        )

        conversation = [{"role": "user", "content": prompt}]
        if system_prompt:
            conversation.insert(0, {"role": "system", "content": system_prompt})

        chat_text = self.tokenizer.apply_chat_template(
            conversation=[conversation],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        if isinstance(chat_text, list):
            chat_text = chat_text[0]

        encoded = self.tokenizer(chat_text, return_tensors="np")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Expand speech token: replace single speech_token_index with
        # fixed_speech_length copies
        expanded_ids = []
        expanded_mask = []
        for i in range(input_ids.shape[0]):
            ids = input_ids[i]
            mask = attention_mask[i]
            new_ids = []
            new_mask = []
            for j, token_id in enumerate(ids):
                if token_id == self.speech_token_index:
                    total_speech_tokens = self.fixed_speech_length * num_chunks
                    new_ids.extend([self.speech_token_index] * total_speech_tokens)
                    new_mask.extend([1] * total_speech_tokens)
                else:
                    new_ids.append(int(token_id))
                    new_mask.append(int(mask[j]))
            expanded_ids.append(new_ids)
            expanded_mask.append(new_mask)

        # Pad to same length within batch
        max_len = max(len(ids) for ids in expanded_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        for i in range(len(expanded_ids)):
            pad_len = max_len - len(expanded_ids[i])
            expanded_ids[i].extend([pad_id] * pad_len)
            expanded_mask[i].extend([0] * pad_len)

        return {
            "input_ids": np.array(expanded_ids, dtype=np.int32),
            "attention_mask": np.array(expanded_mask, dtype=np.int32),
        }

    def decode(self, token_ids: np.ndarray, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, token_ids: np.ndarray, skip_special_tokens: bool = True
    ) -> list[str]:
        """Decode batch of token IDs."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )


# Task-specific prompt templates
TASK_PROMPTS = {
    "asr": "Please transcribe this speech.",
    "asr_singlish": "Please transcribe this Singlish speech.",
    "translate_zh": "Can you please translate this speech into written Chinese?",
    "translate_id": "Can you please translate this speech into written Indonesian?",
    "translate_ms": "Can you please translate this speech into written Malay?",
    "translate_ta": "Can you please translate this speech into written Tamil?",
    "sqa": "Based on the audio, answer the following question: {question}",
    "summarize": "Please summarize this spoken dialogue.",
    "instruction": "Follow the instruction given in the speech.",
    "paralinguistics": "Describe the speaker's characteristics including gender, accent, and emotion.",
}


def get_task_prompt(task: str, **kwargs) -> str:
    """Get the prompt template for a specific task.

    Args:
        task: Task key from TASK_PROMPTS
        **kwargs: Format arguments for the template

    Returns:
        Formatted prompt string
    """
    if task not in TASK_PROMPTS:
        raise ValueError(
            f"Unknown task '{task}'. Available: {list(TASK_PROMPTS.keys())}"
        )
    return TASK_PROMPTS[task].format(**kwargs)
