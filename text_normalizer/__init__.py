"""Audiobench-compatible text normalizers for ASR WER evaluation."""

from text_normalizer.preprocess_text import (
    preprocess_text_asr,
    preprocess_text_asr_code_switch_chinese,
    preprocess_text_asr_malay,
    preprocess_text_asr_tamil,
)

__all__ = [
    "preprocess_text_asr",
    "preprocess_text_asr_code_switch_chinese",
    "preprocess_text_asr_malay",
    "preprocess_text_asr_tamil",
]
