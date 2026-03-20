"""Audiobench per-language text normalizers for ASR WER evaluation.

Adapted from Audiobench ``dataset_src/text_normalizer/preprocess_text.py``
(Bin Wang, 2024). Changes from the original:

* Absolute imports replaced with relative imports.
* Malay normalizer simplified — the original ``MalayCodeSwitchEnglishSpellingNormalizer``
  has a confirmed bug (``" ".join(tokens)`` on line 53 discards all per-token processing),
  making the ``malaya`` dependency dead code. The effective output is just
  ``remove_symbols_and_diacritics`` + whitespace cleanup, which is what we use here.
"""

import re

import jiwer

from .basic import remove_symbols_and_diacritics
from .whisper_english import EnglishTextNormalizer as _EnglishTextNormalizer


all_jiwer_process = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveKaldiNonWords(),
    jiwer.RemovePunctuation()
])

EnglishTextNormalizer = _EnglishTextNormalizer()


def normalize_text(text):
    """Normalize text by converting digits to words and expanding contractions."""
    digits_to_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen',
        '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
        '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty',
        '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety',
    }
    for digit, word in digits_to_words.items():
        text = re.sub(r'\b' + digit + r'\b', word, text)

    contractions = {
        "i'm": "i am", "you're": "you are", "he's": "he is",
        "she's": "she is", "it's": "it is", "we're": "we are",
        "they're": "they are", "i've": "i have", "you've": "you have",
        "we've": "we have", "they've": "they have", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
        "doesn't": "does not", "don't": "do not", "didn't": "did not",
        "that's": "that is",
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)

    return text


def remove_non_speech_elements(text):
    """Remove common non-speech elements like 'uh', 'um', etc."""
    return re.sub(r'\b(uh|umm|um|er|ah)\b', '', text)


def remove_parentheses(text):
    return re.sub(r'(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)', "", text)


def separate_and_space_chinese(text):
    """Split CJK runs into individual space-separated characters."""
    parts = re.split(r'([\u4e00-\u9fff]+)', text)
    processed_parts = []
    for part in parts:
        if re.match(r'[\u4e00-\u9fff]+', part):
            spaced = ' '.join(char for char in part)
            processed_parts.append(spaced)
        else:
            processed_parts.append(part)
    return ''.join(processed_parts)


def _malay_spelling_normalizer(text):
    """Simplified Malay normalizer (replaces MalayCodeSwitchEnglishSpellingNormalizer).

    The original Audiobench Malay normalizer has a bug on line 53 of whisper_malay.py:
    ``text = " ".join(tokens)`` joins the ORIGINAL tokens instead of ``processed_tokens``,
    discarding all ``malaya.dictionary.is_malay`` and ``malaya.word2num`` processing.
    The effective output is just diacritic stripping + whitespace cleanup.
    """
    text = remove_symbols_and_diacritics(text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Public normalizer functions (one per language / dataset family)
# ---------------------------------------------------------------------------

def preprocess_text_asr(text):
    """English normalizer (ytb_asr_batch1, ytb_asr_batch2)."""
    text = text.lower()
    text = EnglishTextNormalizer(text)
    text = normalize_text(text)
    text = remove_parentheses(text)
    text = all_jiwer_process(text)
    text = remove_non_speech_elements(text).strip()
    return text


def preprocess_text_asr_code_switch_chinese(text):
    """Chinese code-switch normalizer (ytb_asr_batch3_chinese)."""
    text = text.lower()
    text = EnglishTextNormalizer(text)
    text = normalize_text(text)
    text = remove_parentheses(text)
    text = all_jiwer_process(text)
    text = remove_non_speech_elements(text).strip()
    text = separate_and_space_chinese(text)
    return text


def preprocess_text_asr_malay(text):
    """Malay normalizer (ytb_asr_batch3_malay)."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # remove speaker tags
    text = _malay_spelling_normalizer(text)
    text = remove_parentheses(text)
    text = all_jiwer_process(text)
    text = remove_non_speech_elements(text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text_asr_tamil(text):
    """Tamil normalizer (ytb_asr_batch3_tamil_v2)."""
    text = text.lower()
    text = EnglishTextNormalizer(text)
    text = normalize_text(text)
    text = remove_parentheses(text)
    text = all_jiwer_process(text)
    text = remove_non_speech_elements(text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
