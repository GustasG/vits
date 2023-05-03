from typing import Iterable, List

from text import cleaners
from text.symbols import get_vocabulary


def text_to_sequence(text: str, cleaner_names: Iterable[str], language: str) -> List[int]:
    text = clean_text(text, cleaner_names)

    return cleaned_text_to_sequence(text, language)


def cleaned_text_to_sequence(text: str, language: str) -> List[int]:
    _, symbol_to_id, _ = get_vocabulary(language)

    return [symbol_to_id[symbol] for symbol in text]


def clean_text(text: str, cleaner_names: Iterable[str]) -> str:
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        text = cleaner(text)

    return text
