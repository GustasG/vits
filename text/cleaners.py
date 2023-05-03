def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return text.replace(' ', '')


def basic_cleaners(text: str) -> str:
    text = lowercase(text)
    text = collapse_whitespace(text)

    return text
