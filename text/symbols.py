from functools import cache
from typing import Tuple, Mapping, List


@cache
def get_vocabulary(language: str) -> Tuple[List[str], Mapping[str, int], Mapping[int, str]]:
    if language == 'lt':
        pad = '_'
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzĄČĘĖĮŠŲŪŽąčęėįšųūž'
        punctuation = '";:,.!?—-\'"() '
        accents = u'\u0300\u0301\u0303'

        symbols = [pad] + list(punctuation) + list(letters) + list(accents)
    else:
        raise NotImplementedError

    symbol_to_id = {
        symbol: i for i, symbol in enumerate(symbols)
    }

    id_to_symbol = {
        i: symbol for i, symbol in enumerate(symbols)
    }

    return symbols, symbol_to_id, id_to_symbol
