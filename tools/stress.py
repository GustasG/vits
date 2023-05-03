from multiprocessing.pool import ThreadPool
import json
from typing import List, Iterator

import requests


def read_file(path: str) -> Iterator[str]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()


def write_file(path: str, lines: List[str]):
    text = '\n'.join(lines)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def stress_text(text: str) -> str:
    stressed_text = ''

    # made from https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/
    r = requests.post('https://kalbu.vdu.lt/ajax-call', data={
        'action': 'text_accents',
        'nonce': '417e0e9ad2',
        'body': text
    })

    r.raise_for_status()
    data = r.json()
    message = json.loads(data['message'])

    for part in message['textParts']:
        # view-source:https://kalbu.vdu.lt/core/views/d1e7573f87/assets/js/priemone1.js (function wordAccentProcess)
        if part['type'] == 'WORD':
            if part['accentType'] == 'NONE':
                stressed_text += part['string']
            else:
                stressed_text += part['accented']
        else:
            stressed_text += part['string']

    return stressed_text


def handle_line(line: str) -> str:
    if not line:
        return ''

    path, text = line.split('|')
    stressed_text = stress_text(text)

    return f'{path}|{stressed_text}'


def handle_file(path: str):
    stressed_file_path = path.replace('.txt', '_stressed.txt')
    lines = list(read_file(path))

    try:
        stressed_lines = list(read_file(stressed_file_path))
        lines = lines[len(stressed_lines):]
    except FileNotFoundError:
        stressed_lines = []

    with ThreadPool() as pool:
        try:
            for line in pool.map(handle_line, lines):
                stressed_lines.append(line)
        finally:
            write_file(stressed_file_path, stressed_lines)


def main():
    files = [
        'files/filelists/lt_text_test.txt',
        'files/filelists/lt_text_train.txt',
        'files/filelists/lt_text_val.txt'
    ]

    for file in files:
        handle_file(file)


if __name__ == '__main__':
    main()
