from __future__ import annotations

import re

_WS = re.compile(r"[ \t]+")

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
_ASCII_MAP = [
    ("š", "sz"), ("Š", "SZ"),
    ("ŋ", "j"), ("Ŋ", "J"), ("ĝ", "j"), ("Ĝ", "J"),
    ("ḫ", "h"), ("Ḫ", "H"),
    ("ṣ", "s,"), ("Ṣ", "S,"),
    ("ṭ", "t,"), ("Ṭ", "T,"),
    ("ś", "s'"), ("Ś", "S'"),
    ("ʾ", "'"),
]


def clean_ws(text: str) -> str:
    lines = [_WS.sub(" ", line).strip() for line in str(text).split("\n")]
    return "\n".join(line for line in lines if line)


def protect_markers(text: str) -> str:
    return text.replace("<unk>", "<UNK>")


def to_cdli_ascii(text: str) -> str:
    text = text.translate(_SUBSCRIPTS)
    for src, dst in _ASCII_MAP:
        text = text.replace(src, dst)
    return text
