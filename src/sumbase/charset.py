from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
STRUCTURAL = ["<SURFACE>", "<COLUMN>", "<BLANK_SPACE>", "<RULING>", "...", " "]
_TOKEN_RE = re.compile(r"<[A-Za-z_]+>|\.\.\.|\s|.", re.S)


def split_tokens(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN_RE.findall(text):
        if tok.isspace():
            tok = " "
            if tokens and tokens[-1] == " ":
                continue
        elif tok in ("<UNK>", "<unk>"):
            tok = UNK
        tokens.append(tok)
    while tokens and tokens[0] == " ":
        tokens.pop(0)
    while tokens and tokens[-1] == " ":
        tokens.pop()
    return tokens


def canon(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for i, t in enumerate(tokens):
        if t == " ":
            prev = out[-1] if out else ""
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if not prev or not nxt or prev.startswith("<") or prev == "..." or nxt.startswith("<") or nxt == "...":
                continue
        out.append(t)
    return out


def glyph_cer_tokens(text: str) -> list[str]:
    return canon(split_tokens(text))


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = (self.stoi[t] for t in SPECIALS)

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1) -> "CharTokenizer":
        counts = Counter(t for text in texts for t in split_tokens(text))
        vocab = SPECIALS + [t for t in STRUCTURAL if t not in SPECIALS]
        vocab += sorted(t for t, c in counts.items() if c >= min_freq and t not in vocab)
        return cls(vocab)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in split_tokens(text)]
        if max_len is not None:
            ids = ids[: max_len - 1]
        return ids + [self.eos_id]

    def decode_tokens(self, ids) -> list[str]:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id) or i < 0 or i >= len(self.itos):
                continue
            out.append(self.itos[i])
        return out

    def decode(self, ids) -> str:
        return "".join(self.decode_tokens(ids))

    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(Path(directory) / "charset.json", "w", encoding="utf-8") as fh:
            json.dump(self.itos, fh, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str | Path) -> "CharTokenizer":
        with open(Path(directory) / "charset.json", encoding="utf-8") as fh:
            return cls(json.load(fh))
