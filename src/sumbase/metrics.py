from __future__ import annotations

from typing import Sequence

import sacrebleu
from rapidfuzz.distance import Levenshtein


def cer(preds: Sequence[Sequence], refs: Sequence[Sequence]) -> float:
    edits = sum(Levenshtein.distance(p, r) for p, r in zip(preds, refs))
    total = sum(len(r) for r in refs)
    return 100.0 * edits / max(total, 1)


def text_metrics(preds: list[str], refs: list[str]) -> dict:
    preds = [p.strip() for p in preds]
    refs = [r.strip() for r in refs]
    return {
        "bleu": round(sacrebleu.corpus_bleu(preds, [refs]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(preds, [refs]).score, 2),
        "cer": round(cer(preds, refs), 2),
        "exact_match": round(100.0 * sum(p == r for p, r in zip(preds, refs)) / max(len(refs), 1), 2),
        "n": len(refs),
    }
