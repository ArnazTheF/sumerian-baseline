"""Поиск пересечений с train и карантин близких текстов."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .parsers import loose_fingerprint


def shingles(text, n=3):
    tokens = loose_fingerprint(text).split()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)} if len(tokens) >= 8 else set()


def screen_mt_near_duplicates(folder: Path):
    """Эвристический поиск совпадений с train по редким триграммам."""
    with (folder / "train.jsonl").open() as source:
        train = [json.loads(s) for s in source]
    sets = [shingles(r["src"]) for r in train]
    index = defaultdict(list)
    for i, values in enumerate(sets):
        for value in values:
            index[value].append(i)
    # Частые формулы не помогают различать документы.
    index = {key: values for key, values in index.items() if len(values) <= 100}
    counts = Counter()
    quarantine = (folder / "near_duplicates.jsonl").open("w", encoding="utf-8")
    for split in ("valid", "test", "external_candidates"):
        path = folder / f"{split}.jsonl"
        temp = path.with_suffix(".screened.jsonl")
        with path.open() as source, temp.open("w", encoding="utf-8") as target:
            for line in source:
                r = json.loads(line)
                values = shingles(r["src"])
                candidates = Counter(i for value in values for i in index.get(value, ()))
                match = None
                for i, _ in candidates.most_common(30):
                    other = sets[i]
                    shared = len(values & other)
                    jaccard = shared / max(1, len(values | other))
                    containment = shared / max(1, min(len(values), len(other)))
                    # Коротких общих формул недостаточно для совпадения.
                    if jaccard >= 0.8 or (shared >= 12 and containment >= 0.95):
                        match = {"train_id": train[i]["id"], "jaccard": round(jaccard, 4),
                                 "containment": round(containment, 4), "shared_trigrams": shared}
                        break
                if match:
                    r["near_duplicate"] = match
                    r["original_split"] = split
                    r["quarantine_reason"] = "heuristic_source_overlap_with_train"
                    quarantine.write(json.dumps(r, ensure_ascii=False) + "\n")
                    counts[split] += 1
                else:
                    target.write(line)
        temp.replace(path)
    quarantine.close()
    return dict(counts)
