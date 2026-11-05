from __future__ import annotations

import random
import urllib.request
from pathlib import Path

from ..config import Config
from ..normalize import clean_ws
from ..utils import get_logger, read_jsonl, write_json, write_jsonl

log = get_logger(__name__)

CDLI_BASE = (
    "https://raw.githubusercontent.com/cdli-gh/Semi-Supervised-NMT-for-Sumerian-English/"
    "master/dataset/dataToUse"
)
CDLI_SUBSETS = {"UrIIILineByLine", "UrIIICompSents", "allLineByLine", "allCompSents"}
SPLIT_FILES = {"train": "train", "valid": "dev", "test": "test"}


def _fetch_lines(url: str, cache: Path) -> list[str]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        log.info("Скачиваю %s", url)
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - фиксированный URL
            cache.write_bytes(resp.read())
    return cache.read_text(encoding="utf-8").split("\n")


def _read_pair_files(src_path: Path, tgt_path: Path) -> list[tuple[str, str]]:
    src = src_path.read_text(encoding="utf-8").split("\n")
    tgt = tgt_path.read_text(encoding="utf-8").split("\n")
    if len(src) != len(tgt):
        raise ValueError(f"Разное число строк: {src_path} ({len(src)}) и {tgt_path} ({len(tgt)})")
    return list(zip(src, tgt))


def _load_split(cfg: Config, split: str, raw_dir: Path) -> list[tuple[str, str]]:
    data = cfg.data
    if data.source == "cdli_github":
        subset = data.get("subset", "UrIIILineByLine")
        if subset not in CDLI_SUBSETS:
            raise ValueError(f"data.subset должен быть одним из {sorted(CDLI_SUBSETS)}")
        stem = SPLIT_FILES[split]
        src = _fetch_lines(f"{CDLI_BASE}/{subset}/{stem}.sum", raw_dir / subset / f"{stem}.sum")
        tgt = _fetch_lines(f"{CDLI_BASE}/{subset}/{stem}.eng", raw_dir / subset / f"{stem}.eng")
        if len(src) != len(tgt):
            raise ValueError(f"CDLI {subset}/{stem}: разное число строк в .sum и .eng")
        return list(zip(src, tgt))
    if data.source == "local":
        root = Path(data.local_dir)
        jsonl = root / f"{split}.jsonl"
        if jsonl.exists():
            return [(r["src"], r["tgt"]) for r in read_jsonl(jsonl)]
        stem = SPLIT_FILES[split]
        return _read_pair_files(root / f"{stem}.sum", root / f"{stem}.eng")
    raise ValueError("data.source для mt: cdli_github | local")


def _clean(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out = []
    for src, tgt in pairs:
        src, tgt = clean_ws(src).replace("\n", " "), clean_ws(tgt).replace("\n", " ")
        if src and tgt:
            out.append((src, tgt))
    return out


def prepare_mt(cfg: Config) -> dict:
    data = cfg.data
    out_dir = Path(cfg.paths.data_dir)
    raw_dir = Path(cfg.paths.get("raw_dir", "data/raw")) / "cdli"
    rng = random.Random(cfg.seed)
    limits = {"train": data.get("max_train"), "valid": data.get("max_valid"), "test": data.get("max_test")}

    splits: dict[str, list[tuple[str, str]]] = {}
    for split in ("train", "valid", "test"):
        pairs = _clean(_load_split(cfg, split, raw_dir))
        if split == "train" and data.get("dedupe_train", True):
            pairs = list(dict.fromkeys(pairs))
        limit = limits[split]
        if limit is not None and len(pairs) > int(limit):
            pairs = rng.sample(pairs, int(limit))
        splits[split] = pairs

    stats: dict = {"source": data.source, "subset": data.get("subset")}
    for split, pairs in splits.items():
        rows = ({"id": f"{split}-{i}", "src": s, "tgt": t} for i, (s, t) in enumerate(pairs))
        stats[split] = write_jsonl(rows, out_dir / f"{split}.jsonl")

    train_src = {s for s, _ in splits["train"]}
    seen = sum(s in train_src for s, _ in splits["test"])
    stats["test_src_seen_in_train_pct"] = round(100 * seen / max(len(splits["test"]), 1), 2)
    stats["avg_src_chars"] = round(sum(len(s) for s, _ in splits["train"]) / max(len(splits["train"]), 1), 1)
    stats["avg_tgt_chars"] = round(sum(len(t) for _, t in splits["train"]) / max(len(splits["train"]), 1), 1)
    write_json(stats, out_dir / "stats.json")
    log.info("MT-данные готовы: %s", stats)
    return stats
