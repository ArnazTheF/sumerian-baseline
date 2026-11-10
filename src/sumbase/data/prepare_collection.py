from __future__ import annotations

import json
from pathlib import Path

from ..utils import write_json, write_jsonl


def prepare_collection(cfg):
    """Подготовка JSONL корпуса для команд проекта."""
    root = Path(cfg.data.get("collection_root", "data/collection"))
    task = cfg.data.get("collection_task", cfg.task)
    folder = root / "exports" / task
    dest = Path(cfg.paths.data_dir)
    if folder.resolve() == dest.resolve():
        raise ValueError("paths.data_dir must not overwrite canonical collection exports")
    test_source = cfg.data.get(
        "test_source", "external_candidates" if cfg.task != "ocr" else "test"
    )
    stats = {"collection_root": str(root), "collection_task": task, "test_source": test_source}
    for split in ("train", "valid", "test"):
        source_split = test_source if split == "test" else split
        path = folder / f"{source_split}.jsonl"
        limit = cfg.data.get("max_" + split)
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if limit is not None and len(rows) >= int(limit):
                    break
                row = json.loads(line)
                normalization = cfg.data.get("input_normalization", "none")
                if normalization not in {"none", "cdli_ascii"}:
                    raise ValueError("data.input_normalization: none | cdli_ascii")
                if cfg.task == "mt" and normalization == "cdli_ascii":
                    from ..normalize import to_cdli_ascii

                    row["src_original"] = row["src"]
                    row["src"] = to_cdli_ascii(row["src"])
                # Лимиты модели применяются при инференсе, не к эталонам.
                rows.append(row)
        stats[split] = write_jsonl(rows, dest / f"{split}.jsonl")
    stats["input_normalization"] = cfg.data.get("input_normalization", "none")
    write_json(stats, dest / "stats.json")
    return stats
