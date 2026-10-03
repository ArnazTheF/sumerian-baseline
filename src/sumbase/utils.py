from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=os.environ.get("SUMBASE_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def resolve_precision(requested: str, model_family: str) -> tuple[bool, bool]:
    import torch

    requested = (requested or "auto").lower()
    cuda = torch.cuda.is_available()
    if requested == "fp32" or not cuda:
        return False, False
    if requested == "bf16":
        return True, False
    if requested == "fp16":
        return False, True
    if torch.cuda.is_bf16_supported():
        return True, False
    return (False, False) if model_family == "t5" else (False, True)


def device_summary() -> dict:
    import torch

    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "mps_available": mps_available(), "device_used": pick_device()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_count=torch.cuda.device_count(),
            vram_gb=round(props.total_memory / 1024**3, 1),
            bf16=torch.cuda.is_bf16_supported(),
        )
    return info


def peak_vram_gb() -> float | None:
    import torch

    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    if mps_available():  
        return round(torch.mps.driver_allocated_memory() / 1024**3, 2)
    return None


def mps_available() -> bool:
    import torch

    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def pick_device() -> str:
    import torch

    forced = os.environ.get("SUMBASE_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def apply_vram_cap(cfg) -> None:
    import torch

    cap = cfg.train.get("max_vram_gb") if hasattr(cfg, "train") else None
    if not cap or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if float(cap) < total:
        torch.cuda.set_per_process_memory_fraction(float(cap) / total, 0)
        get_logger(__name__).info("Видеопамять ограничена до %.1f ГБ из %.1f ГБ (train.max_vram_gb)", float(cap), total)
