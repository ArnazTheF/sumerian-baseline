from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  
            raise AttributeError(f"В конфиге нет ключа '{key}'") from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _set_dotted(tree: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _deep_to_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _deep_to_config(v) for k, v in obj.items()})
    return obj


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Переопределение должно иметь вид key.sub=value, получено: {item}")
        key, value = item.split("=", 1)
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    for required in ("task", "run_name", "paths"):
        if required not in raw:
            raise ValueError(f"В конфиге {path} нет обязательного поля '{required}'")
    if raw["task"] not in {"mt", "translit", "ocr"}:
        raise ValueError("task должен быть одним из: mt, translit, ocr")
    raw.setdefault("seed", 42)
    raw["paths"].setdefault("output_dir", f"runs/{raw['run_name']}")
    return _deep_to_config(raw)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_plain(cfg), fh, allow_unicode=True, sort_keys=False)


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj
