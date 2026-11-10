from __future__ import annotations

import random
import re
from pathlib import Path

from ..config import Config
from ..normalize import clean_ws, protect_markers
from ..utils import get_logger, write_json, write_jsonl

log = get_logger(__name__)

_SURFACE = re.compile(r"\s*<SURFACE>\s*")
HF_SPLITS = {"train": "train", "valid": "validation", "test": "test"}


def _split_surfaces(text: str) -> list[str]:
    return [p for p in (s.strip() for s in _SURFACE.split(text)) if p]


def _glyph_lines(surface: str) -> tuple[list[str], str]:
    sep = "\n" if "\n" in surface else " "
    return [x for x in surface.split(sep) if x.strip()], sep


def chunk_pair(
    glyphs: str, translit: str, mode: str, src_max_chars: int
) -> tuple[list[tuple[str, str]], str]:
    if mode == "tablet":
        return [(glyphs, translit)], "tablet"
    g_surf, t_surf = _split_surfaces(glyphs), _split_surfaces(translit)
    if len(g_surf) != len(t_surf) or not g_surf:
        return [(glyphs, translit)], "tablet"
    if mode == "surface":
        return list(zip(g_surf, t_surf)), "surface"

    pairs: list[tuple[str, str]] = []
    used = "lines"
    for gs, ts in zip(g_surf, t_surf):
        (g_lines, g_sep), t_lines = _glyph_lines(gs), [x for x in ts.split("\n") if x.strip()]
        if len(g_lines) != len(t_lines) or len(t_lines) <= 1:
            pairs.append((gs, ts))
            used = "surface"
            continue
        buf_g: list[str] = []
        buf_t: list[str] = []
        for gl, tl in zip(g_lines, t_lines):
            if buf_g and sum(len(x) for x in buf_g) + len(gl) > src_max_chars:
                pairs.append((g_sep.join(buf_g), "\n".join(buf_t)))
                buf_g, buf_t = [], []
            buf_g.append(gl)
            buf_t.append(tl)
        if buf_g:
            pairs.append((g_sep.join(buf_g), "\n".join(buf_t)))
    return pairs, used


def _load_rows(cfg: Config) -> dict[str, list[dict]]:
    data = cfg.data
    cols = ["id", "glyphs", "transliteration", "period", "genre"]
    if data.source == "hf":
        from datasets import load_dataset

        ds = load_dataset(
            data.get("hf_name", "colesimmons/SumTablets"), revision=data.get("hf_revision")
        )
        return {
            split: [{c: r.get(c) for c in cols} for r in ds[hf]] for split, hf in HF_SPLITS.items()
        }
    if data.source == "local":
        import pandas as pd

        df = pd.read_csv(data.local_csv).fillna("")
        if "split" not in df.columns:
            buckets = df["id"].map(lambda x: sum(map(ord, str(x))) % 20)
            df["split"] = ["test" if b == 0 else "valid" if b == 1 else "train" for b in buckets]
        out = {}
        for split in HF_SPLITS:
            sub = df[df["split"] == split]
            out[split] = [
                {c: (row[c] if c in sub.columns else "") for c in cols} for _, row in sub.iterrows()
            ]
        return out
    raise ValueError("data.source для translit: hf | local")


def prepare_translit(cfg: Config) -> dict:
    data = cfg.data
    rng = random.Random(cfg.seed)
    out_dir = Path(cfg.paths.data_dir)
    mode = data.get("chunk_mode", "surface")
    src_max, tgt_max = int(data.get("src_max_chars", 120)), int(data.get("tgt_max_chars", 600))
    periods = set(data.get("periods") or [])
    genres = set(data.get("genres") or [])
    limits = {
        "train": data.get("max_train_tablets"),
        "valid": data.get("max_valid_tablets"),
        "test": data.get("max_test_tablets"),
    }

    rows_by_split = _load_rows(cfg)
    stats: dict = {"chunk_mode": mode, "src_max_chars": src_max, "tgt_max_chars": tgt_max}
    for split, rows in rows_by_split.items():
        rows = [
            r
            for r in rows
            if (not periods or r["period"] in periods) and (not genres or r["genre"] in genres)
        ]
        rng.shuffle(rows)
        if limits[split] is not None:
            rows = rows[: int(limits[split])]
        examples, modes, dropped = [], {"tablet": 0, "surface": 0, "lines": 0}, 0
        for r in rows:
            glyphs = protect_markers(clean_ws(r["glyphs"]))
            translit = protect_markers(clean_ws(r["transliteration"]))
            pairs, used = chunk_pair(glyphs, translit, mode, src_max)
            modes[used] += 1
            for k, (g, t) in enumerate(pairs):
                if len(g) > src_max or len(t) > tgt_max:
                    dropped += 1
                    continue
                examples.append(
                    {
                        "id": f"{r['id']}_{k}",
                        "tablet": r["id"],
                        "src": g,
                        "tgt": t,
                        "period": r["period"],
                        "genre": r["genre"],
                    }
                )
        stats[split] = {
            "tablets": len(rows),
            "examples": write_jsonl(examples, out_dir / f"{split}.jsonl"),
            "dropped_too_long": dropped,
            "chunking_used": modes,
        }
    write_json(stats, out_dir / "stats.json")
    log.info("Translit-данные готовы: %s", stats)
    return stats
