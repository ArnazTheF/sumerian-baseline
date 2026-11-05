from __future__ import annotations

from pathlib import Path

from ..charset import split_tokens
from ..config import Config
from ..normalize import clean_ws, protect_markers
from ..utils import get_logger, read_jsonl, write_json, write_jsonl

log = get_logger(__name__)

HF_SPLITS = [("test", "test"), ("valid", "validation"), ("train", "train")]


def portable(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _save_image(img, path: Path, max_side: int) -> None:
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=90)


def prepare_ocr(cfg: Config) -> dict:
    data = cfg.data
    out_dir = Path(cfg.paths.data_dir)
    max_glyphs = int(data.get("max_target_tokens", 200))
    min_glyphs = int(data.get("min_target_tokens", 3))
    max_side = int(data.get("image_max_side", 1024))
    limits = {"train": data.get("max_train", 2000), "valid": data.get("max_valid", 200), "test": data.get("max_test", 300)}
    max_scan = int(data.get("max_scan_factor", 20))

    if data.source == "local":
        root = Path(data.local_dir)
        rows = read_jsonl(root / "manifest.jsonl")
        stats = {}
        for split in ("train", "valid", "test"):
            lim = limits[split]
            sel = [r for r in rows if r.get("split", "train") == split][: None if lim is None else int(lim)]
            out = [{"id": str(r.get("id", i)), "image": portable(root / r["image"]),
                    "tgt": protect_markers(clean_ws(r["glyphs"])), "period": r.get("period", ""),
                    "genre": r.get("genre", "")} for i, r in enumerate(sel)]
            stats[split] = write_jsonl(out, out_dir / f"{split}.jsonl")
        write_json(stats, out_dir / "stats.json")
        return stats

    if data.source != "hf":
        raise ValueError("data.source для ocr: hf | local")

    from datasets import load_dataset

    periods = set(data.get("periods") or [])
    reserved_ids: set[str] = set()
    stats: dict = {"hf_config": data.get("hf_config", "photo_only"), "max_target_tokens": max_glyphs}
    for split, hf_split in HF_SPLITS:
        limit = int(limits[split])
        ds = load_dataset(data.get("hf_name", "colesimmons/SumTablets_Photos"), data.get("hf_config", "photo_only"),
                          split=hf_split, streaming=True, revision=data.get("hf_revision"))
        kept, scanned, skipped_len, skipped_leak = [], 0, 0, 0
        per_id: dict[str, int] = {}
        for row in ds:
            scanned += 1
            if scanned > limit * max_scan or len(kept) >= limit:
                break
            tid = str(row["id"])
            if split == "train" and tid in reserved_ids:
                skipped_leak += 1
                continue
            if periods and row.get("period") not in periods:
                continue
            tgt = protect_markers(clean_ws(row["glyphs"]))
            n_tok = len(split_tokens(tgt))
            if not (min_glyphs <= n_tok <= max_glyphs):
                skipped_len += 1
                continue
            k = per_id.get(tid, 0)
            per_id[tid] = k + 1
            img_path = out_dir / "images" / split / f"{tid}_{k}.jpg"
            try:
                _save_image(row["image"], img_path, max_side)
            except Exception as exc:
                log.warning("Пропускаю %s: %s", tid, exc)
                continue
            kept.append({"id": f"{tid}_{k}", "tablet": tid, "image": portable(img_path), "tgt": tgt,
                         "period": row.get("period", ""), "genre": row.get("genre", "")})
            if len(kept) % 200 == 0:
                log.info("%s: сохранено %d / %d", split, len(kept), limit)
        if split != "train":
            reserved_ids.update(r["tablet"] for r in kept)
        stats[split] = {"examples": write_jsonl(kept, out_dir / f"{split}.jsonl"), "scanned": scanned,
                        "skipped_too_long_or_short": skipped_len, "skipped_leak": skipped_leak}
        log.info("%s: %s", split, stats[split])
    write_json(stats, out_dir / "stats.json")
    return stats
