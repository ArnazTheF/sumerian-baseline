from __future__ import annotations

import json
import re
from pathlib import Path

from .config import load_config
from .normalize import clean_ws, protect_markers, to_cdli_ascii
from .utils import get_logger

log = get_logger(__name__)


def _run_config(run_dir: str | Path):
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}: укажите каталог запуска, например runs/mt_byt5_small")
    return load_config(path)


def predict_texts(run_dir: str | Path, texts: list[str], num_beams: int | None = None) -> list[str]:
    from .seq2seq import generate_texts, load_trained

    cfg = _run_config(run_dir)
    tok, model = load_trained(run_dir)
    texts = [protect_markers(clean_ws(t)) for t in texts]
    return generate_texts(model, tok, texts, prefix=cfg.model.get("prefix", "") or "",
                          batch_size=int(cfg.generation.get("batch_size", 32)),
                          num_beams=num_beams or int(cfg.generation.get("num_beams", 4)),
                          max_new_tokens=int(cfg.generation.get("max_new_tokens", 256)),
                          max_source_length=int(cfg.model.max_source_length))


def predict_images(run_dir: str | Path, images: list[str], num_beams: int | None = None) -> list[str]:
    from .ocr import load_trained_ocr, recognize

    cfg = _run_config(run_dir)
    tok, model = load_trained_ocr(run_dir)
    return recognize(model, tok, cfg, images, num_beams=num_beams or int(cfg.generation.get("num_beams", 1)))


def pipeline(image: str, ocr_run: str, translit_run: str, mt_run: str | None = None) -> dict:
    glyphs = predict_images(ocr_run, [image])[0]
    surfaces = [s.strip() for s in re.split(r"<SURFACE>", glyphs) if s.strip()] or [glyphs]
    translits = predict_texts(translit_run, surfaces)
    result: dict = {"image": image, "glyphs": glyphs, "surfaces": []}
    lines_all: list[str] = []
    for g, t in zip(surfaces, translits):
        lines = [x.strip() for x in t.split("\n") if x.strip()] or [t]
        result["surfaces"].append({"glyphs": g, "transliteration": t, "lines": lines})
        lines_all.extend(lines)
    if mt_run and lines_all:
        translations = iter(predict_texts(mt_run, [to_cdli_ascii(x) for x in lines_all]))
        for surface in result["surfaces"]:
            surface["translation"] = [next(translations) for _ in surface["lines"]]
    return result


def run_predict(args) -> None:
    if args.pipeline:
        if not (args.image and args.ocr_run and args.translit_run):
            raise SystemExit("Для --pipeline нужны --image, --ocr-run, --translit-run (и опционально --mt-run)")
        results = [pipeline(img, args.ocr_run, args.translit_run, args.mt_run) for img in args.image]
        text = json.dumps(results, ensure_ascii=False, indent=2)
    else:
        if not args.run:
            raise SystemExit("Укажите --run runs/<имя_запуска>")
        if args.image:
            preds = predict_images(args.run, args.image, args.num_beams)
            inputs = args.image
        else:
            inputs = list(args.text or [])
            if args.input:
                inputs += [x for x in Path(args.input).read_text(encoding="utf-8").split("\n") if x.strip()]
            if not inputs:
                raise SystemExit("Нет входа: используйте --text, --input или --image")
            preds = predict_texts(args.run, inputs, args.num_beams)
        text = "\n".join(f"{i}\t{p}" for i, p in zip(inputs, preds))
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        log.info("Сохранено в %s", args.output)
    print(text)
