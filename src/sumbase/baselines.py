from __future__ import annotations

from pathlib import Path

from .config import Config
from .metrics import text_metrics
from .seq2seq import generate_texts, write_predictions
from .utils import get_logger, read_jsonl, write_json

log = get_logger(__name__)


def translation_memory(cfg: Config, split: str = "test") -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    data_dir = Path(cfg.paths.data_dir)
    out_dir = Path(cfg.paths.output_dir) / "baseline_tm"
    train, test = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / f"{split}.jsonl")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    x_train = vec.fit_transform([r["src"] for r in train])
    x_test = vec.transform([r["src"] for r in test])
    preds = []
    for start in range(0, x_test.shape[0], 512):
        sims = x_test[start:start + 512] @ x_train.T  
        best = sims.argmax(axis=1).A1
        preds.extend(train[i]["tgt"] for i in best)
    metrics = text_metrics(preds, [r["tgt"] for r in test])
    metrics.update(baseline="translation_memory", split=split)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(test, preds, out_dir / f"predictions_{split}.tsv")
    write_json(metrics, out_dir / f"metrics_{split}.json")
    log.info("Translation memory: %s", metrics)
    return metrics


def zeroshot(cfg: Config, split: str = "test") -> dict:
    if cfg.task == "translit":
        from .pretrained import zeroshot_translit

        return zeroshot_translit(cfg, split)
    if cfg.task == "ocr":
        from .pretrained import zeroshot_ocr

        return zeroshot_ocr(cfg, split)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    bl = cfg.get("baselines", {}) or {}
    name = bl.get("zeroshot_model", "praeclarum/cuneiform")
    prefix = bl.get("zeroshot_prefix", "translate Sumerian to English: ")
    out_dir = Path(cfg.paths.output_dir) / "baseline_zeroshot"
    rows = read_jsonl(Path(cfg.paths.data_dir) / f"{split}.jsonl")
    tok, model = AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)
    gen = cfg.generation
    preds = generate_texts(model, tok, [r["src"] for r in rows], prefix=prefix,
                           batch_size=int(gen.get("batch_size", 32)), num_beams=int(gen.get("num_beams", 4)),
                           max_new_tokens=int(gen.get("max_new_tokens", 256)))
    metrics = text_metrics(preds, [r["tgt"] for r in rows])
    from .utils import peak_vram_gb, pick_device

    metrics.update(baseline=f"zeroshot:{name}", split=split, device=pick_device(), peak_memory_gb=peak_vram_gb())
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(rows, preds, out_dir / f"predictions_{split}.tsv")
    write_json(metrics, out_dir / f"metrics_{split}.json")
    log.info("Zero-shot %s: %s", name, metrics)
    return metrics
