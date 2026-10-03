from __future__ import annotations

import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import Config, save_config
from .metrics import text_metrics
from .utils import (apply_vram_cap, device_summary, get_logger, peak_vram_gb, pick_device, read_jsonl, resolve_precision,
                    set_seed, write_json)

log = get_logger(__name__)

TINY_RANDOM = "byt5-tiny-random"  # крошечная случайная модель для смоук-теста без скачивания весов


# модель
def build_text_model(cfg: Config):
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, ByT5Tokenizer, T5Config,
                              T5ForConditionalGeneration)

    name = cfg.model.name
    if name == TINY_RANDOM:
        tok = ByT5Tokenizer()
        conf = T5Config(vocab_size=384, d_model=64, d_ff=128, d_kv=16, num_heads=4, num_layers=2,
                        num_decoder_layers=1, feed_forward_proj="gated-gelu", decoder_start_token_id=0,
                        pad_token_id=0, eos_token_id=1, tie_word_embeddings=False)
        return tok, T5ForConditionalGeneration(conf)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tok, model


def model_family(model) -> str:
    return "t5" if model.config.model_type in {"t5", "mt5", "umt5"} else "other"


def load_trained(run_dir: str | Path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_dir = Path(run_dir) / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Нет обученной модели в {model_dir}. Сначала запустите train.")
    return AutoTokenizer.from_pretrained(model_dir), AutoModelForSeq2SeqLM.from_pretrained(model_dir)


# данные
class EncodedDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def encode_rows(rows: list[dict], tok, cfg: Config, drop_long: bool) -> tuple[EncodedDataset, dict]:
    prefix = cfg.model.get("prefix", "") or ""
    max_src, max_tgt = int(cfg.model.max_source_length), int(cfg.model.max_target_length)
    src = tok([prefix + r["src"] for r in rows], truncation=False)["input_ids"]
    tgt = tok(text_target=[r["tgt"] for r in rows], truncation=False)["input_ids"]
    items, dropped, truncated = [], 0, 0
    for s, t in zip(src, tgt):
        too_long = len(s) > max_src or len(t) > max_tgt
        if too_long and drop_long:
            dropped += 1
            continue
        if too_long:
            truncated += 1
            s = s[: max_src - 1] + [tok.eos_token_id] if len(s) > max_src else s
            t = t[: max_tgt - 1] + [tok.eos_token_id] if len(t) > max_tgt else t
        items.append({"input_ids": s, "attention_mask": [1] * len(s), "labels": t})
    return EncodedDataset(items), {"kept": len(items), "dropped_long": dropped, "truncated": truncated}


# генерация
@torch.no_grad()
def generate_texts(model, tok, sources: list[str], prefix: str = "", batch_size: int = 32, num_beams: int = 4,
                   max_new_tokens: int = 256, max_source_length: int = 512, device: str | None = None) -> list[str]:
    device = device or pick_device()
    model.to(device).eval()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    outputs: list[str] = []
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tok([prefix + sources[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                    max_length=max_source_length).to(device)
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
        with ctx:
            gen = model.generate(**batch, num_beams=num_beams, max_new_tokens=max_new_tokens)
        outputs.extend(zip(idx, tok.batch_decode(gen, skip_special_tokens=True)))
    return [text for _, text in sorted(outputs)]


# обучение
def _training_args(cfg: Config, run_dir: Path, bf16: bool, fp16: bool, metric: str, greater: bool):
    from transformers import Seq2SeqTrainingArguments

    tr, gen = cfg.train, cfg.generation
    strategy = tr.get("eval_strategy", "epoch")
    return Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=int(tr.batch_size),
        per_device_eval_batch_size=int(gen.get("batch_size", tr.batch_size)),
        gradient_accumulation_steps=int(tr.get("grad_accum", 1)),
        learning_rate=float(tr.lr),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        num_train_epochs=float(tr.epochs),
        max_steps=int(tr.get("max_steps", -1)),
        warmup_ratio=float(tr.get("warmup_ratio", 0.0)),
        lr_scheduler_type=tr.get("scheduler", "linear"),
        label_smoothing_factor=float(tr.get("label_smoothing", 0.0)),
        optim=tr.get("optim", "adamw_torch"),
        logging_steps=int(tr.get("logging_steps", 50)),
        eval_strategy=strategy,
        save_strategy=strategy,
        eval_steps=int(tr.get("eval_steps", 500)) if strategy == "steps" else None,
        save_steps=int(tr.get("eval_steps", 500)) if strategy == "steps" else None,
        save_total_limit=int(tr.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=greater,
        predict_with_generate=True,
        generation_max_length=int(gen.get("max_new_tokens", cfg.model.max_target_length)),
        generation_num_beams=int(gen.get("train_eval_beams", 1)),
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(tr.get("gradient_checkpointing", False)),
        dataloader_num_workers=int(tr.get("num_workers", 2)),
        report_to=tr.get("report_to", "tensorboard"),
        logging_dir=str(run_dir / "logs"),
        seed=int(cfg.seed),
        remove_unused_columns=False,
        use_cpu=pick_device() == "cpu",
        dataloader_pin_memory=pick_device() == "cuda",
    )


def train_text(cfg: Config, run_test: bool = True) -> dict:
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer

    set_seed(int(cfg.seed))
    data_dir, run_dir = Path(cfg.paths.data_dir), Path(cfg.paths.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    log.info("Устройство: %s", device_summary())
    apply_vram_cap(cfg)

    tok, model = build_text_model(cfg)
    family = model_family(model)
    bf16, fp16 = resolve_precision(cfg.train.get("precision", "auto"), family)
    log.info("Модель %s: %.1fM параметров, bf16=%s fp16=%s", cfg.model.name,
             sum(p.numel() for p in model.parameters()) / 1e6, bf16, fp16)

    train_rows, valid_rows = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "valid.jsonl")
    if cfg.train.get("max_valid_during_train"):
        valid_rows = valid_rows[: int(cfg.train.max_valid_during_train)]
    train_ds, tr_stats = encode_rows(train_rows, tok, cfg, drop_long=True)
    valid_ds, va_stats = encode_rows(valid_rows, tok, cfg, drop_long=False)
    log.info("train: %s | valid: %s", tr_stats, va_stats)

    metric = cfg.train.get("metric_for_best", "chrf")
    greater = metric not in {"cer", "loss"}

    def compute_metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.copy()
        labels = labels.copy()
        preds[preds < 0] = tok.pad_token_id
        labels[labels < 0] = tok.pad_token_id
        return text_metrics(tok.batch_decode(preds, skip_special_tokens=True),
                            tok.batch_decode(labels, skip_special_tokens=True))

    args = _training_args(cfg, run_dir, bf16, fp16, metric, greater)
    callbacks = []
    if cfg.train.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(cfg.train.early_stopping_patience)))
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds, processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        compute_metrics=compute_metrics, callbacks=callbacks,
    )
    t0 = time.time()
    result = trainer.train()
    trainer.save_model(str(run_dir / "model"))
    tok.save_pretrained(str(run_dir / "model"))
    summary = {
        "train_minutes": round((time.time() - t0) / 60, 1),
        "train_loss": round(result.training_loss, 4),
        "best_metric": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "peak_vram_gb_train": peak_vram_gb(),
        "train_examples": tr_stats,
        "device": device_summary(),
    }
    write_json(summary, run_dir / "train_summary.json")
    log.info("Обучение завершено: %s", summary)
    if run_test:
        summary["test"] = evaluate_text(cfg, run_dir, split="test", model=trainer.model, tok=tok)
    return summary


# оценка
def evaluate_text(cfg: Config, run_dir: str | Path, split: str = "test", model=None, tok=None) -> dict:
    run_dir = Path(run_dir)
    if model is None:
        tok, model = load_trained(run_dir)
    rows = read_jsonl(Path(cfg.paths.data_dir) / f"{split}.jsonl")
    gen = cfg.generation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    preds = generate_texts(model, tok, [r["src"] for r in rows], prefix=cfg.model.get("prefix", "") or "",
                           batch_size=int(gen.get("batch_size", 32)), num_beams=int(gen.get("num_beams", 4)),
                           max_new_tokens=int(gen.get("max_new_tokens", 256)),
                           max_source_length=int(cfg.model.max_source_length))
    metrics = text_metrics(preds, [r["tgt"] for r in rows])
    metrics.update(split=split, num_beams=int(gen.get("num_beams", 4)), seconds=round(time.time() - t0, 1),
                   peak_vram_gb_inference=peak_vram_gb())
    write_predictions(rows, preds, run_dir / f"predictions_{split}.tsv")
    write_json(metrics, run_dir / f"metrics_{split}.json")
    log.info("Метрики на %s: %s", split, metrics)
    return metrics


def write_predictions(rows: list[dict], preds: list[str], path: Path) -> None:
    def esc(x: str) -> str:
        x = str(x).replace("\t", " ").replace("\n", " \\n ")
        return "".join(ch for ch in x if ch.isprintable() or ch == " ")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("id\tsource\treference\tprediction\n")
        for r, p in zip(rows, preds):
            fh.write(f"{esc(r['id'])}\t{esc(r.get('src', r.get('image', '')))}\t{esc(r['tgt'])}\t{esc(p)}\n")
