from __future__ import annotations

import contextlib
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .charset import CharTokenizer, canon
from .config import Config, save_config
from .metrics import cer
from .utils import (apply_vram_cap, device_summary, get_logger, peak_vram_gb, pick_device, read_jsonl, resolve_precision,
                    set_seed, write_json)

log = get_logger(__name__)

TINY_RANDOM_ENCODER = "swin-tiny-random"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# изображения
def letterbox(img: Image.Image, height: int, width: int) -> Image.Image:
    img = img.convert("RGB")
    bg = img.getpixel((0, 0))
    scale = min(width / img.width, height / img.height)
    new = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    canvas = Image.new("RGB", (width, height), bg)
    canvas.paste(new, ((width - new.width) // 2, (height - new.height) // 2))
    return canvas


class ImageTransform:
    def __init__(self, cfg: Config, train: bool):
        m = cfg.model
        self.h, self.w = int(m.image_height), int(m.image_width)
        self.gray = bool(m.get("grayscale", True))
        self.augment = train and bool(cfg.train.get("augment", True))

    def __call__(self, path: str) -> torch.Tensor:
        import torchvision.transforms.functional as F

        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        if self.gray:
            img = ImageOps.grayscale(img).convert("RGB")
        if self.augment:
            img = F.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = F.adjust_contrast(img, random.uniform(0.8, 1.2))
            img = F.rotate(img, random.uniform(-2, 2), fill=img.getpixel((0, 0)))
        img = letterbox(img, self.h, self.w)
        tensor = F.to_tensor(img)
        return F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class OCRDataset(Dataset):
    def __init__(self, rows: list[dict], tok: CharTokenizer, cfg: Config, train: bool):
        self.rows, self.tok = rows, tok
        self.max_len = int(cfg.model.max_target_length)
        self.transform = ImageTransform(cfg, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {"pixel_values": self.transform(r["image"]), "labels": self.tok.encode(r["tgt"], self.max_len)}


def collate(batch: list[dict]) -> dict:
    pixel = torch.stack([b["pixel_values"] for b in batch])
    max_len = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : len(b["labels"])] = torch.tensor(b["labels"])
    return {"pixel_values": pixel, "labels": labels}


# модель
def build_ocr_model(cfg: Config, tok: CharTokenizer, encoder=None):
    from transformers import (AutoModel, SwinConfig, SwinModel, TrOCRConfig, TrOCRForCausalLM,
                              VisionEncoderDecoderModel)

    m = cfg.model
    if "swin" in str(m.encoder).lower() and min(int(m.image_height), int(m.image_width)) < 224:
        raise ValueError("Для Swin image_height и image_width должны быть не меньше 224")
    if encoder is not None:
        pass  
    elif m.encoder == TINY_RANDOM_ENCODER:
        encoder = SwinModel(SwinConfig(embed_dim=24, depths=[1, 1, 1, 1], num_heads=[1, 2, 3, 3], window_size=7))
    else:
        encoder = AutoModel.from_pretrained(m.encoder)
    d = int(m.get("decoder_dim", 384))
    dec_conf = TrOCRConfig(
        vocab_size=len(tok), d_model=d, decoder_layers=int(m.get("decoder_layers", 4)),
        decoder_attention_heads=int(m.get("decoder_heads", 6)), decoder_ffn_dim=4 * d,
        max_position_embeddings=int(m.max_target_length) + 8, dropout=float(m.get("dropout", 0.1)),
        activation_function="gelu", use_learned_position_embeddings=True, layernorm_embedding=True,
        scale_embedding=True, pad_token_id=tok.pad_id, bos_token_id=tok.bos_id, eos_token_id=tok.eos_id,
        decoder_start_token_id=tok.bos_id, is_decoder=True, add_cross_attention=True,
    )
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=TrOCRForCausalLM(dec_conf))
    _set_special_ids(model, tok, int(m.max_target_length))
    _fix_loss(model)
    return model


def _fix_loss(model) -> None:
    model.loss_type = "ForMaskedLM"


def _set_special_ids(model, tok: CharTokenizer, max_len: int) -> None:
    for conf in (model.config, model.generation_config):
        conf.decoder_start_token_id = tok.bos_id
        conf.pad_token_id = tok.pad_id
        conf.eos_token_id = tok.eos_id
    model.config.vocab_size = len(tok)
    model.generation_config.max_length = max_len + 2


def load_trained_ocr(run_dir: str | Path):
    from transformers import VisionEncoderDecoderModel

    model_dir = Path(run_dir) / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Нет обученной модели в {model_dir}. Сначала запустите train.")
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    _fix_loss(model)
    return CharTokenizer.load(model_dir), model


# инференс
@torch.no_grad()
def recognize(model, tok: CharTokenizer, cfg: Config, image_paths: list[str], batch_size: int = 16,
              num_beams: int = 1, device: str | None = None) -> list[str]:
    device = device or pick_device()
    model.to(device).eval()
    transform = ImageTransform(cfg, train=False)
    use_amp = device == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    out: list[str] = []
    for start in range(0, len(image_paths), batch_size):
        pixel = torch.stack([transform(p) for p in image_paths[start:start + batch_size]]).to(device)
        ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else contextlib.nullcontext()
        with ctx:
            gen = model.generate(pixel_values=pixel, num_beams=num_beams,
                                 max_new_tokens=int(cfg.model.max_target_length) + 1)
        out.extend(tok.decode(g) for g in gen.tolist())
    return out


# обучение
def train_ocr(cfg: Config, run_test: bool = True) -> dict:
    from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

    set_seed(int(cfg.seed))
    data_dir, run_dir = Path(cfg.paths.data_dir), Path(cfg.paths.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    log.info("Устройство: %s", device_summary())
    apply_vram_cap(cfg)

    train_rows, valid_rows = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "valid.jsonl")
    if cfg.train.get("max_valid_during_train"):
        valid_rows = valid_rows[: int(cfg.train.max_valid_during_train)]
    tok = CharTokenizer.build([r["tgt"] for r in train_rows], min_freq=int(cfg.model.get("min_char_freq", 2)))
    log.info("Словарь глифов: %d токенов (train: %d изображений)", len(tok), len(train_rows))
    model = build_ocr_model(cfg, tok)
    bf16, fp16 = resolve_precision(cfg.train.get("precision", "auto"), "other")
    log.info("OCR-модель: %.1fM параметров, bf16=%s fp16=%s",
             sum(p.numel() for p in model.parameters()) / 1e6, bf16, fp16)

    tr, gen = cfg.train, cfg.generation
    strategy = tr.get("eval_strategy", "epoch")
    args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=int(tr.batch_size),
        per_device_eval_batch_size=int(gen.get("batch_size", tr.batch_size)),
        gradient_accumulation_steps=int(tr.get("grad_accum", 1)),
        learning_rate=float(tr.lr), weight_decay=float(tr.get("weight_decay", 0.0)),
        num_train_epochs=float(tr.epochs), max_steps=int(tr.get("max_steps", -1)),
        warmup_ratio=float(tr.get("warmup_ratio", 0.0)), lr_scheduler_type=tr.get("scheduler", "cosine"),
        logging_steps=int(tr.get("logging_steps", 50)),
        eval_strategy=strategy, save_strategy=strategy,
        eval_steps=int(tr.get("eval_steps", 500)) if strategy == "steps" else None,
        save_steps=int(tr.get("eval_steps", 500)) if strategy == "steps" else None,
        save_total_limit=int(tr.get("save_total_limit", 2)),
        load_best_model_at_end=True, metric_for_best_model="cer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=int(cfg.model.max_target_length) + 2,
        generation_num_beams=int(gen.get("train_eval_beams", 1)),
        bf16=bf16, fp16=fp16, dataloader_num_workers=int(tr.get("num_workers", 2)),
        report_to=tr.get("report_to", "tensorboard"), logging_dir=str(run_dir / "logs"),
        seed=int(cfg.seed), remove_unused_columns=False,
        use_cpu=pick_device() == "cpu", dataloader_pin_memory=pick_device() == "cuda",
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        p_tok = [canon(tok.decode_tokens(p)) for p in preds.tolist()]
        r_tok = [canon(tok.decode_tokens([x for x in l if x >= 0])) for l in labels.tolist()]
        return {"cer": round(cer(p_tok, r_tok), 2)}

    callbacks = []
    if tr.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(tr.early_stopping_patience)))
    trainer = Seq2SeqTrainer(
        model=model, args=args, data_collator=collate, compute_metrics=compute_metrics, callbacks=callbacks,
        train_dataset=OCRDataset(train_rows, tok, cfg, train=True),
        eval_dataset=OCRDataset(valid_rows, tok, cfg, train=False),
    )
    t0 = time.time()
    result = trainer.train()
    trainer.save_model(str(run_dir / "model"))
    tok.save(run_dir / "model")
    summary = {
        "train_minutes": round((time.time() - t0) / 60, 1),
        "train_loss": round(result.training_loss, 4),
        "best_cer_valid": trainer.state.best_metric,
        "vocab_size": len(tok),
        "peak_vram_gb_train": peak_vram_gb(),
        "device": device_summary(),
    }
    write_json(summary, run_dir / "train_summary.json")
    log.info("Обучение завершено: %s", summary)
    if run_test:
        summary["test"] = evaluate_ocr(cfg, run_dir, "test", model=trainer.model, tok=tok)
    return summary


def evaluate_ocr(cfg: Config, run_dir: str | Path, split: str = "test", model=None, tok=None) -> dict:
    from .seq2seq import write_predictions

    run_dir = Path(run_dir)
    if model is None:
        tok, model = load_trained_ocr(run_dir)
    rows = read_jsonl(Path(cfg.paths.data_dir) / f"{split}.jsonl")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    preds = recognize(model, tok, cfg, [r["image"] for r in rows], batch_size=int(cfg.generation.get("batch_size", 16)),
                      num_beams=int(cfg.generation.get("num_beams", 1)))
    from .charset import glyph_cer_tokens

    metrics = {
        "cer": round(cer([glyph_cer_tokens(p) for p in preds], [glyph_cer_tokens(r["tgt"]) for r in rows]), 2),
        "n": len(rows), "split": split, "seconds": round(time.time() - t0, 1),
        "peak_vram_gb_inference": peak_vram_gb(),
    }
    write_predictions(rows, preds, run_dir / f"predictions_{split}.tsv")
    write_json(metrics, run_dir / f"metrics_{split}.json")
    log.info("Метрики на %s: %s", split, metrics)
    return metrics
