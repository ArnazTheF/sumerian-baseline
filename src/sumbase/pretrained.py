from __future__ import annotations

import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .charset import glyph_cer_tokens
from .config import Config
from .metrics import cer, text_metrics
from .normalize import clean_ws
from .seq2seq import write_predictions
from .utils import get_logger, peak_vram_gb, pick_device, read_jsonl, write_json

log = get_logger(__name__)

DEFAULTS = {
    "translit": {"model": "colesimmons/sumerian-transliteration",
                 "encoder_tokenizer": "colesimmons/SumerianGlyphTokenizer_Roberta",
                 "decoder_tokenizer": "colesimmons/SumerianTransliterationTokenizer_Roberta"},
    "ocr": {"model": "colesimmons/lineart-ocr",
            "decoder_tokenizer": "colesimmons/SumerianGlyphTokenizer_Roberta"},
}
_DROP = {"<s>", "</s>", "<pad>", "<mask>"}


# пост-обработка
def tokens_to_translit(tokens: list[str]) -> str:
    text = "".join(t for t in tokens if t not in _DROP)
    text = text.replace("<SURFACE>", "\n").replace("▁", " ").replace("<unk>", "<UNK>")
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return clean_ws(text).strip()


def tokens_to_glyphs(tokens: list[str]) -> str:
    text = "".join(t for t in tokens if t not in _DROP).replace("<unk>", "<UNK>")
    return clean_ws(text).strip()


def to_glyph_input(src: str) -> str:
    src = src.replace("<UNK>", "<unk>")
    return src if src.startswith("<SURFACE>") else "<SURFACE> " + src


def _bl(cfg: Config, key: str, default=None):
    return (cfg.get("baselines") or {}).get(f"zeroshot_{key}", default)


def _ids_to_tokens(tok, ids) -> list[str]:
    return tok.convert_ids_to_tokens([int(i) for i in ids if int(i) >= 0])


# транслитерация
@torch.no_grad()
def zeroshot_translit(cfg: Config, split: str = "test") -> dict:
    from transformers import AutoTokenizer, EncoderDecoderModel

    d = DEFAULTS["translit"]
    name = _bl(cfg, "model", d["model"])
    enc_tok = AutoTokenizer.from_pretrained(_bl(cfg, "encoder_tokenizer", d["encoder_tokenizer"]))
    dec_tok = AutoTokenizer.from_pretrained(_bl(cfg, "decoder_tokenizer", d["decoder_tokenizer"]))
    model = EncoderDecoderModel.from_pretrained(name)
    rows = read_jsonl(Path(cfg.paths.data_dir) / f"{split}.jsonl")
    preds = _run_text(model, enc_tok, dec_tok, [to_glyph_input(r["src"]) for r in rows], cfg)
    return _finish_text(cfg, rows, preds, f"zeroshot:{name}", split)


def _run_text(model, enc_tok, dec_tok, sources: list[str], cfg: Config) -> list[str]:
    device = pick_device()
    model.to(device).eval()
    bs, beams = int(_bl(cfg, "batch_size", 16)), int(_bl(cfg, "num_beams", 1))
    max_new = int(_bl(cfg, "max_new_tokens", 510))
    out: list[tuple[int, str]] = []
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        batch = enc_tok([sources[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                        max_length=512).to(device)
        gen = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             num_beams=beams, max_new_tokens=max_new, decoder_start_token_id=dec_tok.bos_token_id,
                             bos_token_id=dec_tok.bos_token_id, eos_token_id=dec_tok.eos_token_id,
                             pad_token_id=dec_tok.pad_token_id)
        out.extend((i, tokens_to_translit(_ids_to_tokens(dec_tok, g))) for i, g in zip(idx, gen.tolist()))
        log.info("zero-shot translit: %d / %d", min(k + bs, len(order)), len(order))
    return [p for _, p in sorted(out)]


def _finish_text(cfg: Config, rows: list[dict], preds: list[str], label: str, split: str) -> dict:
    out_dir = Path(cfg.paths.output_dir) / "baseline_zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = text_metrics(preds, [r["tgt"] for r in rows])
    metrics.update(baseline=label, split=split, device=pick_device(), peak_memory_gb=peak_vram_gb())
    write_predictions(rows, preds, out_dir / f"predictions_{split}.tsv")
    write_json(metrics, out_dir / f"metrics_{split}.json")
    log.info("%s: %s", label, metrics)
    return metrics


# OCR по прорисовкам
def pad_square_resize(path: str, size: int, fill: str = "corner") -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    bg = img.getpixel((0, 0)) if fill == "corner" else (255, 255, 255)
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), bg)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


@torch.no_grad()
def zeroshot_ocr(cfg: Config, split: str = "test") -> dict:
    import torchvision.transforms.functional as F
    from transformers import AutoTokenizer, VisionEncoderDecoderModel

    d = DEFAULTS["ocr"]
    name = _bl(cfg, "model", d["model"])
    tok = AutoTokenizer.from_pretrained(_bl(cfg, "decoder_tokenizer", d["decoder_tokenizer"]))
    model = VisionEncoderDecoderModel.from_pretrained(name)
    device = pick_device()
    model.to(device).eval()
    size = int(_bl(cfg, "image_size", model.config.encoder.image_size))
    mean = _bl(cfg, "image_mean", [0.5, 0.5, 0.5])
    std = _bl(cfg, "image_std", [0.5, 0.5, 0.5])
    fill = _bl(cfg, "pad_fill", "corner")
    bs, beams = int(_bl(cfg, "batch_size", 8)), int(_bl(cfg, "num_beams", 1))
    max_new = int(_bl(cfg, "max_new_tokens", 258))

    rows = read_jsonl(Path(cfg.paths.data_dir) / f"{split}.jsonl")
    t0 = time.time()
    preds: list[str] = []
    for k in range(0, len(rows), bs):
        pix = torch.stack([F.normalize(F.to_tensor(pad_square_resize(r["image"], size, fill)), mean, std)
                           for r in rows[k:k + bs]]).to(device)
        gen = model.generate(pixel_values=pix, num_beams=beams, max_new_tokens=max_new,
                             decoder_start_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.pad_token_id)
        preds.extend(tokens_to_glyphs(_ids_to_tokens(tok, g)) for g in gen.tolist())
        log.info("zero-shot OCR: %d / %d", min(k + bs, len(rows)), len(rows))
    out_dir = Path(cfg.paths.output_dir) / "baseline_zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"cer": round(cer([glyph_cer_tokens(p) for p in preds], [glyph_cer_tokens(r["tgt"]) for r in rows]), 2),
               "n": len(rows), "split": split, "baseline": f"zeroshot:{name}", "device": device,
               "seconds": round(time.time() - t0, 1), "peak_memory_gb": peak_vram_gb()}
    write_predictions(rows, preds, out_dir / f"predictions_{split}.tsv")
    write_json(metrics, out_dir / f"metrics_{split}.json")
    log.info("%s: %s", name, metrics)
    return metrics
