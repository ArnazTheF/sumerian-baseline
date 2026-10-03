from __future__ import annotations

import contextlib
import gc

import torch

from .config import Config
from .utils import get_logger

log = get_logger(__name__)

OCR_VOCAB_GUESS = 1000  
CUDA_OVERHEAD_GB = 1.0 
FRAGMENTATION = 1.10


# модель без весов
def _text_model(cfg: Config):
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    from .seq2seq import TINY_RANDOM, build_text_model

    if cfg.model.name == TINY_RANDOM:
        return build_text_model(cfg)[1]
    return AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(cfg.model.name))


def _ocr_model(cfg: Config):
    from transformers import AutoConfig, AutoModel

    from . import ocr
    from .charset import SPECIALS, STRUCTURAL, CharTokenizer

    tok = CharTokenizer(SPECIALS + STRUCTURAL + [chr(0x12000 + i) for i in range(OCR_VOCAB_GUESS)])
    if cfg.model.encoder == ocr.TINY_RANDOM_ENCODER:
        return ocr.build_ocr_model(cfg, tok)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(cfg.model.encoder))
    return ocr.build_ocr_model(cfg, tok, encoder=encoder)


def _model_and_batch_inputs(cfg: Config, model, batch_size: int, device: str = "cpu"):
    if cfg.task == "ocr":
        h, w = int(cfg.model.image_height), int(cfg.model.image_width)
        t = int(cfg.model.max_target_length)
        batch = {"pixel_values": torch.randn(batch_size, 3, h, w),
                 "labels": torch.randint(10, model.config.decoder.vocab_size, (batch_size, t))}
    else:
        s, t = int(cfg.model.max_source_length), int(cfg.model.max_target_length)
        v = model.config.vocab_size
        batch = {"input_ids": torch.randint(3, v, (batch_size, s)), "attention_mask": torch.ones(batch_size, s, dtype=torch.long),
                 "labels": torch.randint(3, v, (batch_size, t))}
    return model, {k: v.to(device) for k, v in batch.items()}


def _model_and_batch(cfg: Config, batch_size: int, device: str):
    model = _ocr_model(cfg) if cfg.task == "ocr" else _text_model(cfg)
    if bool(cfg.train.get("gradient_checkpointing", False)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model.train()
    model = model.to(device)
    return _model_and_batch_inputs(cfg, model, batch_size, device)


def _optimizer_bytes_per_param(optim: str) -> int:
    optim = (optim or "adamw_torch").lower()
    if "adafactor" in optim:
        return 0 
    if "8bit" in optim:
        return 2
    if "sgd" in optim:
        return 0
    return 8  # AdamW: два fp32-момента


def _precision(cfg: Config, family: str) -> str:
    p = str(cfg.train.get("precision", "auto")).lower()
    if p == "auto":
        return "bf16" 
    return p


def _train_steps(cfg: Config, model, batch: dict, prec: str, device: str, steps: int = 2) -> str:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)
    use_amp = dtype is not None and device == "cuda"
    optim = str(cfg.train.get("optim", "adamw_torch")).lower()
    opt = (torch.optim.Adafactor(model.parameters(), lr=1e-4) if "adafactor" in optim
           else torch.optim.AdamW(model.parameters(), lr=1e-4))
    scaler = torch.amp.GradScaler(device, enabled=use_amp and prec == "fp16")
    try:
        for _ in range(steps):
            ctx = torch.autocast(device, dtype=dtype) if use_amp else contextlib.nullcontext()
            with ctx:
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        return "OOM"
    finally:
        del opt
    return "ok"


def _measure_gpu(cfg: Config, batch_size: int) -> dict:
    from .seq2seq import model_family

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, batch = _model_and_batch(cfg, batch_size, "cuda")
    fam = model_family(model) if cfg.task != "ocr" else "other"
    prec = _precision(cfg, fam)
    if prec == "auto" or (prec == "bf16" and not torch.cuda.is_bf16_supported()):
        prec = "fp32" if fam == "t5" else "fp16"
    status = _train_steps(cfg, model, batch, prec, "cuda")
    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    del model, batch
    gc.collect()
    torch.cuda.empty_cache()
    return {"mode": "measured_on_gpu", "precision": prec, "status": status,
            "peak_allocated_gb": round(peak, 2), "peak_reserved_gb": round(reserved, 2),
            "total_gb": round(reserved + 0.5, 2)}  # + контекст CUDA


def _saved_bytes(cfg: Config, model, batch_size: int, half: bool) -> int:
    _, batch = _model_and_batch_inputs(cfg, model, batch_size)
    params = list(model.parameters())
    param_ptrs = {p.untyped_storage().data_ptr() for p in params}
    weight_shapes = {tuple(p.shape) for p in params if p.dim() == 2}
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor):
        try:
            st = t.untyped_storage()
            if st.data_ptr() in param_ptrs or (t.dim() == 2 and tuple(t.shape) in weight_shapes):
                return t
            seen.setdefault(st.data_ptr(), st.nbytes())
        except (RuntimeError, NotImplementedError):
            pass
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=half):
            loss = model(**batch).loss
    del loss, batch
    gc.collect()
    return sum(seen.values())


def _estimate_cpu(cfg: Config, batch_size: int) -> dict:
    from .seq2seq import model_family

    model, _ = _model_and_batch(cfg, 1, "cpu")
    fam = model_family(model) if cfg.task != "ocr" else "other"
    prec = _precision(cfg, fam)
    half = prec in ("bf16", "fp16")
    n_params = sum(p.numel() for p in model.parameters())
    weight_casts = 2 * sum(p.numel() for p in model.parameters() if p.dim() == 2) if half else 0

    per_example = _saved_bytes(cfg, model, 1, half)
    ckpt = bool(cfg.train.get("gradient_checkpointing", False))
    recompute = 0.0
    if ckpt:
        n_blocks = sum(1 for m in model.modules() if m.__class__.__name__.endswith(("Block", "Layer"))) or 1
        model.gradient_checkpointing_disable()
        recompute = 2 * _saved_bytes(cfg, model, 1, half) / n_blocks
    del model
    gc.collect()

    gb = 1024**3
    weights = 4 * n_params / gb                     
    grads = 4 * n_params / gb
    optim = _optimizer_bytes_per_param(cfg.train.get("optim", "adamw_torch")) * n_params / gb
    activ = (weight_casts + (per_example + recompute) * batch_size) / gb
    allocated = (weights + grads + optim + activ) * FRAGMENTATION
    return {"mode": "estimated_on_cpu", "precision": prec, "params_m": round(n_params / 1e6, 1),
            "weights_gb": round(weights, 2), "grads_gb": round(grads, 2), "optimizer_gb": round(optim, 2),
            "activations_gb": round(activ, 2), "activations_per_example_mb": round((per_example + recompute) / 1024**2, 1),
            "gradient_checkpointing": ckpt, "total_gb": round(allocated + CUDA_OVERHEAD_GB, 2)}


def memcheck(cfg: Config, budget_gb: float | None = None, force_estimate: bool = False) -> dict:
    batch_size = int(cfg.train.batch_size)
    budget = budget_gb or cfg.train.get("max_vram_gb")
    res = (_measure_gpu if torch.cuda.is_available() and not force_estimate else _estimate_cpu)(cfg, batch_size)
    res.update(config=cfg.run_name, task=cfg.task, batch_size=batch_size,
               worst_case_lengths=(f"{cfg.model.image_height}x{cfg.model.image_width} px, {cfg.model.max_target_length} tok"
                                   if cfg.task == "ocr" else f"{cfg.model.max_source_length}/{cfg.model.max_target_length}"))
    if budget:
        res["budget_gb"] = float(budget)
        fits = res["total_gb"] <= float(budget) and res.get("status", "ok") == "ok"
        res["fits"] = fits
        log.info("%s: %.1f ГБ из %.0f ГБ — %s", cfg.run_name, res["total_gb"], float(budget),
                 "ВЛЕЗАЕТ" if fits else "НЕ ВЛЕЗАЕТ: уменьшите batch_size или длины, включите gradient_checkpointing")
    return res
