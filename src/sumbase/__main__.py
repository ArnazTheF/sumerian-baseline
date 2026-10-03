from __future__ import annotations

import argparse
import json
import sys


def _add_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True, help="путь к YAML-конфигу")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                   help="переопределить параметр конфига (можно несколько раз)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sumbase", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="информация об окружении")
    for name in ("prepare", "train", "evaluate", "baseline", "memcheck"):
        p = sub.add_parser(name)
        _add_config(p)
        if name == "train":
            p.add_argument("--no-test", action="store_true", help="не оценивать на test после обучения")
        if name in ("evaluate", "baseline"):
            p.add_argument("--split", default="test", choices=["valid", "test"])
        if name == "baseline":
            p.add_argument("--kind", default="tm", choices=["tm", "zeroshot"])
        if name == "memcheck":
            p.add_argument("--budget-gb", type=float, default=None, help="бюджет видеопамяти, ГБ")
            p.add_argument("--estimate", action="store_true", help="только оценка на CPU, даже если есть GPU")
    p = sub.add_parser("predict")
    p.add_argument("--run", help="каталог обученной модели, например runs/mt_byt5_small")
    p.add_argument("--text", action="append", help="входная строка (можно несколько раз)")
    p.add_argument("--input", help="файл, одна строка — один пример")
    p.add_argument("--image", nargs="+", help="изображения для OCR")
    p.add_argument("--num-beams", type=int, default=None)
    p.add_argument("--output", help="куда сохранить результат")
    p.add_argument("--pipeline", action="store_true", help="OCR → транслитерация → перевод")
    p.add_argument("--ocr-run")
    p.add_argument("--translit-run")
    p.add_argument("--mt-run")

    args = parser.parse_args(argv)

    if args.cmd == "info":
        from .utils import device_summary

        print(json.dumps(device_summary(), ensure_ascii=False, indent=2))
        return
    if args.cmd == "predict":
        from .predict import run_predict

        run_predict(args)
        return

    from .config import load_config

    cfg = load_config(args.config, args.override)
    if args.cmd == "memcheck":
        from .memcheck import memcheck

        result = memcheck(cfg, args.budget_gb, args.estimate)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0 if result.get("fits", True) else 1)
    if args.cmd == "prepare":
        from .data import prepare

        result = prepare(cfg)
    elif args.cmd == "train":
        if cfg.task == "ocr":
            from .ocr import train_ocr

            result = train_ocr(cfg, run_test=not args.no_test)
        else:
            from .seq2seq import train_text

            result = train_text(cfg, run_test=not args.no_test)
    elif args.cmd == "evaluate":
        if cfg.task == "ocr":
            from .ocr import evaluate_ocr

            result = evaluate_ocr(cfg, cfg.paths.output_dir, args.split)
        else:
            from .seq2seq import evaluate_text

            result = evaluate_text(cfg, cfg.paths.output_dir, args.split)
    else:
        if cfg.task == "ocr" and args.kind == "tm":
            sys.exit("translation memory определена только для текстовых задач; для ocr есть --kind zeroshot")
        from . import baselines

        result = (baselines.translation_memory if args.kind == "tm" else baselines.zeroshot)(cfg, args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
