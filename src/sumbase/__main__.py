from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import click


def dispatch(args):
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
            sys.exit(
                "translation memory определена только для текстовых задач; для ocr есть --kind zeroshot"
            )
        from . import baselines

        result = (baselines.translation_memory if args.kind == "tm" else baselines.zeroshot)(
            cfg, args.split
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


@click.group()
def cli():
    """Подготовка данных, обучение и предсказание для шумерских текстов."""


def config_options(function):
    function = click.option("--override", "-o", multiple=True, help="Hydra key=value")(function)
    return click.option("--config", "-c", required=True, type=click.Path(exists=True))(function)


def invoke(command, **kwargs):
    dispatch(SimpleNamespace(cmd=command, **kwargs))


@cli.command()
def info():
    invoke("info")


@cli.command()
@config_options
def prepare(**kwargs):
    invoke("prepare", **kwargs)


@cli.command()
@config_options
@click.option("--no-test", is_flag=True)
def train(**kwargs):
    invoke("train", **kwargs)


@cli.command()
@config_options
@click.option("--split", type=click.Choice(["valid", "test"]), default="test")
def evaluate(**kwargs):
    invoke("evaluate", **kwargs)


@cli.command()
@config_options
@click.option("--split", type=click.Choice(["valid", "test"]), default="test")
@click.option("--kind", type=click.Choice(["tm", "zeroshot"]), default="tm")
def baseline(**kwargs):
    invoke("baseline", **kwargs)


@cli.command()
@config_options
@click.option("--budget-gb", type=float)
@click.option("--estimate", is_flag=True)
def memcheck(**kwargs):
    invoke("memcheck", **kwargs)


@cli.command()
@click.option("--run", type=click.Path())
@click.option("--text", multiple=True)
@click.option("--input", type=click.Path(exists=True))
@click.option("--image", multiple=True, type=click.Path(exists=True))
@click.option("--num-beams", type=int)
@click.option("--output", type=click.Path())
@click.option("--pipeline", is_flag=True)
@click.option("--ocr-run", type=click.Path())
@click.option("--translit-run", type=click.Path())
@click.option("--mt-run", type=click.Path())
def predict(**kwargs):
    invoke("predict", **kwargs)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    cli.main(args=argv, prog_name="sumbase")


if __name__ == "__main__":
    main()
