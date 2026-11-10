from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import click


def dispatch(args):
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command == "fetch":
        from .acquire import (
            fetch_bdtns,
            fetch_inventory,
            fetch_photos,
            fetch_supplement,
            fetch_text,
            fetch_translations,
        )

        if args.group == "text":
            fetch_text(args.root / "raw", args.workers, args.oracc_scope)
        elif args.group == "photos":
            fetch_photos(args.root / "raw", args.workers, args.image_budget_gb)
        elif args.group == "supplement":
            fetch_supplement(args.root / "raw", args.workers)
        elif args.group == "translations":
            fetch_translations(args.root / "raw", args.workers)
        elif args.group == "bdtns":
            fetch_bdtns(args.root / "raw", args.workers)
        else:
            fetch_inventory(args.root / "raw", args.workers)
    elif args.command in {"verify", "report"}:
        from .report import report, verify

        (verify if args.command == "verify" else report)(args.root)
    else:
        from .build import audit, build

        (build if args.command == "build" else audit)(args.root)


@click.group()
@click.option("--root", type=click.Path(path_type=Path), default=Path("data/collection"))
@click.pass_context
def cli(context, root):
    """Collect, reconcile and export Sumerian corpora."""
    context.ensure_object(dict)
    context.obj["root"] = root


@cli.command()
@click.option(
    "--group",
    type=click.Choice(["text", "photos", "supplement", "translations", "bdtns", "inventory"]),
    default="text",
)
@click.option("--workers", type=click.IntRange(min=1), default=4)
@click.option("--oracc-scope", type=click.Choice(["sumerian", "all"]), default="sumerian")
@click.option("--image-budget-gb", type=float, default=4.0)
@click.pass_context
def fetch(context, **kwargs):
    dispatch(SimpleNamespace(command="fetch", root=context.obj["root"], **kwargs))


def register_command(name):
    @cli.command(name=name)
    @click.pass_context
    def command(context):
        dispatch(SimpleNamespace(command=name, root=context.obj["root"]))


def main(argv=None):
    for command_name in ("build", "audit", "verify", "report"):
        register_command(command_name)
    cli.main(args=argv, prog_name="sumbase-corpus")
