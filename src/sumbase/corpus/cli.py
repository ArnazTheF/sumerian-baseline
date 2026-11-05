from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description="Collect, reconcile and export Sumerian corpora")
    p.add_argument("--root", type=Path, default=Path("data/collection"))
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--group", choices=["text", "photos", "supplement", "translations", "bdtns", "inventory"], default="text")
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--oracc-scope", choices=["sumerian", "all"], default="sumerian")
    f.add_argument("--image-budget-gb", type=float, default=4.0)
    sub.add_parser("build")
    sub.add_parser("audit")
    sub.add_parser("verify")
    sub.add_parser("report")
    args = p.parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command == "fetch":
        from .acquire import fetch_photos, fetch_text, fetch_supplement, fetch_translations, fetch_bdtns, fetch_inventory
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
        from .report import verify, report
        (verify if args.command == "verify" else report)(args.root)
    else:
        from .build import build, audit
        (build if args.command == "build" else audit)(args.root)
