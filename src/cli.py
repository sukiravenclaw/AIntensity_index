from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from src.common import configure_logging
from src.config import ConfigurationError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect public scholarly records for the AI talent research dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="Run the end-to-end pipeline")
    collect.add_argument("--config", default="config.yaml")
    clean = subparsers.add_parser(
        "clean-processed",
        help="Remove generated processed outputs; the raw cache is preserved",
    )
    clean.add_argument("--config", default="config.yaml")
    return parser


def main(argv: Sequence[str] = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "collect":
            from src.pipeline import run

            run(cfg)
        elif args.command == "clean-processed":
            _clean_processed(cfg)
    except ConfigurationError as exc:
        raise SystemExit(f"configuration error: {exc}")
    return 0


def _clean_processed(cfg) -> None:
    root = Path(cfg["_root"]).resolve()
    processed = Path(cfg["paths"]["processed"]).resolve()
    if processed == root or root not in processed.parents:
        raise ValueError(f"Refusing to remove unsafe processed path: {processed}")
    if processed.exists():
        shutil.rmtree(processed)
    processed.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
