"""Command line entry point: python -m heatlab <command>."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .solver import ROOT


def verify(args) -> None:
    from .verification import run_all, summary

    print(summary(run_all()))


def scale(args) -> None:
    from .scaling import run_benchmarks

    run_benchmarks(repeats=args.repeats)


def sweep(args) -> None:
    from .scaling import run_sweep

    run_sweep(repeats=args.repeats)


def dataset(args) -> None:
    from .surrogate.dataset import generate_all

    generate_all(scale=args.scale, workers=args.workers)


def train(args) -> None:
    from .surrogate.training import train_all

    train_all(seeds=args.seeds, epochs=args.epochs)


def evaluate(args) -> None:
    from .surrogate.evaluation import evaluate_all

    evaluate_all()


def perf(args) -> None:
    from .perfmodel import fit_and_validate

    print(fit_and_validate())


def figures(args) -> None:
    from .figures import render_all

    render_all()


def dashboard(args) -> None:
    app = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(prog="heatlab")
    commands = parser.add_subparsers(required=True)

    commands.add_parser("verify", help="check the solver against exact solutions").set_defaults(
        action=verify
    )
    scale_parser = commands.add_parser("scale", help="run scaling benchmarks")
    scale_parser.add_argument("--repeats", type=int, default=5)
    scale_parser.set_defaults(action=scale)
    sweep_parser = commands.add_parser("sweep", help="collect performance model data")
    sweep_parser.add_argument("--repeats", type=int, default=3)
    sweep_parser.set_defaults(action=sweep)
    dataset_parser = commands.add_parser("dataset", help="generate surrogate data with the solver")
    dataset_parser.add_argument("--scale", type=float, default=1.0)
    dataset_parser.add_argument("--workers", type=int, default=8)
    dataset_parser.set_defaults(action=dataset)
    train_parser = commands.add_parser("train", help="train neural surrogates")
    train_parser.add_argument("--seeds", type=int, default=3)
    train_parser.add_argument("--epochs", type=int, default=150)
    train_parser.set_defaults(action=train)
    commands.add_parser("evaluate", help="evaluate surrogates and baselines").set_defaults(
        action=evaluate
    )
    commands.add_parser("perf", help="fit and validate the runtime model").set_defaults(action=perf)
    commands.add_parser("figures", help="render report figures").set_defaults(action=figures)
    commands.add_parser("dashboard", help="open the interactive dashboard").set_defaults(
        action=dashboard
    )

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
