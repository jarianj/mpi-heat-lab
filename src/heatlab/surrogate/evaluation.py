"""Held-out evaluation: accuracy, physics violations, and the cost-accuracy trade-off."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..solver import RESULTS
from .baselines import PodRidge, coarse_solver, persistence, spectral_mean_medium
from .dataset import IN_DISTRIBUTION, draw, load_split, render_alpha, render_initial, solve
from .models import MODELS, conserve_energy
from .training import device, load_model


def bootstrap_interval(values: np.ndarray, resamples: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values))).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def score(prediction: np.ndarray, split: dict) -> dict:
    initial, target = split["initial"], split["target"]
    axes = (-2, -1)
    field_error = np.linalg.norm(prediction - target, axis=axes) / np.linalg.norm(target, axis=axes)
    change_error = np.linalg.norm(prediction - target, axis=axes) / np.linalg.norm(
        target - initial, axis=axes
    )
    energy = np.abs(prediction.mean(axis=axes) - initial.mean(axis=axes)) / np.abs(
        initial.mean(axis=axes)
    )
    upper = initial.max(axis=axes)
    lower = initial.min(axis=axes)
    above = np.maximum(prediction.max(axis=axes) - upper, 0.0)
    below = np.maximum(lower - prediction.min(axis=axes), 0.0)
    return {
        "relative_l2": float(field_error.mean()),
        "relative_l2_ci95": bootstrap_interval(field_error),
        "relative_l2_median": float(np.median(field_error)),
        "relative_l2_p95": float(np.quantile(field_error, 0.95)),
        "change_relative_l2": float(change_error.mean()),
        "change_relative_l2_ci95": bootstrap_interval(change_error),
        "max_error": float(np.abs(prediction - target).max(axis=axes).mean()),
        "energy_drift": float(energy.mean()),
        "max_principle_violation": float((above + below).mean()),
    }


@torch.no_grad()
def predict_neural(model, split: dict, target_device, conservative: bool = False) -> np.ndarray:
    outputs = []
    for start in range(0, len(split["initial"]), 8):
        initial = torch.from_numpy(split["initial"][start : start + 8]).to(target_device)
        alpha = torch.from_numpy(split["alpha"][start : start + 8]).to(target_device)
        change = model(initial, alpha)
        if conservative:
            change = conserve_energy(change)
        outputs.append((initial + change).cpu().numpy())
    return np.concatenate(outputs)


def average_neural_scores(name: str, split: dict, seeds, conservative: bool, target_device) -> dict:
    rows = []
    for seed in seeds:
        model = load_model(name, seed, target_device)
        rows.append(score(predict_neural(model, split, target_device, conservative), split))
    keys = [
        "relative_l2",
        "change_relative_l2",
        "max_error",
        "energy_drift",
        "max_principle_violation",
    ]
    combined = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    combined.update(
        {f"{key}_seed_std": float(np.std([row[key] for row in rows], ddof=1)) for key in keys}
    )
    combined["change_relative_l2_ci95"] = rows[0]["change_relative_l2_ci95"]
    combined["relative_l2_ci95"] = rows[0]["relative_l2_ci95"]
    combined["seeds"] = len(rows)
    return combined


def coarse_scores(split: dict, factor: int) -> dict:
    predictions, seconds = [], []
    for initial, alpha in zip(split["initial"], split["alpha"], strict=True):
        field, wall = coarse_solver(initial.astype(np.float64), alpha.astype(np.float64), factor)
        predictions.append(field)
        seconds.append(wall)
    report = score(np.stack(predictions), split)
    report["mean_solver_seconds"] = float(np.mean(seconds))
    return report


def accuracy_tables(seeds) -> dict:
    target_device = device()
    train = load_split("train")
    pod = PodRidge().fit(train["initial"], train["alpha"], train["target"])
    tables = {}
    for split_name in ("test", "ood"):
        split = load_split(split_name)
        table = {
            "persistence": score(persistence(split["initial"], split["alpha"]), split),
            "spectral mean medium": score(
                spectral_mean_medium(split["initial"], split["alpha"]), split
            ),
            "POD-ridge": score(pod.predict(split["initial"], split["alpha"]), split),
            "coarse solver 32x32": coarse_scores(split, 2),
            "coarse solver 16x16": coarse_scores(split, 4),
        }
        for name in MODELS:
            table[name] = average_neural_scores(name, split, seeds, False, target_device)
            table[f"{name} conservative"] = average_neural_scores(
                name, split, seeds, True, target_device
            )
        tables[split_name] = table
    return tables


def resolution_transfer(seeds) -> dict:
    target_device = device()
    report = {}
    for split_name, n in (("test", 64), ("test128", 128), ("test256", 256)):
        split = load_split(split_name)
        report[str(n)] = {
            "fno conservative": average_neural_scores("fno", split, seeds, True, target_device),
            "spectral mean medium": score(
                spectral_mean_medium(split["initial"], split["alpha"]), split
            ),
        }
    return report


def cost_accuracy(seeds) -> list[dict]:
    target_device = device()
    rows = []
    plan = (("test", 64, (2, 4)), ("test128", 128, (2, 4, 8)), ("test256", 256, (2, 4, 8, 16)))
    for split_name, n, factors in plan:
        split = load_split(split_name)
        for factor in factors:
            result = coarse_scores(split, factor)
            rows.append(
                {
                    "n": n,
                    "method": "coarse solver",
                    "coarse_n": n // factor,
                    "change_relative_l2": result["change_relative_l2"],
                    "seconds": result["mean_solver_seconds"],
                }
            )
        neural = average_neural_scores("fno", split, seeds, True, target_device)
        rows.append(
            {
                "n": n,
                "method": "fno conservative",
                "change_relative_l2": neural["change_relative_l2"],
                "change_relative_l2_seed_std": neural["change_relative_l2_seed_std"],
            }
        )
        print(rows[-1], flush=True)
    return rows


def _timed(function, repeats: int = 7) -> float:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def timing_table(sizes=(64, 128, 256, 512), samples: int = 3) -> list[dict]:
    rows = []
    gpu = torch.cuda.is_available()
    cpu_model = load_model("fno", 0, torch.device("cpu"))
    gpu_model = load_model("fno", 0, torch.device("cuda")) if gpu else None
    for n in sizes:
        for index in range(samples):
            sample = draw(300_000 + index, IN_DISTRIBUTION)
            initial = render_initial(sample, n)
            alpha = render_alpha(sample, n)
            solver_1 = solve(initial, alpha, ranks=1, cfl=0.9)
            solver_8 = solve(initial, alpha, ranks=8, cfl=0.9)
            initial_t = torch.tensor(initial[None], dtype=torch.float32)
            alpha_t = torch.tensor(alpha[None], dtype=torch.float32)

            def run_cpu(initial_t=initial_t, alpha_t=alpha_t):
                with torch.no_grad():
                    cpu_model(initial_t, alpha_t)

            row = {
                "n": n,
                "sample": index,
                "steps": solver_1["completed_steps"],
                "solver_1_rank": solver_1["wall"],
                "solver_8_ranks": solver_8["wall"],
                "fno_cpu": _timed(run_cpu),
            }
            if gpu:
                initial_g, alpha_g = initial_t.cuda(), alpha_t.cuda()

                def run_gpu(initial_g=initial_g, alpha_g=alpha_g):
                    with torch.no_grad():
                        gpu_model(initial_g, alpha_g)
                    torch.cuda.synchronize()

                run_gpu()
                row["fno_gpu"] = _timed(run_gpu)
            rows.append(row)
            print(row, flush=True)
    return rows


def evaluate_all(seeds=(0, 1, 2)) -> dict:
    report = {
        "accuracy": accuracy_tables(seeds),
        "resolution_transfer": resolution_transfer(seeds),
        "cost_accuracy": cost_accuracy(seeds),
        "timing": timing_table(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "surrogate.json").write_text(json.dumps(report, indent=2))
    return report


__all__ = ["evaluate_all", "solve"]
