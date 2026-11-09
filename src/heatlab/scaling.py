"""Strong, weak, halo, and decomposition benchmarks with randomized execution order."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .solver import RESULTS, Run, SolverError, execute

RANKS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16)


@dataclass(frozen=True)
class Case:
    experiment: str
    label: str
    nx: int
    ny: int
    nz: int
    ranks: int
    steps: int
    halo: str = "overlap"
    dims: tuple[int, int, int] | None = None


def balanced_dims(ranks: int) -> tuple[int, int]:
    a = int(np.sqrt(ranks))
    while ranks % a:
        a -= 1
    return ranks // a, a


def strong_cases() -> list[Case]:
    problems = [
        ("1D 2^24", 2**24, 1, 1, 200),
        ("2D 2048^2", 2048, 2048, 1, 400),
        ("3D 192^3", 192, 192, 192, 300),
    ]
    return [
        Case("strong", label, nx, ny, nz, ranks, steps)
        for label, nx, ny, nz, steps in problems
        for ranks in RANKS
    ]


def weak_cases() -> list[Case]:
    cases = []
    for ranks in RANKS:
        a, b = balanced_dims(ranks)
        cases.append(
            Case("weak", "2D 512^2 per rank", 512 * a, 512 * b, 1, ranks, 1000, dims=(a, b, 1))
        )
    return cases


def halo_cases() -> list[Case]:
    cases = []
    for ranks in (2, 4, 8, 12, 16):
        for halo in ("chain", "sendrecv", "overlap"):
            cases.append(
                Case("halo", "1D 4096 cells per rank", 4096 * ranks, 1, 1, ranks, 100000, halo)
            )
            cases.append(Case("halo", "2D 1024^2 total", 1024, 1024, 1, ranks, 1000, halo))
    return cases


def decomposition_cases() -> list[Case]:
    layouts = {
        8: [(8, 1, 1), (4, 2, 1), (2, 2, 2)],
        12: [(12, 1, 1), (4, 3, 1), (3, 2, 2)],
        16: [(16, 1, 1), (4, 4, 1), (4, 2, 2)],
    }
    return [
        Case(
            "decomposition", f"{dims[0]}x{dims[1]}x{dims[2]}", 192, 192, 192, ranks, 300, dims=dims
        )
        for ranks, options in layouts.items()
        for dims in options
    ]


def sweep_cases() -> list[Case]:
    cases = []
    sizes = {
        1: [2**16, 2**18, 2**20, 2**22],
        2: [256, 512, 1024, 2048],
        3: [48, 96, 144, 192],
    }
    targets = {1: 4e8, 2: 4e8, 3: 4e8}
    for dimension, grid_sizes in sizes.items():
        for n in grid_sizes:
            cells = n**dimension
            steps = max(50, int(targets[dimension] / cells))
            for ranks in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16):
                for halo in ("sendrecv", "overlap"):
                    shape = [n if axis < dimension else 1 for axis in range(3)]
                    cases.append(
                        Case(
                            "sweep",
                            f"{dimension}D n={n}",
                            shape[0],
                            shape[1],
                            shape[2],
                            ranks,
                            steps,
                            halo,
                        )
                    )
    return cases


def run_case(case: Case) -> dict:
    run = Run(
        nx=case.nx,
        ny=case.ny,
        nz=case.nz,
        ranks=case.ranks,
        steps=case.steps,
        cfl=0.5,
        halo=case.halo,
        dims=case.dims,
    )
    result = execute(run, timeout=1800)
    row = asdict(case)
    row.pop("dims")
    row.update(
        dims="x".join(str(d) for d in result["dims"]),
        wall=result["wall"],
        compute_max=result["compute_max"],
        compute_mean=result["compute_mean"],
        halo_max=result["halo_max"],
        halo_mean=result["halo_mean"],
        mcups=result["mcups"],
    )
    return row


def run_cases(
    cases: list[Case], repeats: int, seed: int = 2026, name: str = "scaling", resume: bool = False
) -> pd.DataFrame:
    schedule = [(case, repeat) for case in cases for repeat in range(repeats)]
    random.Random(seed).shuffle(schedule)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}_raw.csv"
    rows = pd.read_csv(path).to_dict("records") if resume and path.exists() else []
    done = {row["order"] for row in rows}
    for index, (case, repeat) in enumerate(schedule, 1):
        if index in done:
            continue
        try:
            row = run_case(case)
        except SolverError:
            time.sleep(5)
            row = run_case(case)
        row["repeat"] = repeat
        row["order"] = index
        rows.append(row)
        if index % 25 == 0 or index == len(schedule):
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{name}: {index}/{len(schedule)}", flush=True)
    return pd.DataFrame(rows)


def universal_scalability(p, sigma, kappa):
    return p / (1.0 + sigma * (p - 1.0) + kappa * p * (p - 1.0))


def analyze(raw: pd.DataFrame) -> dict:
    report: dict = {"strong": {}, "weak": {}}
    strong = raw[raw.experiment == "strong"]
    for label, group in strong.groupby("label"):
        medians = group.groupby("ranks").wall.median()
        low = group.groupby("ranks").wall.quantile(0.1)
        high = group.groupby("ranks").wall.quantile(0.9)
        base = medians.loc[1]
        ranks = medians.index.to_numpy(dtype=float)
        speedup = base / medians.to_numpy()
        (sigma, kappa), _ = curve_fit(
            universal_scalability, ranks, speedup, p0=(0.05, 0.001), bounds=(0, 1)
        )
        fit = universal_scalability(ranks, sigma, kappa)
        report["strong"][label] = {
            "ranks": ranks.astype(int).tolist(),
            "wall_median": medians.tolist(),
            "wall_p10": low.tolist(),
            "wall_p90": high.tolist(),
            "speedup": speedup.tolist(),
            "efficiency": (speedup / ranks).tolist(),
            "halo_share": (group.groupby("ranks").halo_max.median() / medians).tolist(),
            "usl_sigma": float(sigma),
            "usl_kappa": float(kappa),
            "usl_r2": float(
                1 - np.sum((speedup - fit) ** 2) / np.sum((speedup - speedup.mean()) ** 2)
            ),
            "usl_peak_ranks": float(np.sqrt((1 - sigma) / kappa)) if kappa > 0 else None,
        }
    weak = raw[raw.experiment == "weak"]
    for label, group in weak.groupby("label"):
        medians = group.groupby("ranks").wall.median()
        report["weak"][label] = {
            "ranks": medians.index.astype(int).tolist(),
            "wall_median": medians.tolist(),
            "efficiency": (medians.iloc[0] / medians).tolist(),
        }
    for experiment in ("halo", "decomposition"):
        subset = raw[raw.experiment == experiment]
        table = (
            subset.groupby(["label", "ranks", "halo", "dims"])
            .agg(
                wall=("wall", "median"),
                halo_max=("halo_max", "median"),
                compute_max=("compute_max", "median"),
                p10=("wall", lambda s: s.quantile(0.1)),
                p90=("wall", lambda s: s.quantile(0.9)),
            )
            .reset_index()
        )
        report[experiment] = table.to_dict(orient="records")
    return report


def run_benchmarks(repeats: int = 5) -> dict:
    cases = strong_cases() + weak_cases() + halo_cases() + decomposition_cases()
    raw = run_cases(cases, repeats)
    report = analyze(raw)
    (RESULTS / "scaling.json").write_text(json.dumps(report, indent=2))
    return report


def run_sweep(repeats: int = 3) -> pd.DataFrame:
    return run_cases(sweep_cases(), repeats, seed=2027, name="sweep", resume=True)
