"""Random materials and initial conditions, solved by the MPI code to build datasets."""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..solver import DATA, Run, execute, read_field, write_field

TAU = 0.02


@dataclass(frozen=True)
class Distribution:
    alpha_min: float = 0.05
    alpha_max: float = 1.0
    max_mode: int = 4
    sharpness: tuple[float, ...] = (0.5, 0.05)
    blobs: tuple[int, int] = (1, 5)
    width: tuple[float, float] = (0.04, 0.15)


IN_DISTRIBUTION = Distribution()
OUT_OF_DISTRIBUTION = Distribution(
    alpha_min=0.01, max_mode=8, sharpness=(0.02,), width=(0.02, 0.06)
)


@dataclass(frozen=True)
class Sample:
    coefficients: np.ndarray
    sharpness: float
    centers: np.ndarray
    widths: np.ndarray
    amplitudes: np.ndarray
    distribution: Distribution


def draw(seed: int, distribution: Distribution) -> Sample:
    rng = np.random.default_rng(seed)
    m = distribution.max_mode
    decay = 1.0 / (1.0 + np.add.outer(np.arange(m + 1) ** 2, np.arange(m + 1) ** 2))
    coefficients = rng.normal(size=(m + 1, m + 1)) * decay
    coefficients[0, 0] = 0.0
    blobs = rng.integers(distribution.blobs[0], distribution.blobs[1] + 1)
    return Sample(
        coefficients=coefficients,
        sharpness=float(rng.choice(distribution.sharpness)),
        centers=rng.uniform(0.1, 0.9, size=(blobs, 2)),
        widths=rng.uniform(*distribution.width, size=blobs),
        amplitudes=rng.uniform(0.3, 1.0, size=blobs),
        distribution=distribution,
    )


def cell_centers(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def render_alpha(sample: Sample, n: int) -> np.ndarray:
    x = cell_centers(n)
    m = sample.coefficients.shape[0]
    basis = np.cos(np.pi * np.outer(np.arange(m), x))
    g = basis.T @ sample.coefficients @ basis
    g = g / np.sqrt(np.sum(sample.coefficients**2) / 4.0 + 1e-12)
    blend = 0.5 * (1.0 + np.tanh(g / sample.sharpness))
    d = sample.distribution
    return np.exp(np.log(d.alpha_min) + blend * (np.log(d.alpha_max) - np.log(d.alpha_min)))


def render_initial(sample: Sample, n: int) -> np.ndarray:
    y, x = np.meshgrid(cell_centers(n), cell_centers(n), indexing="ij")
    field = np.zeros((n, n))
    for center, width, amplitude in zip(
        sample.centers, sample.widths, sample.amplitudes, strict=True
    ):
        distance = (x - center[0]) ** 2 + (y - center[1]) ** 2
        field += amplitude * np.exp(-distance / (2 * width**2))
    return field


def solve(
    initial: np.ndarray, alpha: np.ndarray, tau: float = TAU, ranks: int = 1, cfl: float = 0.25
) -> dict:
    n = initial.shape[0]
    with tempfile.TemporaryDirectory(dir=_workspace()) as tmp:
        directory = Path(tmp)
        write_field(directory / "ic.bin", initial)
        write_field(directory / "alpha.bin", alpha)
        output = directory / "out"
        result = execute(
            Run(
                nx=n,
                ny=n,
                ranks=ranks,
                ic="file",
                medium="file",
                t_end=tau,
                cfl=cfl,
                ic_file=directory / "ic.bin",
                alpha_file=directory / "alpha.bin",
                output=output,
                alpha=float(alpha.max()),
            )
        )
        result["field"] = read_field(output / "T_final.bin", n, n)
    return result


def _workspace() -> Path:
    path = DATA / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_split(
    name: str, count: int, n: int, distribution: Distribution, base_seed: int, workers: int
) -> None:
    DATA.mkdir(exist_ok=True)

    def work(index: int):
        sample = draw(base_seed + index, distribution)
        initial = render_initial(sample, n)
        alpha = render_alpha(sample, n)
        result = solve(initial, alpha)
        return initial, alpha, result["field"], result["wall"], result["energy_final"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(work, range(count)))
    np.savez_compressed(
        DATA / f"{name}.npz",
        initial=np.stack([r[0] for r in rows]).astype(np.float32),
        alpha=np.stack([r[1] for r in rows]).astype(np.float32),
        target=np.stack([r[2] for r in rows]).astype(np.float32),
        solver_seconds=np.array([r[3] for r in rows]),
        seeds=base_seed + np.arange(count),
    )
    print(f"{name}: {count} samples at {n}x{n}", flush=True)


def split_plan(scale: float = 1.0) -> list[tuple]:
    def size(count: int) -> int:
        return max(4, int(count * scale))

    return [
        ("train", size(1600), 64, IN_DISTRIBUTION, 100_000),
        ("val", size(200), 64, IN_DISTRIBUTION, 200_000),
        ("test", size(300), 64, IN_DISTRIBUTION, 300_000),
        ("ood", size(200), 64, OUT_OF_DISTRIBUTION, 400_000),
        ("test128", size(100), 128, IN_DISTRIBUTION, 300_000),
        ("test256", size(40), 256, IN_DISTRIBUTION, 300_000),
    ]


def generate_all(scale: float = 1.0, workers: int | None = None) -> None:
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    plan = split_plan(scale)
    for name, count, n, distribution, seed in plan:
        generate_split(name, count, n, distribution, seed, workers)
    manifest = {
        "tau": TAU,
        "boundary": "neumann",
        "splits": [
            {"name": name, "count": count, "n": n, "seed": seed, "distribution": asdict(dist)}
            for name, count, n, dist, seed in plan
        ],
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_split(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz") as archive:
        return {key: archive[key] for key in archive.files}
