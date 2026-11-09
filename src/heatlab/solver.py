"""Run the compiled MPI solver from Python."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
MODELS = ROOT / "models"


class SolverError(RuntimeError):
    pass


@dataclass(frozen=True)
class Run:
    nx: int
    ny: int = 1
    nz: int = 1
    ranks: int = 1
    scheme: str = "explicit"
    halo: str = "overlap"
    bc: str = "neumann"
    ic: str = "gaussian"
    medium: str = "uniform"
    alpha: float = 1.0
    alpha2: float = 1.0
    t_end: float | None = None
    steps: int | None = None
    dt: float | None = None
    cfl: float | None = None
    mode: tuple[int, int, int] | None = None
    dims: tuple[int, int, int] | None = None
    bc_x0: float | None = None
    bc_x1: float | None = None
    cg_tol: float | None = None
    ic_file: Path | None = None
    alpha_file: Path | None = None
    output: Path | None = None
    snapshot_every: int | None = None
    diag_every: int | None = None
    allow_unstable: bool = False

    def arguments(self) -> list[str]:
        args = ["--nx", self.nx, "--ny", self.ny, "--nz", self.nz, "--scheme", self.scheme]
        args += ["--halo", self.halo, "--bc", self.bc, "--ic", self.ic, "--medium", self.medium]
        args += ["--alpha", self.alpha, "--alpha2", self.alpha2]
        optional = {
            "--t-end": self.t_end,
            "--steps": self.steps,
            "--dt": self.dt,
            "--cfl": self.cfl,
            "--bc-x0": self.bc_x0,
            "--bc-x1": self.bc_x1,
            "--cg-tol": self.cg_tol,
            "--ic-file": self.ic_file,
            "--alpha-file": self.alpha_file,
            "--output": self.output,
            "--snapshot-every": self.snapshot_every,
            "--diag-every": self.diag_every,
        }
        for key, value in optional.items():
            if value is not None:
                args += [key, value]
        if self.mode is not None:
            args += ["--kx", self.mode[0], "--ky", self.mode[1], "--kz", self.mode[2]]
        if self.dims is not None:
            args += ["--dims", ",".join(str(d) for d in self.dims)]
        if self.allow_unstable:
            args.append("--allow-unstable")
        return [str(a) for a in args]


def binary() -> Path:
    override = os.environ.get("HEATLAB_BINARY")
    if override:
        return Path(override)
    for name in ("heatsolve.exe", "heatsolve"):
        path = ROOT / "bin" / name
        if path.exists():
            return path
    raise SolverError("solver binary not found: run `make build`")


@lru_cache(maxsize=1)
def launcher() -> tuple[str, ...]:
    executable = os.environ.get("HEATLAB_MPIEXEC") or shutil.which("mpiexec")
    if executable is None:
        raise SolverError("mpiexec not found on PATH")
    command: tuple[str, ...] = (executable,)
    try:
        version = subprocess.run([executable, "--version"], capture_output=True, text=True).stdout
    except OSError:
        version = ""
    if "Open MPI" in version or "OpenRTE" in version:
        command += ("--oversubscribe",)
    return command


def command_for(run_config: Run) -> list[str]:
    return [*launcher(), "-n", str(run_config.ranks), str(binary()), *run_config.arguments()]


def execute(run_config: Run, timeout: float = 600.0) -> dict:
    completed = subprocess.run(
        command_for(run_config), capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise SolverError(detail or f"solver exited with {completed.returncode}")
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise SolverError(f"solver produced no result: {completed.stdout!r}")
    return json.loads(lines[-1])


def write_field(path: Path, field: np.ndarray) -> None:
    np.asarray(field, dtype=np.float64).tofile(path)


def read_field(path: Path, nx: int, ny: int = 1, nz: int = 1) -> np.ndarray:
    return np.squeeze(np.fromfile(path, dtype=np.float64).reshape(nz, ny, nx))
