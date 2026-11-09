"""Numerical verification against exact solutions and solver invariants."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from .solver import RESULTS, Run, execute, read_field

RANK_COUNTS = (1, 2, 3, 4, 6, 8, 12)


def observed_order(spacings, errors) -> float:
    return float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])


def spatial_convergence() -> list[dict]:
    cases = [
        ("1D neumann", 1, "neumann", (2, 0, 0), (32, 64, 128, 256, 512)),
        ("1D dirichlet", 1, "dirichlet", (2, 1, 1), (32, 64, 128, 256, 512)),
        ("1D periodic", 1, "periodic", (2, 0, 0), (32, 64, 128, 256, 512)),
        ("2D neumann", 2, "neumann", (1, 2, 0), (32, 64, 128, 256)),
        ("2D dirichlet", 2, "dirichlet", (1, 2, 1), (32, 64, 128, 256)),
        ("2D periodic", 2, "periodic", (1, 2, 0), (32, 64, 128, 256)),
        ("3D neumann", 3, "neumann", (1, 1, 1), (16, 32, 64, 128)),
        ("3D dirichlet", 3, "dirichlet", (1, 1, 1), (16, 32, 64, 128)),
        ("3D periodic", 3, "periodic", (1, 1, 1), (16, 32, 64, 128)),
    ]
    rows = []
    for label, dimension, bc, mode, sizes in cases:
        errors = []
        for n in sizes:
            shape = [n if axis < dimension else 1 for axis in range(3)]
            run = Run(
                nx=shape[0],
                ny=shape[1],
                nz=shape[2],
                ranks=4,
                bc=bc,
                ic="mode",
                mode=mode,
                t_end=0.005,
                cfl=0.5,
            )
            errors.append(execute(run)["l2_error"])
        rows.append(
            {
                "case": label,
                "n": list(sizes),
                "l2_error": errors,
                "order": observed_order(1.0 / np.array(sizes), errors),
            }
        )
    return rows


def temporal_convergence() -> list[dict]:
    steps = (12, 25, 50, 100, 200)
    rows = []
    for scheme, expected in (("cn", 2), ("be", 1)):
        errors = []
        for count in steps:
            dt = 0.1 / count
            run = Run(
                nx=8192,
                ranks=4,
                scheme=scheme,
                ic="mode",
                mode=(1, 0, 0),
                t_end=0.1,
                dt=dt,
                cg_tol=1e-13,
            )
            errors.append(execute(run)["l2_error"])
        rows.append(
            {
                "scheme": scheme,
                "expected_order": expected,
                "dt": [0.1 / count for count in steps],
                "l2_error": errors,
                "order": observed_order(0.1 / np.array(steps), errors),
            }
        )
    return rows


def _final_field_bytes(run: Run, directory: Path) -> bytes:
    output = directory / "out"
    execute(replace(run, output=output))
    return (output / "T_final.bin").read_bytes()


def decomposition_independence() -> list[dict]:
    cases = [
        ("1D uneven grid", Run(nx=1021, ic="step", t_end=0.001)),
        ("2D uneven grid", Run(nx=127, ny=101, ic="gaussian", t_end=0.002)),
        ("3D uneven grid", Run(nx=43, ny=37, nz=31, ic="gaussian", t_end=0.002)),
        ("2D periodic", Run(nx=120, ny=96, bc="periodic", ic="gaussian", t_end=0.002)),
        ("2D dirichlet", Run(nx=120, ny=96, bc="dirichlet", ic="gaussian", t_end=0.002)),
        (
            "2D inclusion",
            Run(nx=128, ny=128, medium="inclusion", alpha=1.0, alpha2=0.05, t_end=0.002),
        ),
    ]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for label, base in cases:
            reference = hashlib.sha256(_final_field_bytes(base, directory)).hexdigest()
            matches = {}
            for ranks in RANK_COUNTS[1:]:
                for halo in ("sendrecv", "overlap"):
                    digest = hashlib.sha256(
                        _final_field_bytes(replace(base, ranks=ranks, halo=halo), directory)
                    ).hexdigest()
                    matches[f"{ranks}/{halo}"] = digest == reference
            rows.append(
                {
                    "case": label,
                    "reference_sha256": reference,
                    "comparisons": len(matches),
                    "bit_identical": sum(matches.values()),
                    "details": matches,
                }
            )
    return rows


def energy_conservation() -> list[dict]:
    cases = [
        ("explicit uniform", Run(nx=256, ny=256, t_end=0.01)),
        ("explicit inclusion", Run(nx=256, ny=256, medium="inclusion", alpha2=0.01, t_end=0.01)),
        (
            "crank-nicolson layers",
            Run(
                nx=256,
                ny=256,
                scheme="cn",
                medium="layers",
                alpha2=0.1,
                cfl=10,
                t_end=0.01,
                cg_tol=1e-13,
            ),
        ),
        (
            "backward euler uniform",
            Run(nx=64, ny=64, nz=64, scheme="be", cfl=10, t_end=0.01, cg_tol=1e-13),
        ),
    ]
    rows = []
    for label, base in cases:
        for ranks in (1, 4, 8):
            result = execute(replace(base, ranks=ranks))
            rows.append(
                {
                    "case": label,
                    "ranks": ranks,
                    "energy_initial": result["energy_initial"],
                    "energy_final": result["energy_final"],
                    "relative_drift": abs(result["energy_final"] - result["energy_initial"])
                    / abs(result["energy_initial"]),
                }
            )
    return rows


def composite_slab() -> dict:
    alpha_left, alpha_right = 1.0, 0.1
    n = 128
    x = (np.arange(n) + 0.5) / n
    flux = 1.0 / (0.5 / alpha_left + 0.5 / alpha_right)
    exact = np.where(
        x < 0.5, flux * x / alpha_left, 0.5 * flux / alpha_left + flux * (x - 0.5) / alpha_right
    )
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out"
        run = Run(
            nx=n,
            ranks=4,
            scheme="be",
            bc="dirichlet",
            ic="uniform",
            medium="layers",
            alpha=alpha_left,
            alpha2=alpha_right,
            bc_x0=0.0,
            bc_x1=1.0,
            t_end=50.0,
            dt=1.0,
            cg_tol=1e-13,
            output=output,
        )
        execute(run)
        field = read_field(output / "T_final.bin", n)
    return {
        "n": n,
        "alpha": [alpha_left, alpha_right],
        "linf_error": float(np.abs(field - exact).max()),
        "exact_interface_temperature": 0.5 * flux / alpha_left,
    }


def stability_boundary() -> list[dict]:
    rows = []
    cases = [
        ("explicit", 0.50),
        ("explicit", 0.99),
        ("explicit", 1.00),
        ("explicit", 1.01),
        ("explicit", 1.10),
        ("cn", 50.0),
        ("be", 50.0),
    ]
    for scheme, cfl in cases:
        run = Run(
            nx=128,
            ny=128,
            ranks=4,
            scheme=scheme,
            ic="gaussian",
            cfl=cfl,
            steps=6000,
            diag_every=100,
            allow_unstable=True,
        )
        result = execute(run)
        rows.append(
            {
                "scheme": scheme,
                "cfl": cfl,
                "status": result["status"],
                "completed_steps": result["completed_steps"],
                "max_abs": max(abs(result["max"]), abs(result["min"])),
            }
        )
    return rows


def original_defects() -> dict:
    uniform = [
        execute(Run(nx=1024, ranks=ranks, ic="uniform", steps=100))["energy_final"]
        for ranks in (1, 3, 4, 7)
    ]
    return {
        "description": "Uniform start and non-divisible rank counts from the original program",
        "energy_at_ranks_1_3_4_7": uniform,
    }


def run_all() -> dict:
    report = {
        "spatial_convergence": spatial_convergence(),
        "temporal_convergence": temporal_convergence(),
        "decomposition_independence": decomposition_independence(),
        "energy_conservation": energy_conservation(),
        "composite_slab": composite_slab(),
        "stability_boundary": stability_boundary(),
        "original_defects": original_defects(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "verification.json").write_text(json.dumps(report, indent=2))
    return report


def summary(report: dict) -> str:
    lines = ["Spatial convergence (expected order 2):"]
    lines += [f"  {r['case']:<14} {r['order']:.3f}" for r in report["spatial_convergence"]]
    lines.append("Temporal convergence:")
    lines += [
        f"  {r['scheme']:<14} {r['order']:.3f} (expected {r['expected_order']})"
        for r in report["temporal_convergence"]
    ]
    total = sum(r["comparisons"] for r in report["decomposition_independence"])
    same = sum(r["bit_identical"] for r in report["decomposition_independence"])
    lines.append(f"Bit-identical decompositions: {same}/{total}")
    worst = max(r["relative_drift"] for r in report["energy_conservation"])
    lines.append(f"Worst relative energy drift: {worst:.2e}")
    lines.append(f"Composite slab Linf error: {report['composite_slab']['linf_error']:.2e}")
    return "\n".join(lines)
