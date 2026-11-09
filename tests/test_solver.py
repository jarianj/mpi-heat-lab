from dataclasses import replace

import numpy as np
import pytest

from heatlab.solver import Run, SolverError, execute, read_field, write_field
from heatlab.verification import observed_order


def final_field(run: Run, tmp_path, name: str) -> bytes:
    output = tmp_path / name
    execute(replace(run, output=output))
    return (output / "T_final.bin").read_bytes()


@pytest.mark.parametrize(
    "bc,mode", [("neumann", (1, 2, 0)), ("dirichlet", (1, 2, 1)), ("periodic", (1, 1, 0))]
)
def test_explicit_scheme_is_second_order(bc, mode):
    sizes = (32, 64, 128)
    errors = [
        execute(Run(nx=n, ny=n, ranks=2, bc=bc, ic="mode", mode=mode, t_end=0.005, cfl=0.5))[
            "l2_error"
        ]
        for n in sizes
    ]
    assert observed_order(1 / np.array(sizes), errors) == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("scheme,expected", [("cn", 2.0), ("be", 1.0)])
def test_implicit_schemes_reach_design_order(scheme, expected):
    steps = np.array([25, 50, 100])
    errors = [
        execute(
            Run(
                nx=4096,
                ranks=2,
                scheme=scheme,
                ic="mode",
                mode=(1, 0, 0),
                t_end=0.1,
                dt=0.1 / count,
                cg_tol=1e-13,
            )
        )["l2_error"]
        for count in steps
    ]
    assert observed_order(0.1 / steps, errors) == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("ranks", [2, 3, 4, 7])
@pytest.mark.parametrize("halo", ["chain", "sendrecv", "overlap"])
def test_results_are_bit_identical_across_decompositions(tmp_path, ranks, halo):
    base = Run(nx=101, ny=83, ic="step", t_end=0.002)
    assert final_field(base, tmp_path, "serial") == final_field(
        replace(base, ranks=ranks, halo=halo), tmp_path, "parallel"
    )


def test_three_dimensional_periodic_decomposition(tmp_path):
    base = Run(nx=24, ny=20, nz=18, bc="periodic", ic="gaussian", t_end=0.002)
    assert final_field(base, tmp_path, "serial") == final_field(
        replace(base, ranks=8), tmp_path, "parallel"
    )


@pytest.mark.parametrize("ranks", [1, 3, 7])
def test_non_divisible_rank_counts_keep_every_cell(ranks):
    result = execute(Run(nx=1024, ranks=ranks, ic="uniform", steps=100))
    assert result["energy_final"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("medium,alpha2", [("uniform", 1.0), ("inclusion", 0.01), ("layers", 0.1)])
def test_insulated_domain_conserves_energy(medium, alpha2):
    result = execute(Run(nx=128, ny=128, ranks=4, medium=medium, alpha2=alpha2, t_end=0.005))
    drift = abs(result["energy_final"] - result["energy_initial"]) / result["energy_initial"]
    assert drift < 1e-12


def test_maximum_principle():
    result = execute(Run(nx=128, ny=128, ranks=4, ic="step", t_end=0.01))
    assert result["min"] >= -1e-12
    assert result["max"] <= 1.0 + 1e-12


def test_composite_slab_matches_exact_steady_state(tmp_path):
    n, alpha_left, alpha_right = 64, 1.0, 0.1
    output = tmp_path / "out"
    execute(
        Run(
            nx=n,
            ranks=3,
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
    )
    field = read_field(output / "T_final.bin", n)
    x = (np.arange(n) + 0.5) / n
    flux = 1.0 / (0.5 / alpha_left + 0.5 / alpha_right)
    exact = np.where(x < 0.5, flux * x, 0.5 * flux + flux * (x - 0.5) / alpha_right)
    assert np.abs(field - exact).max() < 1e-9


def test_explicit_limit_is_enforced():
    with pytest.raises(SolverError, match="stability limit"):
        execute(Run(nx=64, cfl=1.5))


def test_unstable_run_is_detected_when_allowed():
    result = execute(
        Run(nx=64, ny=64, ic="gaussian", cfl=1.5, steps=4000, diag_every=10, allow_unstable=True)
    )
    assert result["status"] == "diverged"


def test_implicit_scheme_is_stable_beyond_explicit_limit():
    result = execute(Run(nx=64, ny=64, ranks=2, scheme="be", ic="step", cfl=100, steps=20))
    assert result["status"] == "ok"
    assert result["max"] <= 1.0 + 1e-8


def test_file_roundtrip_and_snapshots(tmp_path):
    rng = np.random.default_rng(0)
    initial = rng.random((40, 48))
    alpha = np.exp(rng.normal(size=(40, 48)) * 0.2)
    write_field(tmp_path / "ic.bin", initial)
    write_field(tmp_path / "alpha.bin", alpha)
    output = tmp_path / "out"
    result = execute(
        Run(
            nx=48,
            ny=40,
            ranks=4,
            ic="file",
            medium="file",
            ic_file=tmp_path / "ic.bin",
            alpha_file=tmp_path / "alpha.bin",
            alpha=float(alpha.max()),
            steps=10,
            cfl=0.5,
            output=output,
            snapshot_every=5,
            diag_every=5,
        )
    )
    start = read_field(output / "T_00000000.bin", 48, 40)
    assert np.array_equal(start, initial)
    assert (output / "T_00000005.bin").exists()
    assert result["energy_initial"] == pytest.approx(initial.mean())
    diagnostics = (output / "diag.csv").read_text().splitlines()
    assert len(diagnostics) == 4


@pytest.mark.parametrize(
    "run,message",
    [
        (Run(nx=4, ranks=8), "cells"),
        (Run(nx=64, dims=(3, 1, 1), ranks=2), "does not match"),
        (Run(nx=64, bc="periodic", halo="chain", ranks=2), "deadlocks"),
        (Run(nx=1), "more than one cell"),
    ],
)
def test_invalid_configurations_fail_cleanly(run, message):
    with pytest.raises(SolverError, match=message):
        execute(run)


def test_wrong_file_size_is_rejected(tmp_path):
    write_field(tmp_path / "ic.bin", np.zeros(10))
    with pytest.raises(SolverError, match="expected"):
        execute(Run(nx=32, ic="file", ic_file=tmp_path / "ic.bin"))
