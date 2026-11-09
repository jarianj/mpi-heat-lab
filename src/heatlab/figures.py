"""Static report figures built from the committed result files."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .solver import RESULTS, ROOT, Run, execute, read_field

FIGURES = ROOT / "reports" / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("error", ["#2a78d6", "#f0efec", "#e34948"])


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def load(name: str):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png")
    plt.close(figure)


def convergence(report: dict) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    for color, dimension in zip(SERIES, ("1D", "2D", "3D"), strict=False):
        for row in report["spatial_convergence"]:
            if row["case"] == f"{dimension} neumann":
                left.loglog(
                    row["n"],
                    row["l2_error"],
                    "o-",
                    color=color,
                    label=f"{dimension}, order {row['order']:.2f}",
                )
    reference = np.array([32, 512], dtype=float)
    left.loglog(
        reference, 2e-3 * (32 / reference) ** 2, "--", color=MUTED, linewidth=1, label="slope 2"
    )
    left.set(
        title="Spatial error falls with grid size squared",
        xlabel="cells per axis",
        ylabel="L2 error against exact solution",
    )
    left.legend()
    names = {"cn": "Crank-Nicolson", "be": "backward Euler"}
    for color, row in zip(SERIES, report["temporal_convergence"], strict=False):
        right.loglog(
            row["dt"],
            row["l2_error"],
            "o-",
            color=color,
            label=f"{names[row['scheme']]}, order {row['order']:.2f}",
        )
    right.set(
        title="Implicit schemes reach their design order",
        xlabel="time step",
        ylabel="L2 error against exact solution",
    )
    right.legend()
    figure.tight_layout()
    save(figure, "convergence")


def scaling(report: dict) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    highest = 1
    for color, (label, row) in zip(SERIES, report["strong"].items(), strict=False):
        ranks = np.array(row["ranks"])
        highest = max(highest, ranks.max())
        left.plot(ranks, row["speedup"], "o-", color=color, label=label)
        right.plot(ranks, np.array(row["efficiency"]) * 100, "o-", color=color, label=label)
    left.plot([1, highest], [1, highest], "--", color=MUTED, linewidth=1, label="ideal")
    left.set(title="Strong scaling speedup", xlabel="MPI ranks", ylabel="speedup vs 1 rank")
    left.legend()
    right.set(
        title="Parallel efficiency", xlabel="MPI ranks", ylabel="efficiency (%)", ylim=(0, 115)
    )
    right.axvline(8, color=MUTED, linewidth=1, linestyle=":")
    right.annotate("8 performance threads", (8, 108), xytext=(8.3, 108), color=MUTED, fontsize=8)
    figure.tight_layout()
    save(figure, "strong_scaling")


def halo_comparison(report: dict) -> None:
    rows = [r for r in report["halo"] if r["label"].startswith("1D")]
    figure, axis = plt.subplots(figsize=(6, 3.6))
    names = {"chain": "chain (original)", "sendrecv": "paired sendrecv", "overlap": "non-blocking"}
    for color, halo in zip(SERIES, names, strict=False):
        selected = sorted((r for r in rows if r["halo"] == halo), key=lambda r: r["ranks"])
        steps = 100000
        axis.plot(
            [r["ranks"] for r in selected],
            [r["halo_max"] / steps * 1e6 for r in selected],
            "o-",
            color=color,
            label=names[halo],
        )
    axis.set(
        title="Halo time per step, 4096 cells per rank",
        xlabel="MPI ranks",
        ylabel="halo time per step (microseconds)",
    )
    axis.legend()
    figure.tight_layout()
    save(figure, "halo_comparison")


def surrogate_accuracy(report: dict) -> None:
    table = report["accuracy"]["test"]
    order = [
        "persistence",
        "spectral mean medium",
        "POD-ridge",
        "coarse solver 16x16",
        "coarse solver 32x32",
        "unet conservative",
        "fno conservative",
    ]
    labels = [
        "Persistence",
        "Spectral, mean medium",
        "POD-ridge",
        "Coarse solver 16x16",
        "Coarse solver 32x32",
        "U-Net",
        "FNO",
    ]
    values = [table[name]["change_relative_l2"] * 100 for name in order]
    figure, axis = plt.subplots(figsize=(7, 3.4))
    colors = ["#c3c2b7"] * 5 + [SERIES[1], SERIES[0]]
    bars = axis.barh(labels, values, color=colors, height=0.6)
    axis.bar_label(bars, fmt="%.1f%%", padding=3, color=INK, fontsize=9)
    axis.invert_yaxis()
    axis.set(
        title="Error in predicted temperature change, held-out test set",
        xlabel="relative L2 error of the change (%)",
    )
    axis.grid(axis="y", visible=False)
    axis.set_xlim(0, max(values) * 1.15)
    figure.tight_layout()
    save(figure, "surrogate_accuracy")


def cost_curve(report: dict) -> None:
    rows = report["timing"]
    sizes = sorted({row["n"] for row in rows})

    def median(key: str, n: int) -> float:
        return float(np.median([row[key] for row in rows if row["n"] == n]))

    figure, axis = plt.subplots(figsize=(6, 3.8))
    series = [
        ("solver_1_rank", "Solver, 1 rank"),
        ("solver_8_ranks", "Solver, 8 ranks"),
        ("fno_cpu", "FNO, CPU"),
        ("fno_gpu", "FNO, GPU"),
    ]
    for color, (key, label) in zip(SERIES, series, strict=False):
        if key in rows[0]:
            axis.loglog(sizes, [median(key, n) for n in sizes], "o-", color=color, label=label)
    axis.set_xticks(sizes)
    axis.set_xticklabels(sizes)
    axis.minorticks_off()
    axis.set(
        title="Time to advance one sample to the target time",
        xlabel="cells per axis",
        ylabel="seconds",
    )
    axis.legend()
    figure.tight_layout()
    save(figure, "cost_curve")


def surrogate_example() -> None:
    import torch

    from .surrogate.dataset import load_split
    from .surrogate.models import conserve_energy
    from .surrogate.training import device, load_model

    split = load_split("test")
    target_device = device()
    model = load_model("fno", 0, target_device)
    index = 0
    initial = torch.from_numpy(split["initial"][index : index + 1]).to(target_device)
    alpha = torch.from_numpy(split["alpha"][index : index + 1]).to(target_device)
    with torch.no_grad():
        prediction = (initial + conserve_energy(model(initial, alpha))).cpu().numpy()[0]
    truth = split["target"][index]
    panels = [
        (split["initial"][index], "Initial temperature", HEAT, None),
        (np.log10(split["alpha"][index]), "log10 diffusivity", HEAT, None),
        (truth, "MPI solver", HEAT, None),
        (prediction, "FNO prediction", HEAT, None),
        (prediction - truth, "Prediction error", DIVERGING, np.abs(prediction - truth).max()),
    ]
    figure, axes = plt.subplots(1, 5, figsize=(13, 2.8))
    for axis, (field, title, cmap, limit) in zip(axes, panels, strict=True):
        arguments = {"vmin": -limit, "vmax": limit} if limit else {}
        image = axis.imshow(field, origin="lower", cmap=cmap, **arguments)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    save(figure, "surrogate_example")


def diffusion_snapshots() -> None:
    import tempfile
    from pathlib import Path

    n = 256
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out"
        run = Run(
            nx=n,
            ny=n,
            ranks=8,
            ic="step",
            medium="inclusion",
            alpha=1.0,
            alpha2=0.02,
            t_end=0.03,
            cfl=0.9,
            output=output,
            snapshot_every=500,
        )
        result = execute(run)
        names = sorted(output.glob("T_0*.bin"))
        chosen = [names[0], names[len(names) // 3], names[2 * len(names) // 3], names[-1]]
        fields = [read_field(path, n, n) for path in chosen]
        times = [int(path.stem.split("_")[1]) * result["dt"] for path in chosen]
    figure, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for axis, field, time in zip(axes, fields, times, strict=True):
        image = axis.imshow(field, origin="lower", cmap=HEAT, vmin=0, vmax=1)
        axis.set_title(f"t = {time:.3f}", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
    figure.colorbar(image, ax=axes, fraction=0.012, pad=0.01, label="temperature")
    figure.suptitle(
        "Heat crossing a low-diffusivity inclusion, 256x256 on 8 ranks",
        x=0.05,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    save(figure, "diffusion_snapshots")


def render_all() -> None:
    style()
    diffusion_snapshots()
    verification = load("verification")
    if verification:
        convergence(verification)
    bench = load("scaling")
    if bench:
        scaling(bench)
        halo_comparison(bench)
    surrogate = load("surrogate")
    if surrogate:
        surrogate_accuracy(surrogate)
        cost_curve(surrogate)
        surrogate_example()
