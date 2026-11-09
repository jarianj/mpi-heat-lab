import math
import os
import shlex
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heatlab.dashboard.common import heatmap, line, percent, styled
from heatlab.solver import Run, SolverError, execute, read_field

MAX_FRAMES = 24


def count_steps(shape: tuple[int, int, int], alpha_max: float, t_end: float, cfl: float) -> int:
    inverse = sum(n * n for n in shape if n > 1)
    return max(1, math.ceil(t_end / (cfl / (2 * alpha_max * inverse))))


@st.cache_data(max_entries=20, show_spinner=False)
def simulate(run: Run, steps: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out"
        every = max(1, steps // MAX_FRAMES)
        configured = replace(
            run,
            output=output,
            snapshot_every=every,
            diag_every=max(1, steps // 200),
            allow_unstable=True,
        )
        result = execute(configured, timeout=180)
        frames = []
        for path in sorted(output.glob("T_0*.bin")):
            field = read_field(path, run.nx, run.ny, run.nz)
            if field.ndim == 3:
                field = field[field.shape[0] // 2]
            frames.append((int(path.stem.split("_")[1]) * result["dt"], field))
        diagnostics = pd.read_csv(output / "diag.csv")
    return {"result": result, "frames": frames, "diagnostics": diagnostics}


st.title("Live solver")
st.caption("Each run launches the compiled C solver through mpiexec.")

with st.sidebar.form("configuration"):
    dimension = st.segmented_control("Dimensions", [1, 2, 3], default=2, width="stretch")
    sizes = {1: [256, 1024, 4096, 16384], 2: [64, 128, 256, 512], 3: [32, 64, 96, 128]}
    size_index = st.select_slider("Grid size", ["small", "medium", "large", "largest"], "medium")
    ranks = st.slider("MPI ranks", 1, os.cpu_count() or 4, min(4, os.cpu_count() or 4))
    scheme_label = st.selectbox("Scheme", ["Explicit", "Crank-Nicolson", "Backward Euler"])
    bc_label = st.selectbox("Boundary", ["Insulated", "Fixed temperature", "Periodic"])
    halo_label = st.selectbox("Halo exchange", ["Non-blocking", "Paired sendrecv", "Chain"])
    ic_label = st.selectbox("Initial temperature", ["Gaussian", "Step", "Uniform"])
    medium_label = st.selectbox("Material", ["Uniform", "Two layers", "Inclusion"])
    alpha2 = st.slider("Second material diffusivity", 0.01, 2.0, 0.05, 0.01)
    t_end = st.slider("End time", 0.001, 0.05, 0.01, 0.001, format="%.3f")
    cfl = st.slider("Time step relative to explicit limit", 0.1, 20.0, 0.9, 0.01)
    submitted = st.form_submit_button("Run solver", type="primary", width="stretch")

dimension = dimension or 2
n = sizes[dimension][["small", "medium", "large", "largest"].index(size_index)]
shape = tuple(n if axis < dimension else 1 for axis in range(3))
scheme = {"Explicit": "explicit", "Crank-Nicolson": "cn", "Backward Euler": "be"}[scheme_label]
bc = {"Insulated": "neumann", "Fixed temperature": "dirichlet", "Periodic": "periodic"}[bc_label]
halo = {"Non-blocking": "overlap", "Paired sendrecv": "sendrecv", "Chain": "chain"}[halo_label]
ic = ic_label.lower()
medium = {"Uniform": "uniform", "Two layers": "layers", "Inclusion": "inclusion"}[medium_label]
if medium == "inclusion" and dimension == 1:
    medium = "layers"

steps = count_steps(shape, max(1.0, alpha2 if medium != "uniform" else 1.0), t_end, cfl)
run = Run(
    nx=shape[0],
    ny=shape[1],
    nz=shape[2],
    ranks=ranks,
    scheme=scheme,
    halo=halo,
    bc=bc,
    ic=ic,
    medium=medium,
    alpha2=alpha2,
    t_end=t_end,
    cfl=cfl,
)

if scheme == "explicit" and cfl > 1:
    st.warning(
        "This time step is above the explicit stability limit, so the run should blow up.",
        icon=":material/warning:",
    )

if submitted or "simulation" not in st.session_state:
    if steps > 400_000:
        st.error(
            "That combination needs too many steps. Shorten the end time or use a coarser grid."
        )
        st.stop()
    try:
        with st.spinner("Running MPI solver"):
            st.session_state.simulation = simulate(run, steps)
            st.session_state.simulated_run = run
    except SolverError as error:
        st.error(str(error), icon=":material/error:")
        st.stop()

simulation = st.session_state.simulation
result = simulation["result"]
frames = simulation["frames"]
diagnostics = simulation["diagnostics"]

if result["status"] == "diverged":
    st.error(
        f"Diverged after {result['completed_steps']:,} steps: the temperature grew past 1e30.",
        icon=":material/error:",
    )

columns = st.columns(5)
columns[0].metric("Steps", f"{result['completed_steps']:,}")
columns[1].metric("Wall time", f"{result['wall']:.3f} s")
columns[2].metric(
    "Throughput", f"{result['mcups']:,.0f} MCUPS", help="Million cell updates per second"
)
columns[3].metric("Halo share", percent(result["halo_max"] / max(result["wall"], 1e-12)))
drift = abs(result["energy_final"] - result["energy_initial"]) / max(
    abs(result["energy_initial"]), 1e-300
)
columns[4].metric("Energy change", f"{drift:.1e}")

left, right = st.columns([3, 2])
with left:
    if frames:
        index = st.slider("Snapshot", 0, len(frames) - 1, len(frames) - 1)
        time_value, field = frames[index]
        low = min(float(f.min()) for _, f in frames)
        high = max(float(f.max()) for _, f in frames)
        if field.ndim == 1:
            x = (np.arange(field.size) + 0.5) / field.size
            figure = go.Figure(line(x, field, "Temperature", 0, mode="lines"))
            figure.update_yaxes(range=[low, high])
            st.plotly_chart(
                styled(figure, f"Temperature at t = {time_value:.4f}", "x", "temperature"),
                alt="Temperature profile along x",
            )
        else:
            suffix = ", middle slice" if dimension == 3 else ""
            st.plotly_chart(
                heatmap(
                    field, f"Temperature at t = {time_value:.4f}{suffix}", low, high, height=480
                ),
                alt="Temperature field heatmap",
            )
    else:
        st.info("The run diverged before a usable snapshot was written.", icon=":material/info:")

with right:
    peak = go.Figure(line(diagnostics["time"], diagnostics["max"], "Maximum", 0, mode="lines"))
    peak.add_trace(line(diagnostics["time"], diagnostics["min"], "Minimum", 1, mode="lines"))
    st.plotly_chart(
        styled(peak, "Temperature range", "time", "temperature", height=250),
        alt="Minimum and maximum temperature over time",
    )
    stages = ["Compute", "Halo exchange", "Reductions"]
    values = [result["compute_max"], result["halo_max"], result["reduce_max"]]
    bars = go.Figure(go.Bar(x=values, y=stages, orientation="h", marker_color="#2a78d6"))
    bars.update_yaxes(autorange="reversed")
    st.plotly_chart(
        styled(bars, "Slowest-rank time by stage", "seconds", height=220),
        alt="Time spent in compute, halo exchange, and reductions",
    )

with st.expander("Command and raw result"):
    simulated = st.session_state.simulated_run
    st.code(
        f"mpiexec -n {simulated.ranks} bin/heatsolve " + shlex.join(simulated.arguments()),
        language="bash",
    )
    st.json(result, expanded=False)
