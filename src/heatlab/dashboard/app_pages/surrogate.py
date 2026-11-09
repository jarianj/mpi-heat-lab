import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from heatlab.dashboard.common import DIVERGING, heatmap, line, load_json, percent, require, styled
from heatlab.solver import MODELS
from heatlab.surrogate.dataset import (
    IN_DISTRIBUTION,
    OUT_OF_DISTRIBUTION,
    draw,
    render_alpha,
    render_initial,
    solve,
)
from heatlab.surrogate.models import conserve_energy
from heatlab.surrogate.training import load_model

report = require(load_json("surrogate"), "python -m heatlab evaluate")


@st.cache_resource
def fno_model():
    if not (MODELS / "fno_seed0.pt").exists():
        return None
    model = load_model("fno", 0, torch.device("cpu"))
    with torch.no_grad():
        model(torch.zeros(1, 64, 64), torch.ones(1, 64, 64))
    return model


@st.cache_data(max_entries=32, show_spinner=False)
def compare(seed: int, shifted: bool, n: int) -> dict:
    distribution = OUT_OF_DISTRIBUTION if shifted else IN_DISTRIBUTION
    sample = draw(seed, distribution)
    initial = render_initial(sample, n)
    alpha = render_alpha(sample, n)
    solved = solve(initial, alpha, ranks=4)
    initial_t = torch.tensor(initial[None], dtype=torch.float32)
    alpha_t = torch.tensor(alpha[None], dtype=torch.float32)
    start = time.perf_counter()
    with torch.no_grad():
        change = conserve_energy(fno_model()(initial_t, alpha_t))
    seconds = time.perf_counter() - start
    prediction = (initial_t + change)[0].numpy()
    return {
        "initial": initial,
        "alpha": alpha,
        "truth": solved["field"],
        "prediction": prediction,
        "solver_seconds": solved["wall"],
        "fno_seconds": seconds,
    }


st.title("Neural surrogate")
st.caption(
    "A Fourier Neural Operator trained on data from the MPI solver. It predicts the temperature at "
    "t = 0.02 from the initial temperature and material map, with insulated boundaries."
)

test = report["accuracy"]["test"]
shifted_table = report["accuracy"]["ood"]

columns = st.columns(4)
columns[0].metric(
    "FNO error, test",
    percent(test["fno conservative"]["change_relative_l2"]),
    help="Relative L2 error of the predicted temperature change",
)
columns[1].metric("32x32 coarse solver", percent(test["coarse solver 32x32"]["change_relative_l2"]))
columns[2].metric(
    "FNO error, shifted data", percent(shifted_table["fno conservative"]["change_relative_l2"])
)
columns[3].metric("FNO energy drift", f"{test['fno conservative']['energy_drift']:.1e}")

st.subheader("Try a new sample")
controls = st.columns([2, 2, 2], vertical_alignment="bottom")
seed = controls[0].number_input("Sample seed", 0, 10_000_000, 900_001, key="sample_seed")
distribution_choice = controls[1].segmented_control(
    "Data", ["Training-like", "Shifted"], default="Training-like", key="distribution_choice"
)
resolution = controls[2].segmented_control(
    "Grid", [64, 128, 256], default=64, key="resolution_choice"
)

if fno_model() is None:
    st.info(
        "Train a model with `python -m heatlab train` to enable this panel.", icon=":material/info:"
    )
else:
    with st.spinner("Running the solver and the model"):
        output = compare(int(seed), distribution_choice == "Shifted", resolution or 64)
    error = output["prediction"] - output["truth"]
    change = output["truth"] - output["initial"]
    limit = float(np.abs(error).max())
    low, high = float(output["initial"].min()), float(output["initial"].max())

    columns = st.columns(3)
    columns[0].metric(
        "Error on this sample", percent(np.linalg.norm(error) / np.linalg.norm(change))
    )
    columns[1].metric("Solver, 4 ranks, CFL 0.25", f"{output['solver_seconds'] * 1000:.1f} ms")
    columns[2].metric("FNO, CPU", f"{output['fno_seconds'] * 1000:.1f} ms")

    panels = st.columns(5)
    panels[0].plotly_chart(
        heatmap(output["initial"], "Initial", low, high, height=250), alt="Initial temperature"
    )
    panels[1].plotly_chart(
        heatmap(np.log10(output["alpha"]), "log10 diffusivity", height=250), alt="Material map"
    )
    panels[2].plotly_chart(heatmap(output["truth"], "Solver", height=250), alt="Solver temperature")
    panels[3].plotly_chart(heatmap(output["prediction"], "FNO", height=250), alt="FNO temperature")
    panels[4].plotly_chart(
        heatmap(error, "FNO - solver", -limit, limit, DIVERGING, height=250), alt="Prediction error"
    )

st.subheader("How it compares")
names = {
    "persistence": "Persistence (no change)",
    "spectral mean medium": "Exact solve, averaged material",
    "POD-ridge": "POD-ridge regression",
    "coarse solver 16x16": "Coarse solver 16x16",
    "coarse solver 32x32": "Coarse solver 32x32",
    "unet conservative": "U-Net",
    "fno conservative": "FNO",
}
rows = []
for key, name in names.items():
    rows.append(
        {
            "Method": name,
            "Test error": test[key]["change_relative_l2"],
            "Shifted error": shifted_table[key]["change_relative_l2"],
            "Energy drift": test[key]["energy_drift"],
            "Max-principle excess": test[key]["max_principle_violation"],
        }
    )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    column_config={
        "Test error": st.column_config.NumberColumn(format="percent"),
        "Shifted error": st.column_config.NumberColumn(format="percent"),
        "Energy drift": st.column_config.NumberColumn(format="%.1e"),
        "Max-principle excess": st.column_config.NumberColumn(format="%.2e"),
    },
    alt="Accuracy of each method on test and shifted data",
)

timing = pd.DataFrame(report["timing"]).groupby("n").median(numeric_only=True).reset_index()
accuracy = pd.DataFrame(report["cost_accuracy"])

left, right = st.columns(2)
with left:
    figure = go.Figure()
    series = {
        "solver_1_rank": "Solver, 1 rank",
        "solver_8_ranks": "Solver, 8 ranks",
        "fno_cpu": "FNO, CPU",
        "fno_gpu": "FNO, GPU",
    }
    for slot, (key, name) in enumerate(series.items()):
        if key in timing:
            figure.add_trace(line(timing["n"], timing[key], name, slot))
    figure.update_xaxes(type="log", tickvals=timing["n"].tolist())
    figure.update_yaxes(type="log")
    st.plotly_chart(
        styled(figure, "Time per sample", "cells per axis", "seconds"),
        alt="Inference and solver time against grid size",
    )
with right:
    target = st.segmented_control("Target grid", [64, 128, 256], default=256, key="target_grid")
    subset = accuracy[accuracy["n"] == (target or 256)]
    coarse = subset[subset["method"] == "coarse solver"].sort_values("seconds")
    fno_row = subset[subset["method"] == "fno conservative"].iloc[0]
    fno_time = timing.loc[timing["n"] == (target or 256), "fno_cpu"].iloc[0]
    figure = go.Figure(
        line(coarse["seconds"], coarse["change_relative_l2"] * 100, "Coarse solver", 0)
    )
    figure.add_trace(
        line([fno_time], [fno_row["change_relative_l2"] * 100], "FNO, CPU", 1, mode="markers")
    )
    figure.update_xaxes(type="log")
    figure.update_yaxes(type="log")
    st.plotly_chart(
        styled(figure, "Cost against accuracy", "seconds per sample", "error in change (%)"),
        alt="Error against cost for coarse solver and FNO",
    )
