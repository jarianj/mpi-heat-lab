import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heatlab.dashboard.common import line, load_csv, load_json, percent, require, styled
from heatlab.perfmodel import block_features, fit_final, planning_table

report = require(load_json("perfmodel"), "python -m heatlab sweep && python -m heatlab perf")
raw = require(load_csv("sweep_raw"), "python -m heatlab sweep")


@st.cache_resource
def models():
    return fit_final(raw)


st.title("Runtime planner")
st.caption(
    f"Predicts time per step from {report['runs']} timed runs on this laptop. "
    "The result only applies to this machine."
)

grid = report["leave_one_grid_out"]
ranks_fold = report["leave_one_rank_count_out"]
columns = st.columns(4)
columns[0].metric("Analytic model, unseen grid", percent(grid["analytic_mean"], 0))
columns[1].metric("Boosted trees, unseen grid", percent(grid["boosted_mean"], 0))
columns[2].metric("Analytic model, unseen ranks", percent(ranks_fold["analytic_mean"], 0))
columns[3].metric("Boosted trees, unseen ranks", percent(ranks_fold["boosted_mean"], 0))
st.caption(f"Mean absolute percentage error. Repeat-to-repeat noise is {percent(report['repeat_noise_cv'], 0)}.")

controls = st.columns(3)
dimension = controls[0].segmented_control("Dimensions", [1, 2, 3], default=2, key="plan_dimension") or 2
limits = {1: (2**16, 2**22), 2: (256, 2048), 3: (48, 192)}
low, high = limits[dimension]
n = controls[1].slider("Cells per axis", low, high, int(np.sqrt(low * high)), key=f"plan_n_{dimension}")
steps = controls[2].number_input("Steps", 1, 10_000_000, 10_000, key="plan_steps")

available = os.cpu_count() or 16
ranks = [r for r in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16) if r <= max(available, 16)]
frame = planning_table(dimension, n, ranks)
features = block_features(frame)
analytic, boosted = models()
frame["analytic"] = analytic.predict(features) * steps
frame["boosted"] = boosted.predict(features) * steps

best = frame.loc[frame["boosted"].idxmin()]
columns = st.columns(3)
columns[0].metric("Recommended ranks", int(best["ranks"]))
columns[1].metric("Predicted time", f"{best['boosted']:.2f} s")
columns[2].metric("Decomposition", best["dims"])

figure = go.Figure(line(frame["ranks"], frame["analytic"], "Analytic model", 0))
figure.add_trace(line(frame["ranks"], frame["boosted"], "Boosted trees", 1))
st.plotly_chart(
    styled(figure, "Predicted runtime", "MPI ranks", "seconds"),
    alt="Predicted runtime against rank count for both models",
)

with st.expander("Analytic model coefficients"):
    coefficients = report["analytic_coefficients"]
    st.dataframe(
        pd.DataFrame(
            {
                "Term": list(coefficients),
                "Seconds per unit": list(coefficients.values()),
            }
        ),
        hide_index=True,
        column_config={"Seconds per unit": st.column_config.NumberColumn(format="%.3e")},
        alt="Fitted coefficients of the analytic model",
    )
