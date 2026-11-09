import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heatlab.dashboard.common import line, load_json, require, styled

report = require(load_json("verification"), "python -m heatlab verify")

st.title("Verification")
st.caption("Evidence that the solver computes the right answer, measured against exact results.")

spatial = report["spatial_convergence"]
temporal = report["temporal_convergence"]
independent = report["decomposition_independence"]
energy = report["energy_conservation"]

identical = sum(row["bit_identical"] for row in independent)
comparisons = sum(row["comparisons"] for row in independent)
orders = [row["order"] for row in spatial]

columns = st.columns(4)
columns[0].metric("Spatial order", f"{min(orders):.2f} to {max(orders):.2f}", help="Design order is 2")
columns[1].metric("Bit-identical parallel runs", f"{identical} of {comparisons}")
columns[2].metric("Worst energy drift", f"{max(row['relative_drift'] for row in energy):.1e}")
columns[3].metric("Composite slab error", f"{report['composite_slab']['linf_error']:.1e}")

left, right = st.columns(2)
with left:
    boundary = st.segmented_control(
        "Boundary", ["neumann", "dirichlet", "periodic"], default="neumann", key="boundary_choice"
    )
    figure = go.Figure()
    for slot, dimension in enumerate(("1D", "2D", "3D")):
        for row in spatial:
            if row["case"] == f"{dimension} {boundary or 'neumann'}":
                figure.add_trace(
                    line(row["n"], row["l2_error"], f"{dimension}: order {row['order']:.2f}", slot)
                )
    figure.update_xaxes(type="log")
    figure.update_yaxes(type="log")
    st.plotly_chart(
        styled(figure, "Spatial convergence", "cells per axis", "L2 error"),
        alt="Log-log plot of error against grid size",
    )
with right:
    figure = go.Figure()
    names = {"cn": "Crank-Nicolson", "be": "Backward Euler"}
    for slot, row in enumerate(temporal):
        label = f"{names[row['scheme']]}: order {row['order']:.2f}"
        figure.add_trace(line(row["dt"], row["l2_error"], label, slot))
    figure.update_xaxes(type="log")
    figure.update_yaxes(type="log")
    st.plotly_chart(
        styled(figure, "Temporal convergence", "time step", "L2 error"),
        alt="Log-log plot of error against time step",
    )

st.subheader("Same answer on any decomposition")
st.caption("SHA-256 of the output compared with the 1-rank run, for 2 to 12 ranks and two halo modes.")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Case": row["case"],
                "Bit-identical": row["bit_identical"],
                "Comparisons": row["comparisons"],
            }
            for row in independent
        ]
    ),
    hide_index=True,
    alt="Decomposition independence results",
)

left, right = st.columns(2)
with left:
    st.subheader("Energy conservation")
    st.dataframe(
        pd.DataFrame(energy)[["case", "ranks", "relative_drift"]].rename(
            columns={"case": "Case", "ranks": "Ranks", "relative_drift": "Relative drift"}
        ),
        hide_index=True,
        column_config={"Relative drift": st.column_config.NumberColumn(format="%.2e")},
        alt="Relative energy drift by case and rank count",
    )
with right:
    st.subheader("Stability boundary")
    st.dataframe(
        pd.DataFrame(report["stability_boundary"]).rename(
            columns={
                "scheme": "Scheme",
                "cfl": "dt / explicit limit",
                "status": "Status",
                "completed_steps": "Steps",
                "max_abs": "Largest value",
            }
        ),
        hide_index=True,
        column_config={"Largest value": st.column_config.NumberColumn(format="%.2e")},
        alt="Outcome of runs around the explicit stability limit",
    )
