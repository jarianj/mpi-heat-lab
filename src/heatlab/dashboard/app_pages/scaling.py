import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heatlab.dashboard.common import SERIES, line, load_json, percent, require, styled

report = require(load_json("scaling"), "python -m heatlab scale")

st.title("Parallel scaling")
st.caption(
    "Measured on one laptop: Intel i7-1360P with 4 performance and 8 efficiency cores, 16 threads. "
    "Medians of 5 repeats in randomized order. Rank placement is left to the operating system."
)

strong = report["strong"]
label = st.segmented_control("Problem", list(strong), default=list(strong)[1], key="problem")
row = strong[label or list(strong)[1]]

columns = st.columns(4)
columns[0].metric("Best speedup", f"{max(row['speedup']):.1f}x")
columns[1].metric("Efficiency at 4 ranks", percent(row["efficiency"][row["ranks"].index(4)], 0))
columns[2].metric("Efficiency at 16 ranks", percent(row["efficiency"][row["ranks"].index(16)], 0))
columns[3].metric(
    "Contention fit",
    f"{row['usl_sigma']:.3f}",
    help="Universal Scalability Law sigma. Values near 0 mean little serial contention.",
)

left, right = st.columns(2)
with left:
    figure = go.Figure()
    for slot, (name, data) in enumerate(strong.items()):
        figure.add_trace(line(data["ranks"], data["speedup"], name, slot))
    figure.add_trace(line([1, 16], [1, 16], "Ideal", 7, dash="dot", mode="lines"))
    st.plotly_chart(
        styled(figure, "Strong scaling speedup", "MPI ranks", "speedup vs 1 rank"),
        alt="Speedup against rank count for three problems",
    )
with right:
    figure = go.Figure()
    for slot, (name, data) in enumerate(strong.items()):
        figure.add_trace(line(data["ranks"], [v * 100 for v in data["efficiency"]], name, slot))
    st.plotly_chart(
        styled(figure, "Parallel efficiency", "MPI ranks", "efficiency (%)"),
        alt="Parallel efficiency against rank count for three problems",
    )

left, right = st.columns(2)
with left:
    figure = go.Figure()
    for slot, (name, data) in enumerate(report["weak"].items()):
        figure.add_trace(line(data["ranks"], [v * 100 for v in data["efficiency"]], name, slot))
    st.plotly_chart(
        styled(figure, "Weak scaling efficiency", "MPI ranks", "efficiency (%)"),
        alt="Weak scaling efficiency against rank count",
    )
with right:
    halo = pd.DataFrame(report["halo"])
    selected = halo[halo["label"].str.startswith("1D")]
    names = {"chain": "Chain (original)", "sendrecv": "Paired sendrecv", "overlap": "Non-blocking"}
    figure = go.Figure()
    for slot, (mode, name) in enumerate(names.items()):
        data = selected[selected["halo"] == mode].sort_values("ranks")
        figure.add_trace(line(data["ranks"], data["halo_max"] * 10, name, slot))
    st.plotly_chart(
        styled(figure, "Halo time per step, 4096 cells per rank", "MPI ranks", "microseconds"),
        alt="Halo time per step for three exchange strategies",
    )

decomposition = pd.DataFrame(report["decomposition"])
figure = go.Figure()
ordered = decomposition.sort_values(["ranks", "halo_max"])
labels = [f"{row.ranks} ranks, {row.label}" for row in ordered.itertuples()]
figure.add_trace(
    go.Bar(y=labels, x=ordered["compute_max"], name="Compute", orientation="h", marker_color=SERIES[0])
)
figure.add_trace(
    go.Bar(y=labels, x=ordered["halo_max"], name="Halo exchange", orientation="h", marker_color=SERIES[1])
)
figure.update_layout(barmode="stack")
figure.update_yaxes(autorange="reversed")
st.plotly_chart(
    styled(figure, "3D decomposition: 192 cubed cells, 300 steps", "seconds", height=420),
    alt="Compute and halo time for slab, pencil, and block decompositions",
)

st.caption(
    "This shared-memory machine has a fast interconnect and unequal cores. Cluster results would "
    "differ, and communication would matter more."
)
