"""Shared helpers for dashboard pages."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heatlab.solver import RESULTS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
HEAT = [
    [0.0, "#cde2fb"],
    [0.25, "#86b6ef"],
    [0.5, "#2a78d6"],
    [0.75, "#184f95"],
    [1.0, "#0d366b"],
]
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#b9b8b0"], [1.0, "#e34948"]]


@st.cache_data
def load_json(name: str) -> dict | list | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def require(data, command: str):
    if data is None:
        st.info(f"Generate these results with `{command}`.", icon=":material/info:")
        st.stop()
    return data


def styled(figure: go.Figure, title: str, x: str = "", y: str = "", height: int = 360) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0},
        hovermode="closest",
    )
    figure.update_xaxes(title=x, showgrid=True, zeroline=False)
    figure.update_yaxes(title=y, showgrid=True, zeroline=False)
    return figure


def line(x, y, name: str, slot: int, dash: str | None = None, mode: str = "lines+markers"):
    return go.Scatter(
        x=list(x),
        y=list(y),
        name=name,
        mode=mode,
        line={"color": SERIES[slot % len(SERIES)], "width": 2, "dash": dash},
        marker={"size": 8},
    )


def heatmap(
    field: np.ndarray,
    title: str,
    zmin: float | None = None,
    zmax: float | None = None,
    colorscale=None,
    height: int = 340,
) -> go.Figure:
    n_y, n_x = field.shape
    figure = go.Figure(
        go.Heatmap(
            z=field,
            x=(np.arange(n_x) + 0.5) / n_x,
            y=(np.arange(n_y) + 0.5) / n_y,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale or HEAT,
            colorbar={"thickness": 12},
            hovertemplate="x %{x:.3f}<br>y %{y:.3f}<br>value %{z:.4g}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0, "font": {"size": 14}},
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    figure.update_yaxes(scaleanchor="x", constrain="domain", showgrid=False)
    figure.update_xaxes(constrain="domain", showgrid=False)
    return figure


def percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"
