"""Runtime prediction: an interpretable analytic model against gradient boosting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor

from .solver import RESULTS


def block_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    shape = frame[["nx", "ny", "nz"]].to_numpy(dtype=float)
    dims = np.array([[int(v) for v in text.split("x")] for text in frame["dims"]], dtype=float)
    local = shape / dims
    cells = shape.prod(axis=1)
    surface = np.zeros(len(frame))
    neighbors = np.zeros(len(frame))
    for axis in range(3):
        cross_section = np.prod(np.delete(local, axis, axis=1), axis=1)
        split = dims[:, axis] > 1
        surface += np.where(split, 2 * cross_section, 0.0)
        neighbors += np.where(split, 2.0, 0.0)
    out["cells"] = cells
    out["ranks"] = frame["ranks"].to_numpy(dtype=float)
    out["cells_per_rank"] = cells / out["ranks"]
    out["surface_per_rank"] = surface
    out["neighbors"] = neighbors
    out["dimension"] = (shape > 1).sum(axis=1).astype(float)
    out["overlap"] = (frame["halo"] == "overlap").astype(float).to_numpy()
    return out


def target_seconds_per_step(frame: pd.DataFrame) -> pd.Series:
    return frame["wall"] / frame["steps"]


class AnalyticModel:
    columns = ("cells_per_rank", "cells", "surface_per_rank", "neighbors")

    def design(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [features[c].to_numpy() for c in self.columns] + [np.ones(len(features))]
        )

    def fit(self, features: pd.DataFrame, seconds: np.ndarray) -> AnalyticModel:
        matrix = self.design(features)
        weights = 1.0 / seconds
        self.coefficients, _ = nnls(matrix * weights[:, None], np.ones_like(seconds))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.design(features) @ self.coefficients


class BoostedModel:
    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, random_state=seed
        )

    @staticmethod
    def _inputs(features: pd.DataFrame) -> np.ndarray:
        data = features.copy()
        for column in ("cells", "cells_per_rank", "surface_per_rank", "ranks"):
            data[column] = np.log(data[column].clip(lower=1.0))
        return data.to_numpy()

    def fit(self, features: pd.DataFrame, seconds: np.ndarray) -> BoostedModel:
        self.model.fit(self._inputs(features), np.log(seconds))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(self._inputs(features)))


def mape(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - truth) / truth))


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["label", "nx", "ny", "nz", "ranks", "halo", "dims", "steps"]
    return raw.groupby(keys, as_index=False).wall.median()


def validate(features, seconds, groups) -> dict:
    report = {"analytic": [], "boosted": [], "folds": []}
    for group in sorted(pd.unique(groups)):
        test = groups == group
        analytic = AnalyticModel().fit(features[~test], seconds[~test])
        boosted = BoostedModel().fit(features[~test], seconds[~test])
        report["folds"].append(str(group))
        report["analytic"].append(mape(analytic.predict(features[test]), seconds[test]))
        report["boosted"].append(mape(boosted.predict(features[test]), seconds[test]))
    return {
        "folds": report["folds"],
        "analytic_mape": report["analytic"],
        "boosted_mape": report["boosted"],
        "analytic_mean": float(np.mean(report["analytic"])),
        "boosted_mean": float(np.mean(report["boosted"])),
    }


def fit_and_validate() -> dict:
    raw = pd.read_csv(RESULTS / "sweep_raw.csv")
    frame = aggregate(raw)
    features = block_features(frame)
    seconds = target_seconds_per_step(frame).to_numpy()
    noise = raw.groupby(["label", "ranks", "halo"]).wall.agg(lambda s: s.std() / s.mean()).mean()
    report = {
        "configurations": len(frame),
        "runs": len(raw),
        "repeat_noise_cv": float(noise),
        "leave_one_grid_out": validate(features, seconds, frame["label"].to_numpy()),
        "leave_one_rank_count_out": validate(features, seconds, frame["ranks"].to_numpy()),
    }
    analytic = AnalyticModel().fit(features, seconds)
    report["analytic_coefficients"] = dict(
        zip([*AnalyticModel.columns, "constant"], analytic.coefficients.tolist(), strict=True)
    )
    (RESULTS / "perfmodel.json").write_text(json.dumps(report, indent=2))
    return report


def fit_final(raw: pd.DataFrame | None = None) -> tuple[AnalyticModel, BoostedModel]:
    raw = pd.read_csv(RESULTS / "sweep_raw.csv") if raw is None else raw
    frame = aggregate(raw)
    features = block_features(frame)
    seconds = target_seconds_per_step(frame).to_numpy()
    return AnalyticModel().fit(features, seconds), BoostedModel().fit(features, seconds)


def balanced_dims(ranks: int, dimension: int) -> tuple[int, int, int]:
    factors = []
    remaining = ranks
    divisor = 2
    while remaining > 1:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    dims = [1] * dimension
    for factor in sorted(factors, reverse=True):
        dims[int(np.argmin(dims))] *= factor
    return tuple(sorted(dims, reverse=True) + [1] * (3 - dimension))


def planning_table(dimension: int, n: int, ranks, halo: str = "overlap") -> pd.DataFrame:
    shape = [n if axis < dimension else 1 for axis in range(3)]
    return pd.DataFrame(
        {
            "nx": shape[0],
            "ny": shape[1],
            "nz": shape[2],
            "ranks": list(ranks),
            "halo": halo,
            "dims": ["x".join(str(d) for d in balanced_dims(r, dimension)) for r in ranks],
        }
    )
