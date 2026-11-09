"""Non-neural competitors a surrogate has to beat."""

from __future__ import annotations

import time

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .dataset import TAU, solve


def persistence(initial: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return initial.copy()


def spectral_mean_medium(initial: np.ndarray, alpha: np.ndarray, tau: float = TAU) -> np.ndarray:
    n = initial.shape[-1]
    effective = np.exp(np.log(alpha).mean(axis=(-2, -1)))
    wave = 2.0 * n * np.sin(np.pi * np.arange(n) / (2 * n))
    eigenvalues = wave[:, None] ** 2 + wave[None, :] ** 2
    spectrum = dctn(initial, axes=(-2, -1), type=2, norm="ortho")
    if initial.ndim == 3:
        decay = np.exp(-effective[:, None, None] * eigenvalues * tau)
    else:
        decay = np.exp(-effective * eigenvalues * tau)
    return idctn(spectrum * decay, axes=(-2, -1), type=2, norm="ortho")


class PodRidge:
    def __init__(self, components: int = 128, alpha: float = 1e-3):
        self.components = components
        self.ridge = Ridge(alpha=alpha)

    @staticmethod
    def _inputs(initial: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        flat_initial = initial.reshape(len(initial), -1)
        flat_alpha = np.log(alpha).reshape(len(alpha), -1)
        return np.concatenate([flat_initial, flat_alpha], axis=1)

    def fit(self, initial: np.ndarray, alpha: np.ndarray, target: np.ndarray) -> PodRidge:
        rank = min(self.components, len(initial) - 1)
        self.input_pca = PCA(n_components=rank, random_state=0)
        self.output_pca = PCA(n_components=rank, random_state=0)
        codes = self.input_pca.fit_transform(self._inputs(initial, alpha))
        outputs = self.output_pca.fit_transform((target - initial).reshape(len(target), -1))
        self.ridge.fit(codes, outputs)
        return self

    def predict(self, initial: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        codes = self.input_pca.transform(self._inputs(initial, alpha))
        change = self.output_pca.inverse_transform(self.ridge.predict(codes))
        return initial + change.reshape(initial.shape)


def block_mean(field: np.ndarray, factor: int) -> np.ndarray:
    n = field.shape[0] // factor
    return field.reshape(n, factor, n, factor).mean(axis=(1, 3))


def block_geometric_mean(field: np.ndarray, factor: int) -> np.ndarray:
    return np.exp(block_mean(np.log(field), factor))


def coarse_solver(initial: np.ndarray, alpha: np.ndarray, factor: int) -> tuple[np.ndarray, float]:
    result = solve(block_mean(initial, factor), block_geometric_mean(alpha, factor), cfl=0.9)
    start = time.perf_counter()
    fine = zoom(result["field"], factor, order=3, mode="nearest", grid_mode=True)
    return fine, result["wall"] + time.perf_counter() - start
