import numpy as np
import pytest

from heatlab.surrogate.baselines import block_mean, spectral_mean_medium
from heatlab.surrogate.dataset import IN_DISTRIBUTION, draw, render_alpha, render_initial, solve

torch = pytest.importorskip("torch")

from heatlab.surrogate.models import FNO, UNet, conserve_energy  # noqa: E402


def test_samples_are_reproducible_and_resolution_consistent():
    first, second = draw(7, IN_DISTRIBUTION), draw(7, IN_DISTRIBUTION)
    assert np.array_equal(render_alpha(first, 32), render_alpha(second, 32))
    fine = render_initial(first, 128)
    coarse = render_initial(first, 32)
    assert np.abs(block_mean(fine, 4) - coarse).max() < 0.05


def test_alpha_respects_distribution_bounds():
    alpha = render_alpha(draw(3, IN_DISTRIBUTION), 64)
    assert alpha.min() >= IN_DISTRIBUTION.alpha_min * 0.999
    assert alpha.max() <= IN_DISTRIBUTION.alpha_max * 1.001


def test_spectral_baseline_matches_solver_for_uniform_medium():
    sample = draw(11, IN_DISTRIBUTION)
    initial = render_initial(sample, 64)
    alpha = np.full_like(initial, 0.5)
    truth = solve(initial, alpha, tau=0.005, cfl=0.05)["field"]
    spectral = spectral_mean_medium(initial, alpha, tau=0.005)
    assert np.linalg.norm(spectral - truth) / np.linalg.norm(truth) < 1e-3


@pytest.mark.parametrize("model", [FNO(width=8, modes=4, layers=2), UNet(base=8, levels=2)])
@pytest.mark.parametrize("size", [32, 64])
def test_models_handle_multiple_resolutions(model, size):
    initial = torch.rand(2, size, size)
    alpha = torch.rand(2, size, size) + 0.1
    assert model(initial, alpha).shape == (2, size, size)


def test_conservative_projection_removes_net_energy():
    change = conserve_energy(torch.rand(3, 16, 16))
    assert change.mean(dim=(-2, -1)).abs().max() < 1e-6
