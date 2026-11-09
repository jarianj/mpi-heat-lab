import pytest

from heatlab.solver import SolverError, binary, launcher


@pytest.fixture(scope="session", autouse=True)
def solver_available():
    try:
        binary()
        launcher()
    except SolverError as error:
        pytest.skip(str(error))
