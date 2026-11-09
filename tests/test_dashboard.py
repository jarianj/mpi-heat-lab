from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("torch")

from streamlit.testing.v1 import AppTest  # noqa: E402

import heatlab.dashboard  # noqa: E402

PAGES = Path(heatlab.dashboard.__file__).parent / "app_pages"


@pytest.mark.parametrize(
    "page", ["solver.py", "verification.py", "scaling.py", "surrogate.py", "planner.py"]
)
def test_page_runs_without_exceptions(page):
    app = AppTest.from_file(str(PAGES / page), default_timeout=120).run()
    assert not app.exception


def test_live_solver_reports_divergence():
    app = AppTest.from_file(str(PAGES / "solver.py"), default_timeout=120).run()
    app.sidebar.slider[3].set_value(1.5)
    app.sidebar.select_slider[0].set_value("small")
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert any("Diverged" in element.value for element in app.error) or app.warning
