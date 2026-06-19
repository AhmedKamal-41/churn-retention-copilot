"""Headless tests for the Streamlit dashboard using Streamlit's AppTest.

These run each page as a script (no browser) and check it renders without error,
including the case where no customer has been analysed yet.
"""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.components.analysis import analyze_customer
from app.components.input_form import PRESETS
from app.components.loaders import build_reference
from src.data.split import FEATURE_COLUMNS

HOME = "app/streamlit_app.py"
ANALYZE = "app/pages/1_Customer_Assessment.py"
MAP = "app/pages/2_Vector_Map.py"
PLAN = "app/pages/3_Retention_Actions.py"
METHODOLOGY = "app/pages/4_Model_Reference.py"

TIMEOUT = 180


@pytest.fixture(scope="module")
def sample_analysis():
    reference = build_reference()
    customer = pd.DataFrame(
        [{column: PRESETS["High risk"][f"f_{column}"] for column in FEATURE_COLUMNS}]
    )[FEATURE_COLUMNS]
    return analyze_customer(reference, customer)


def test_home_page_runs():
    app = AppTest.from_file(HOME, default_timeout=TIMEOUT).run()
    assert not app.exception
    page_text = " ".join(m.value for m in app.markdown)
    assert "Customer Churn" in page_text


def test_methodology_page_runs():
    app = AppTest.from_file(METHODOLOGY, default_timeout=TIMEOUT).run()
    assert not app.exception


def test_analyze_page_produces_results():
    app = AppTest.from_file(ANALYZE, default_timeout=TIMEOUT).run()
    assert not app.exception

    analyze_button = next(b for b in app.button if b.label == "Analyze Customer")
    analyze_button.click().run()

    assert not app.exception
    assert "analysis" in app.session_state
    page_text = " ".join(m.value for m in app.markdown)
    assert "Churn probability" in page_text


def test_map_page_without_analysis_shows_prompt():
    app = AppTest.from_file(MAP, default_timeout=TIMEOUT).run()
    assert not app.exception
    assert len(app.info) >= 1


def test_plan_page_without_analysis_shows_prompt():
    app = AppTest.from_file(PLAN, default_timeout=TIMEOUT).run()
    assert not app.exception
    assert len(app.info) >= 1


def test_map_page_renders_with_analysis(sample_analysis):
    app = AppTest.from_file(MAP, default_timeout=TIMEOUT)
    app.session_state["analysis"] = sample_analysis
    app.run()
    assert not app.exception


def test_plan_page_renders_with_analysis(sample_analysis):
    app = AppTest.from_file(PLAN, default_timeout=TIMEOUT)
    app.session_state["analysis"] = sample_analysis
    app.run()
    assert not app.exception
