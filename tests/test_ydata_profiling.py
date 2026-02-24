"""
test_ydata_profiling.py – ydata-profiling (formerly pandas-profiling) tests.
"""

import numpy as np
import pandas as pd
import pytest


class TestYDataProfiling:
    @pytest.fixture()
    def sample_df(self):
        """Small DataFrame for profiling."""
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "int_col": rng.integers(0, 100, size=200),
                "float_col": rng.standard_normal(200),
                "cat_col": rng.choice(["A", "B", "C", "D"], size=200),
                "bool_col": rng.choice([True, False], size=200),
                "date_col": pd.date_range("2024-01-01", periods=200, freq="h"),
            }
        )

    def test_import(self):
        from ydata_profiling import ProfileReport  # noqa: F401

    def test_minimal_profile(self, sample_df):
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            sample_df,
            title="Test Profile",
            minimal=True,
            progress_bar=False,
        )
        # Just make sure it can generate the description set
        desc = profile.get_description()
        assert desc is not None

    def test_profile_to_json(self, sample_df):
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            sample_df,
            title="Test JSON",
            minimal=True,
            progress_bar=False,
        )
        json_data = profile.to_json()
        assert isinstance(json_data, str)
        assert len(json_data) > 0

    def test_profile_to_html(self, sample_df):
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            sample_df,
            title="Test HTML",
            minimal=True,
            progress_bar=False,
        )
        html = profile.to_html()
        assert "<html" in html.lower() or "<div" in html.lower()

    def test_profile_variables(self, sample_df):
        """Check that the profile detects all columns."""
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            sample_df,
            title="Variables",
            minimal=True,
            progress_bar=False,
        )
        desc = profile.get_description()
        # The description should include stats for each variable
        variables = desc.variables
        assert len(variables) >= len(sample_df.columns)
