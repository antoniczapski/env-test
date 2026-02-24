"""
test_numpy_pandas.py – NumPy ↔ pandas compatibility and core functionality tests.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# NumPy core
# ---------------------------------------------------------------------------
class TestNumpyCore:
    """Basic NumPy operations."""

    def test_array_creation(self):
        a = np.array([1, 2, 3])
        assert a.shape == (3,)
        assert a.dtype == np.int64 or np.issubdtype(a.dtype, np.integer)

    def test_dtypes(self):
        for dt in (np.float32, np.float64, np.int32, np.int64, np.bool_):
            a = np.ones(5, dtype=dt)
            assert a.dtype == dt

    def test_arithmetic(self):
        a = np.arange(5, dtype=np.float64)
        assert np.allclose(a * 2, [0, 2, 4, 6, 8])
        assert np.allclose(a + a, a * 2)

    def test_linear_algebra(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        inv = np.linalg.inv(A)
        assert np.allclose(A @ inv, np.eye(2))

    def test_random(self):
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(1000)
        assert samples.shape == (1000,)
        assert -0.5 < samples.mean() < 0.5

    def test_broadcasting(self):
        a = np.ones((3, 4))
        b = np.arange(4)
        result = a + b
        assert result.shape == (3, 4)
        assert np.allclose(result[0], [1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Pandas core
# ---------------------------------------------------------------------------
class TestPandasCore:
    """Basic pandas operations."""

    def test_dataframe_creation(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        assert df.shape == (3, 2)
        assert list(df.columns) == ["a", "b"]

    def test_series_operations(self):
        s = pd.Series([10, 20, 30, 40, 50])
        assert s.mean() == 30.0
        assert s.sum() == 150

    def test_groupby(self):
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})
        result = df.groupby("group")["value"].sum()
        assert result["a"] == 3
        assert result["b"] == 7

    def test_missing_values(self):
        df = pd.DataFrame({"x": [1, np.nan, 3]})
        assert df["x"].isna().sum() == 1
        filled = df["x"].fillna(0)
        assert filled.isna().sum() == 0

    def test_categorical(self):
        s = pd.Categorical(["a", "b", "a", "c"])
        assert len(s.categories) == 3

    def test_merge(self):
        left = pd.DataFrame({"key": [1, 2, 3], "val_l": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [2, 3, 4], "val_r": ["x", "y", "z"]})
        merged = pd.merge(left, right, on="key", how="inner")
        assert len(merged) == 2
        assert set(merged["key"]) == {2, 3}


# ---------------------------------------------------------------------------
# NumPy ↔ pandas interoperability
# ---------------------------------------------------------------------------
class TestNumpyPandasCompat:
    """Ensure seamless data exchange between NumPy and pandas."""

    def test_array_to_series_roundtrip(self):
        arr = np.arange(10, dtype=np.float64)
        s = pd.Series(arr)
        assert np.array_equal(s.values, arr)

    def test_array_to_dataframe_roundtrip(self):
        arr = np.random.default_rng(0).standard_normal((50, 5))
        df = pd.DataFrame(arr, columns=[f"c{i}" for i in range(5)])
        recovered = df.values
        assert np.allclose(arr, recovered)

    def test_dataframe_to_numpy(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        arr = df.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)

    def test_dtype_preservation_float(self):
        for dt in (np.float32, np.float64):
            arr = np.ones(5, dtype=dt)
            s = pd.Series(arr)
            assert s.dtype == dt

    def test_dtype_preservation_int(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        s = pd.Series(arr)
        assert s.dtype == np.int32

    def test_index_from_numpy(self):
        idx = pd.Index(np.arange(5))
        assert len(idx) == 5

    def test_nan_handling(self):
        arr = np.array([1.0, np.nan, 3.0])
        s = pd.Series(arr)
        assert s.isna().sum() == 1
        assert np.isnan(s.values[1])

    def test_apply_numpy_ufunc_on_series(self):
        s = pd.Series([0.0, 1.0, 2.0])
        result = np.exp(s)
        assert isinstance(result, pd.Series)
        assert np.allclose(result.values, np.exp([0.0, 1.0, 2.0]))

    def test_apply_numpy_ufunc_on_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 4.0, 9.0]})
        result = np.sqrt(df)
        assert isinstance(result, pd.DataFrame)
        assert np.allclose(result["a"].values, [1.0, 2.0, 3.0])

    def test_mixed_dtype_dataframe_to_numpy(self):
        df = pd.DataFrame({"int_col": [1, 2], "float_col": [1.5, 2.5]})
        arr = df.to_numpy()
        assert arr.dtype in (np.float64, np.object_)
