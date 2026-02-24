"""
test_imports.py – Verify that every required library can be imported
and reports a sensible version string.
"""

import importlib
import sys

import pytest

# (module_name, pip_package_name)
REQUIRED = [
    ("catboost", "catboost"),
    ("xgboost", "xgboost"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("ydata_profiling", "ydata-profiling"),
    ("ipywidgets", "ipywidgets"),
    # Core scientific stack (should already be present)
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
]


@pytest.mark.parametrize("module_name, package_name", REQUIRED, ids=[r[1] for r in REQUIRED])
def test_import(module_name, package_name):
    """Library can be imported without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None, f"Failed to import {module_name}"


@pytest.mark.parametrize("module_name, package_name", REQUIRED, ids=[r[1] for r in REQUIRED])
def test_version(module_name, package_name):
    """Library exposes a version string."""
    mod = importlib.import_module(module_name)
    version = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
    # ydata-profiling stores version differently
    if version is None:
        try:
            from importlib.metadata import version as meta_version
            version = meta_version(package_name)
        except Exception:
            pass
    assert version is not None, f"{package_name} does not expose a version"
    print(f"  {package_name}=={version}")


def test_python_version():
    """Python >= 3.10 (expected 3.13)."""
    assert sys.version_info >= (3, 10), f"Python {sys.version} is too old"


def test_numpy_pandas_compat():
    """NumPy arrays can be converted to pandas and back."""
    import numpy as np
    import pandas as pd

    arr = np.arange(10, dtype=np.float64)
    s = pd.Series(arr)
    assert (s.values == arr).all()
