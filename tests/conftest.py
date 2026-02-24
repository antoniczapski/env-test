"""
conftest.py – shared fixtures for the environment test suite.

Provides reusable datasets (classification & regression) built with
scikit-learn so every test module starts from the same data.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Classification dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def classification_data():
    """Binary classification dataset: 1000 samples, 20 features."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def multiclass_data():
    """Multi-class classification dataset: 1000 samples, 20 features, 5 classes."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Regression dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def regression_data():
    """Regression dataset: 1000 samples, 20 features."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        noise=0.1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Pandas DataFrame version (for libraries that prefer DataFrames)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def classification_df(classification_data):
    """Classification data as pandas DataFrames."""
    import pandas as pd

    X_train, X_test, y_train, y_test = classification_data
    cols = [f"feat_{i}" for i in range(X_train.shape[1])]
    return (
        pd.DataFrame(X_train, columns=cols),
        pd.DataFrame(X_test, columns=cols),
        pd.Series(y_train, name="target"),
        pd.Series(y_test, name="target"),
    )


@pytest.fixture(scope="session")
def regression_df(regression_data):
    """Regression data as pandas DataFrames."""
    import pandas as pd

    X_train, X_test, y_train, y_test = regression_data
    cols = [f"feat_{i}" for i in range(X_train.shape[1])]
    return (
        pd.DataFrame(X_train, columns=cols),
        pd.DataFrame(X_test, columns=cols),
        pd.Series(y_train, name="target"),
        pd.Series(y_test, name="target"),
    )


# ---------------------------------------------------------------------------
# Small dataset with categorical features (for CatBoost / LightGBM)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def categorical_df():
    """Small dataset with mixed numeric + categorical columns."""
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame(
        {
            "num_1": rng.standard_normal(n),
            "num_2": rng.standard_normal(n),
            "cat_1": rng.choice(["A", "B", "C"], size=n),
            "cat_2": rng.choice(["X", "Y"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    train = df.iloc[:400]
    test = df.iloc[400:]
    feature_cols = ["num_1", "num_2", "cat_1", "cat_2"]
    return (
        train[feature_cols],
        test[feature_cols],
        train["target"],
        test["target"],
        ["cat_1", "cat_2"],  # categorical column names
    )
