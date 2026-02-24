"""
test_sklearn.py – scikit-learn core functionality tests.
"""

import numpy as np
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestSklearnClassification:
    def test_logistic_regression(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.5, f"Accuracy too low: {acc}"

    def test_random_forest(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.5

    def test_gradient_boosting(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.5

    def test_predict_proba(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestSklearnRegression:
    def test_ridge(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        assert r2 > 0.5

    def test_random_forest_regressor(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        assert r2 > 0.5


class TestSklearnPipeline:
    def test_pipeline_fit_predict(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=42)),
            ]
        )
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        assert acc > 0.5

    def test_cross_val_score(self, classification_data):
        X_train, _, y_train, _ = classification_data
        model = LogisticRegression(max_iter=500, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        assert scores.mean() > 0.5
        assert len(scores) == 3


class TestSklearnUtilities:
    def test_standard_scaler(self, classification_data):
        X_train, X_test, _, _ = classification_data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        assert abs(X_scaled.mean()) < 0.1
        assert abs(X_scaled.std() - 1.0) < 0.1

    def test_train_test_split_deterministic(self):
        from sklearn.model_selection import train_test_split

        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X1, _, y1, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        X2, _, y2, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
