"""
test_catboost.py – CatBoost standalone and sklearn-integration tests.
"""

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, r2_score


class TestCatBoostNativeAPI:
    def test_pool_creation(self, classification_data):
        X_train, _, y_train, _ = classification_data
        pool = Pool(X_train, label=y_train)
        assert pool.num_row() == len(X_train)
        assert pool.num_col() == X_train.shape[1]

    def test_binary_classification(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = CatBoostClassifier(
            iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        assert acc > 0.5

    def test_regression(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = CatBoostRegressor(
            iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        assert r2 > 0.5

    def test_feature_importance(self, classification_data):
        X_train, _, y_train, _ = classification_data
        model = CatBoostClassifier(
            iterations=30, depth=4, random_seed=42, verbose=0
        )
        model.fit(X_train, y_train)
        importance = model.get_feature_importance()
        assert len(importance) == X_train.shape[1]
        assert importance.sum() > 0

    def test_categorical_features(self, categorical_df):
        """CatBoost handles categorical features natively."""
        X_train, X_test, y_train, y_test, cat_cols = categorical_df
        cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
        model = CatBoostClassifier(
            iterations=30,
            depth=4,
            random_seed=42,
            verbose=0,
            cat_features=cat_indices,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_proba(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = CatBoostClassifier(
            iterations=30, depth=4, random_seed=42, verbose=0
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestCatBoostSklearnCompat:
    def test_sklearn_pipeline(self, classification_data):
        """CatBoost works inside an sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = classification_data
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "cb",
                    CatBoostClassifier(
                        iterations=30, depth=3, random_seed=42, verbose=0
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        assert acc > 0.5

    def test_cross_val(self, classification_data):
        from sklearn.model_selection import cross_val_score

        X_train, _, y_train, _ = classification_data
        model = CatBoostClassifier(
            iterations=30, depth=3, random_seed=42, verbose=0
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        assert scores.mean() > 0.5
