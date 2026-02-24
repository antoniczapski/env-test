"""
test_xgboost.py – XGBoost standalone and sklearn-integration tests.
"""

import numpy as np
import pytest
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score


class TestXGBoostNativeAPI:
    def test_dmatrix_creation(self, classification_data):
        X_train, _, y_train, _ = classification_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        assert dtrain.num_row() == len(X_train)
        assert dtrain.num_col() == X_train.shape[1]

    def test_binary_classification(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "seed": 42,
        }
        bst = xgb.train(params, dtrain, num_boost_round=50)
        preds = bst.predict(dtest)
        labels = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, labels)
        assert acc > 0.5

    def test_regression(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params = {
            "objective": "reg:squarederror",
            "max_depth": 4,
            "seed": 42,
        }
        bst = xgb.train(params, dtrain, num_boost_round=50)
        preds = bst.predict(dtest)
        r2 = r2_score(y_test, preds)
        assert r2 > 0.5

    def test_feature_importance(self, classification_data):
        X_train, _, y_train, _ = classification_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {"objective": "binary:logistic", "max_depth": 4, "seed": 42}
        bst = xgb.train(params, dtrain, num_boost_round=30)
        importance = bst.get_score(importance_type="gain")
        assert len(importance) > 0


class TestXGBoostSklearnAPI:
    def test_xgb_classifier(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.5

    def test_xgb_regressor(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        assert r2 > 0.5

    def test_sklearn_pipeline_compat(self, classification_data):
        """XGBoost works inside an sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = classification_data
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        n_estimators=30,
                        max_depth=3,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        assert acc > 0.5

    def test_xgb_cross_val(self, classification_data):
        from sklearn.model_selection import cross_val_score

        X_train, _, y_train, _ = classification_data
        model = xgb.XGBClassifier(
            n_estimators=30,
            max_depth=3,
            eval_metric="logloss",
            random_state=42,
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        assert scores.mean() > 0.5

    def test_predict_proba(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = xgb.XGBClassifier(
            n_estimators=30, max_depth=3, eval_metric="logloss", random_state=42
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
