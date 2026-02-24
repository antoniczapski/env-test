"""
test_lightgbm.py – LightGBM standalone and sklearn-integration tests.
"""

import numpy as np
import pytest
import lightgbm as lgb
from sklearn.metrics import accuracy_score, r2_score


class TestLightGBMNativeAPI:
    def test_dataset_creation(self, classification_data):
        X_train, _, y_train, _ = classification_data
        ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        ds.construct()
        assert ds.num_data() == len(X_train)
        assert ds.num_feature() == X_train.shape[1]

    def test_binary_classification(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "seed": 42,
            "verbose": -1,
        }
        bst = lgb.train(params, dtrain, num_boost_round=50)
        preds = bst.predict(X_test)
        labels = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, labels)
        assert acc > 0.5

    def test_regression(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "seed": 42,
            "verbose": -1,
        }
        bst = lgb.train(params, dtrain, num_boost_round=50)
        preds = bst.predict(X_test)
        r2 = r2_score(y_test, preds)
        assert r2 > 0.5

    def test_feature_importance(self, classification_data):
        X_train, _, y_train, _ = classification_data
        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {"objective": "binary", "num_leaves": 31, "verbose": -1, "seed": 42}
        bst = lgb.train(params, dtrain, num_boost_round=30)
        importance = bst.feature_importance(importance_type="gain")
        assert len(importance) == X_train.shape[1]
        assert importance.sum() > 0

    def test_categorical_feature(self, categorical_df):
        """LightGBM handles categorical features natively."""
        X_train, X_test, y_train, y_test, cat_cols = categorical_df
        # Encode categoricals as category dtype
        X_tr = X_train.copy()
        X_te = X_test.copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("category")
            X_te[c] = X_te[c].astype("category")
        dtrain = lgb.Dataset(X_tr, label=y_train)
        params = {
            "objective": "binary",
            "num_leaves": 15,
            "verbose": -1,
            "seed": 42,
        }
        bst = lgb.train(params, dtrain, num_boost_round=30)
        preds = bst.predict(X_te)
        assert preds.shape == (len(X_test),)


class TestLightGBMSklearnAPI:
    def test_lgbm_classifier(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = lgb.LGBMClassifier(
            n_estimators=50, num_leaves=31, random_state=42, verbose=-1
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.5

    def test_lgbm_regressor(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = lgb.LGBMRegressor(
            n_estimators=50, num_leaves=31, random_state=42, verbose=-1
        )
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        assert r2 > 0.5

    def test_sklearn_pipeline_compat(self, classification_data):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = classification_data
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lgbm",
                    lgb.LGBMClassifier(
                        n_estimators=30, num_leaves=15, random_state=42, verbose=-1
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
        model = lgb.LGBMClassifier(
            n_estimators=30, num_leaves=15, random_state=42, verbose=-1
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        assert scores.mean() > 0.5

    def test_predict_proba(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = lgb.LGBMClassifier(
            n_estimators=30, num_leaves=15, random_state=42, verbose=-1
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
