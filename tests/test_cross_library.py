"""
test_cross_library.py – Cross-library compatibility and integration tests.

Ensures that the ML libraries (scikit-learn, XGBoost, LightGBM, CatBoost)
work together seamlessly: shared data, ensemble voting, model comparison,
serialization, and data handoff.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, r2_score


# -----------------------------------------------------------------------
# 1. Shared data round-trip: numpy → pandas → each library
# -----------------------------------------------------------------------
class TestDataInterop:
    def test_numpy_to_all_classifiers(self, classification_data):
        """All boosting libraries accept plain numpy arrays."""
        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = classification_data
        models = {
            "xgb": xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss", random_state=42, verbosity=0),
            "lgbm": lgb.LGBMClassifier(n_estimators=20, num_leaves=15, random_state=42, verbose=-1),
            "catboost": catboost.CatBoostClassifier(iterations=20, depth=3, random_seed=42, verbose=0),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            assert len(preds) == len(y_test), f"{name}: wrong prediction length"

    def test_pandas_to_all_classifiers(self, classification_df):
        """All boosting libraries accept pandas DataFrames."""
        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = classification_df
        models = {
            "xgb": xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss", random_state=42, verbosity=0),
            "lgbm": lgb.LGBMClassifier(n_estimators=20, num_leaves=15, random_state=42, verbose=-1),
            "catboost": catboost.CatBoostClassifier(iterations=20, depth=3, random_seed=42, verbose=0),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            assert len(preds) == len(y_test), f"{name}: wrong prediction length"


# -----------------------------------------------------------------------
# 2. sklearn VotingClassifier with XGB + LGBM + CatBoost
# -----------------------------------------------------------------------
class TestEnsembleVoting:
    def test_voting_classifier(self, classification_data):
        from sklearn.ensemble import VotingClassifier

        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = classification_data

        estimators = [
            ("xgb", xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss", random_state=42, verbosity=0)),
            ("lgbm", lgb.LGBMClassifier(n_estimators=20, num_leaves=15, random_state=42, verbose=-1)),
            ("cb", catboost.CatBoostClassifier(iterations=20, depth=3, random_seed=42, verbose=0)),
        ]
        voting = VotingClassifier(estimators=estimators, voting="hard")
        voting.fit(X_train, y_train)
        acc = accuracy_score(y_test, voting.predict(X_test))
        assert acc > 0.5

    def test_soft_voting_classifier(self, classification_data):
        from sklearn.ensemble import VotingClassifier

        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = classification_data

        estimators = [
            ("xgb", xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss", random_state=42, verbosity=0)),
            ("lgbm", lgb.LGBMClassifier(n_estimators=20, num_leaves=15, random_state=42, verbose=-1)),
            ("cb", catboost.CatBoostClassifier(iterations=20, depth=3, random_seed=42, verbose=0)),
        ]
        voting = VotingClassifier(estimators=estimators, voting="soft")
        voting.fit(X_train, y_train)
        proba = voting.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)


# -----------------------------------------------------------------------
# 3. sklearn VotingRegressor with XGB + LGBM + CatBoost
# -----------------------------------------------------------------------
class TestEnsembleVotingRegression:
    def test_voting_regressor(self, regression_data):
        from sklearn.ensemble import VotingRegressor

        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = regression_data

        estimators = [
            ("xgb", xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42, verbosity=0)),
            ("lgbm", lgb.LGBMRegressor(n_estimators=20, num_leaves=15, random_state=42, verbose=-1)),
            ("cb", catboost.CatBoostRegressor(iterations=20, depth=3, random_seed=42, verbose=0)),
        ]
        voting = VotingRegressor(estimators=estimators)
        voting.fit(X_train, y_train)
        r2 = r2_score(y_test, voting.predict(X_test))
        assert r2 > 0.5


# -----------------------------------------------------------------------
# 4. Model comparison – all return same-shape predictions
# -----------------------------------------------------------------------
class TestModelComparison:
    def test_prediction_shapes_match(self, classification_data):
        """All models produce predictions of identical shape."""
        from sklearn.ensemble import RandomForestClassifier

        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = classification_data
        models = [
            RandomForestClassifier(n_estimators=20, random_state=42),
            xgb.XGBClassifier(n_estimators=20, eval_metric="logloss", random_state=42, verbosity=0),
            lgb.LGBMClassifier(n_estimators=20, random_state=42, verbose=-1),
            catboost.CatBoostClassifier(iterations=20, random_seed=42, verbose=0),
        ]
        shapes = []
        for m in models:
            m.fit(X_train, y_train)
            shapes.append(m.predict(X_test).shape)
        assert len(set(shapes)) == 1, f"Shapes differ: {shapes}"

    def test_proba_shapes_match(self, classification_data):
        """predict_proba shapes are identical across all boosting libs."""
        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = classification_data
        models = [
            xgb.XGBClassifier(n_estimators=20, eval_metric="logloss", random_state=42, verbosity=0),
            lgb.LGBMClassifier(n_estimators=20, random_state=42, verbose=-1),
            catboost.CatBoostClassifier(iterations=20, random_seed=42, verbose=0),
        ]
        shapes = []
        for m in models:
            m.fit(X_train, y_train)
            shapes.append(m.predict_proba(X_test).shape)
        assert len(set(shapes)) == 1, f"Proba shapes differ: {shapes}"


# -----------------------------------------------------------------------
# 5. Serialization (pickle) – all sklearn-compatible models
# -----------------------------------------------------------------------
class TestSerialization:
    @pytest.mark.parametrize(
        "model_factory",
        [
            pytest.param(
                lambda: __import__("xgboost").XGBClassifier(n_estimators=10, eval_metric="logloss", random_state=42, verbosity=0),
                id="xgboost",
            ),
            pytest.param(
                lambda: __import__("lightgbm").LGBMClassifier(n_estimators=10, random_state=42, verbose=-1),
                id="lightgbm",
            ),
            pytest.param(
                lambda: __import__("catboost").CatBoostClassifier(iterations=10, random_seed=42, verbose=0),
                id="catboost",
            ),
        ],
    )
    def test_pickle_round_trip(self, classification_data, model_factory):
        X_train, X_test, y_train, y_test = classification_data
        model = model_factory()
        model.fit(X_train, y_train)
        preds_before = model.predict(X_test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            tmp_path = f.name

        with open(tmp_path, "rb") as f:
            loaded = pickle.load(f)

        preds_after = loaded.predict(X_test)
        np.testing.assert_array_equal(preds_before, preds_after)
        Path(tmp_path).unlink()


# -----------------------------------------------------------------------
# 6. StackingClassifier with mixed boosting back-ends
# -----------------------------------------------------------------------
class TestStacking:
    def test_stacking_classifier(self, classification_data):
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression

        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = classification_data
        estimators = [
            ("xgb", xgb.XGBClassifier(n_estimators=15, eval_metric="logloss", random_state=42, verbosity=0)),
            ("lgbm", lgb.LGBMClassifier(n_estimators=15, random_state=42, verbose=-1)),
            ("cb", catboost.CatBoostClassifier(iterations=15, random_seed=42, verbose=0)),
        ]
        stacker = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=3,
        )
        stacker.fit(X_train, y_train)
        acc = accuracy_score(y_test, stacker.predict(X_test))
        assert acc > 0.5


# -----------------------------------------------------------------------
# 7. Multiclass – all libraries handle >2 classes
# -----------------------------------------------------------------------
class TestMulticlass:
    def test_multiclass_all(self, multiclass_data):
        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        X_train, X_test, y_train, y_test = multiclass_data
        n_classes = len(set(y_train))

        models = {
            "xgb": xgb.XGBClassifier(n_estimators=20, eval_metric="mlogloss", random_state=42, verbosity=0),
            "lgbm": lgb.LGBMClassifier(n_estimators=20, random_state=42, verbose=-1),
            "catboost": catboost.CatBoostClassifier(iterations=20, random_seed=42, verbose=0),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            assert proba.shape == (len(X_test), n_classes), f"{name}: wrong shape"
            acc = accuracy_score(y_test, model.predict(X_test))
            assert acc > 1 / n_classes, f"{name}: accuracy worse than random"
