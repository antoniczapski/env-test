# env-test

Environment compatibility test suite for a Python ML stack.

## Required Python version

Python **3.10+** (target: **3.13**)

## Libraries under test

| Library | Purpose |
|---|---|
| **catboost** | Gradient boosting with native categorical support |
| **xgboost** | Gradient boosting (native + sklearn API) |
| **scikit-learn** | ML toolkit, pipelines, preprocessing, metrics |
| **lightgbm** | Gradient boosting (native + sklearn API) |
| **ydata-profiling** | Automated exploratory data analysis |
| **ipywidgets** | Jupyter interactive widgets |

## Repository structure

```
.
├── README.md
├── requirements-test.txt       # reference list of required packages (not auto-installed)
├── run_tests.sh                # read-only test runner (never installs anything)
└── tests/
    ├── conftest.py             # shared fixtures (datasets)
    ├── test_imports.py         # import & version checks for every library
    ├── test_numpy_pandas.py    # NumPy & pandas core + interop
    ├── test_sklearn.py         # scikit-learn core functionality
    ├── test_xgboost.py         # XGBoost native + sklearn API
    ├── test_lightgbm.py        # LightGBM native + sklearn API
    ├── test_catboost.py        # CatBoost native + sklearn API
    ├── test_ipywidgets.py      # ipywidgets widget creation & state
    ├── test_ydata_profiling.py # ydata-profiling report generation
    └── test_cross_library.py   # cross-library integration
```

## Quick start

```bash
git clone <repo-url>
cd env-test
./run_tests.sh
```

> **Read-only:** The test runner does **not** install, upgrade, or remove any
> packages. It only verifies that the environment already has the expected
> setup. This makes it safe to run on any machine without side-effects.

### What the script does

1. Checks that Python and pytest are available.
2. Runs `pytest tests/ -v --tb=short`.

All required dependencies (see `requirements-test.txt`) must already be
installed in the target environment before running the tests.

### Manual run (without the script)

```bash
python -m pytest tests/ -v --tb=short
```

### Options

```bash
# Use a specific Python
PYTHON=/usr/bin/python3.13 ./run_tests.sh

# Pass extra pytest flags
./run_tests.sh -x -s --timeout=60
```

## What is tested (84 tests)

### Imports & versions (`test_imports.py`)
- Every required library can be imported.
- Every library exposes a version string.
- Python version ≥ 3.10.
- NumPy ↔ pandas round-trip.

### NumPy & pandas (`test_numpy_pandas.py`)
- NumPy: array creation, dtypes, arithmetic, linear algebra, random, broadcasting.
- pandas: DataFrame creation, Series ops, groupby, missing values, categoricals, merge.
- Interop: array↔Series/DataFrame round-trips, dtype preservation, NaN handling, ufunc dispatch.

### scikit-learn (`test_sklearn.py`)
- Classification (LogisticRegression, RandomForest, GradientBoosting).
- Regression (Ridge, RandomForestRegressor).
- Pipeline (fit/predict, cross_val_score).
- StandardScaler, train_test_split determinism.

### XGBoost (`test_xgboost.py`)
- Native API: DMatrix, binary classification, regression, feature importance.
- sklearn API: XGBClassifier, XGBRegressor, Pipeline, cross-val, predict_proba.

### LightGBM (`test_lightgbm.py`)
- Native API: Dataset, binary classification, regression, feature importance, categorical features.
- sklearn API: LGBMClassifier, LGBMRegressor, Pipeline, cross-val, predict_proba.

### CatBoost (`test_catboost.py`)
- Native API: Pool, classification, regression, feature importance, categorical features.
- sklearn API: Pipeline, cross-val, predict_proba.

### ydata-profiling (`test_ydata_profiling.py`)
- Import ProfileReport.
- Generate minimal profile, export to JSON, export to HTML.
- Verify all columns detected.

### ipywidgets (`test_ipywidgets.py`)
- Widget types: IntSlider, FloatSlider, Text, Dropdown, Checkbox, Output.
- Layouts: HBox, VBox.
- Interactive function wrapping.
- Widget state serialization.

### Cross-library integration (`test_cross_library.py`)
- All boosting libs accept NumPy arrays and pandas DataFrames.
- sklearn `VotingClassifier` (hard + soft) with XGB + LGBM + CatBoost.
- sklearn `VotingRegressor` with all three.
- sklearn `StackingClassifier` with mixed back-ends.
- Prediction shape consistency across all libraries.
- Pickle round-trip for all boosting models.
- Multi-class classification across all libraries.
