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
├── requirements-test.txt       # pip dependencies for the test suite
├── run_tests.sh                # one-command test runner
└── tests/
    ├── conftest.py             # shared fixtures (datasets)
    ├── test_imports.py         # import & version checks for every library
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

### What the script does

1. Prints the active Python version.
2. Installs dependencies from `requirements-test.txt` (skip with `SKIP_INSTALL=1`).
3. Runs `pytest tests/ -v --tb=short`.

### Manual run (without the script)

```bash
pip install -r requirements-test.txt
python -m pytest tests/ -v --tb=short
```

### Options

```bash
# Skip pip install (deps already present)
SKIP_INSTALL=1 ./run_tests.sh

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
