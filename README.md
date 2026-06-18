# Student Academic Risk Intelligent Early-Warning System

A data-driven early-warning platform that identifies college students at
risk of poor academic performance, helps instructors prioritize
interventions, and provides an interpretable, model-backed prediction
interface.

## What this project does

The system ingests a tabular dataset of student features (attendance,
grades, engagement metrics, etc.) and trains five supervised classifiers
(Random Forest, XGBoost, LightGBM, CatBoost, and a soft-voting
ensemble) to predict an `Academic_Risk_Class` label (low / medium /
high risk).  Trained models are then served through a Flask web
application that lets users fill in feature values, inspect predicted
risk, and visualize per-model feature importance.

## Key Features

* **Multi-model comparison** - five classifiers are trained and evaluated
  together so educators can pick the trade-off between interpretability
  and accuracy that fits their workflow.
* **Human-readable predictions** - the web UI shows predicted risk,
  per-class probabilities, and the top features that deviate from the
  student population mean.
* **Lightweight Flask JSON API** - `/api/status`, `/api/models`,
  `/api/predict`, and `/api/feature_importance` enable integration with
  dashboards, notebooks, and other downstream tools.
* **Graceful fallback** - missing optional libraries (XGBoost,
  LightGBM, CatBoost, SHAP) are detected at runtime; at minimum the
  Random Forest baseline always trains and serves predictions.
* **Reproducible pipeline** - every training run records accuracy,
  weighted F1, multi-class ROC-AUC, training time, and feature
  importance as machine-readable CSVs.

## Project Structure

```
project-root/
|-- app/                       # Flask web application
|   |-- routes/                # (reserved for future route modules)
|   |-- services/              # (reserved for domain services)
|   |-- static/
|   |   |-- css/style.css
|   |   |-- js/
|   |   |   |-- app.js
|   |   |   |-- models.js
|   |   |   `-- predict.js
|   |-- templates/
|   |   |-- about.html
|   |   |-- dashboard.html
|   |   |-- layout.html
|   |   |-- models.html
|   |   `-- predict.html
|   |-- __init__.py
|   `-- server.py              # Entry point: python app/server.py
|-- data/                      # Raw CSV datasets
|   `-- college_students_academic_performance_dataset.csv
|-- models/                    # Serialized scikit-learn estimators
|   `-- *.pkl
|-- outputs/
|   |-- confusion_matrix/
|   |-- shap_summary/
|   |-- model_comparison/
|   `-- feature_importance/    # Per-model top-15 feature importance
|-- scripts/
|   |-- train_all.py           # End-to-end training pipeline
|   `-- generate_demo_data.py  # Generates a synthetic dataset
|-- src/                       # Core algorithm modules
|   |-- __init__.py
|   |-- config.py              # Paths, targets, model names
|   |-- evaluator.py           # Accuracy / F1 / AUC computation
|   |-- model_manager.py       # Disk <-> in-memory model loader
|   |-- preprocessor.py        # Missing values + encoding + split
|   `-- trainer.py             # Builds and trains the five classifiers
|-- requirements.txt
`-- README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The project supports Python 3.8+. `numpy`, `pandas`, `scikit-learn`,
`flask`, and `joblib` are required; `xgboost`, `lightgbm`, `catboost`,
`shap`, `matplotlib`, `seaborn`, and `plotly` are optional but
recommended.

### 2. Prepare the dataset

Either drop your own CSV into
`data/college_students_academic_performance_dataset.csv` (ensure it
contains an `Academic_Risk_Class` column with integer labels `0/1/2`)
or generate synthetic demo data:

```bash
python scripts/generate_demo_data.py
```

### 3. Train the models

```bash
python scripts/train_all.py
```

This script loads the dataset, preprocesses it, trains every supported
classifier, prints a concise evaluation table, persists pickled models
to `models/`, and exports feature importance CSVs to
`outputs/feature_importance/`.

### 4. Start the web server

```bash
python app/server.py
```

Then open <http://localhost:5000/> in a browser.

## Models

All models are three-class classifiers (`0` = Low Risk, `1` = Medium
Risk, `2` = High Risk).  Training uses an 80/20 stratified train/test
split and a fixed random seed for reproducibility.

| Model         | Description                                                |
|---------------|------------------------------------------------------------|
| Random Forest | Interpretable ensemble of 200 decision trees; baseline.    |
| XGBoost       | Gradient boosting with column sampling (200 rounds).       |
| LightGBM      | Histogram-based gradient boosting (200 rounds).            |
| CatBoost      | Ordered boosting with native handling of categorical data. |
| Soft Voting   | Ensemble average of the four base classifiers above.       |

Per-model feature importance (top-15) is written to
`outputs/feature_importance/<model_name>.csv` during training.

## Web Platform

### Routes (HTML pages)

* `/` or `/dashboard` - summary dashboard with counts and links.
* `/predict` - interactive feature-input form + prediction card.
* `/models` - per-model status and feature importance charts.
* `/about` - project description and acknowledgements.

### API endpoints (JSON)

* `GET /api/status` - server health, model count, feature count.
* `GET /api/models` - lists available and registered model names.
* `POST /api/predict` - expects `{ "model": "<name>", "features": { "f0": 1.2, ... } }`; returns predicted risk, class probabilities, a human-readable explanation, and raw feature values.
* `GET /api/feature_importance?model=<name>` - returns the feature
  names and importance values for the requested model.

## Tech Stack

* **Language** - Python 3.8+
* **Data** - NumPy, Pandas
* **Machine Learning** - scikit-learn, XGBoost, LightGBM, CatBoost
* **Model persistence** - joblib
* **Explainability** - SHAP (optional)
* **Web framework** - Flask
* **Visualization** - Matplotlib, Seaborn, Plotly (for dashboards)

## FAQ

**Q: What if I only have scikit-learn installed?**
A: The pipeline falls back to training only the Random Forest model.
You can still start the web server and use the prediction endpoints.

**Q: Can I use my own features?**
A: Yes. Replace the CSV in the `data/` directory, keeping the
`Academic_Risk_Class` target column as an integer. The preprocessor
automatically fills missing numeric values with the median and drops
non-target categorical columns after label-encoding them.

**Q: How do I re-deploy after retraining?**
A: Re-run `python scripts/train_all.py`, then restart the Flask
process. The server auto-loads all `*.pkl` files found in `models/` on
startup.

## License

MIT License.
