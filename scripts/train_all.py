"""End-to-end training pipeline.

This script runs the full training flow:
    1. Load the dataset referenced by ``src.config.DATA_FILE``.
    2. Preprocess it with :class:`src.preprocessor.DataPreprocessor`.
    3. Train every model with :class:`src.trainer.ModelTrainer`.
    4. Evaluate the models with :func:`src.evaluator.evaluate_all`.
    5. Persist models to ``models/`` and feature importance CSVs to
       ``outputs/feature_importance/``.

Missing optional libraries (e.g. ``xgboost``, ``lightgbm``, ``catboost``)
are handled gracefully so that at least the ``random_forest`` baseline
always runs.

Run from the project root with::

    python scripts/train_all.py
"""

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    print("=" * 60)
    print("Student Academic Risk - Training Pipeline")
    print("=" * 60)

    try:
        from src import config
        from src.preprocessor import DataPreprocessor
        from src.trainer import ModelTrainer
        from src.evaluator import evaluate_all
    except ImportError as exc:
        print("Missing core dependencies: {}".format(exc))
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    print("")
    print("[1/4] Loading and preprocessing data ...")
    try:
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess_all()
        if len(result) == 7:
            X_train, X_test, y_train, y_test, y_train_reg, y_test_reg, feature_cols = result
        else:
            X_train, X_test, y_train, y_test, feature_cols = result
            y_train_reg = None
            y_test_reg = None
    except FileNotFoundError as exc:
        print("Data file missing: {}".format(exc))
        print("Run: python scripts/generate_demo_data.py first")
        sys.exit(1)
    except Exception as exc:
        print("Preprocessing failed: {}".format(exc))
        sys.exit(1)

    print("")
    print("[2/4] Training models ...")
    trainer = ModelTrainer()
    try:
        trainer.train_all(X_train, y_train)
        if y_train_reg is not None:
            print("")
            print("  Training regression models ...")
            trainer.train_all_regression(X_train, y_train_reg)
        else:
            print("  No regression target available - skipping regression training")
    except Exception as exc:
        print("Full training failed ({}). Retrying with random_forest only.".format(exc))
        trainer.models = {}
        trainer.train_times = {}
        start = time.time()
        rf = trainer.build_model("random_forest")
        rf.fit(X_train, y_train)
        trainer.models["random_forest"] = rf
        trainer.train_times["random_forest"] = time.time() - start
        print("  random_forest done in {:.2f}s".format(trainer.train_times["random_forest"]))
        if y_train_reg is not None:
            trainer.reg_models = {}
            trainer.reg_train_times = {}
            start_r = time.time()
            rfr = trainer.build_regressor("random_forest")
            rfr.fit(X_train, y_train_reg)
            trainer.reg_models["random_forest"] = rfr
            trainer.reg_train_times["random_forest"] = time.time() - start_r
            print("  random_forest (regression) done in {:.2f}s".format(
                trainer.reg_train_times["random_forest"]))

    print("")
    print("[3/4] Evaluating models ...")
    try:
        results = evaluate_all(trainer.models, X_test, y_test)
    except Exception as exc:
        print("Evaluation failed: {}".format(exc))
        results = {}

    if y_test_reg is not None:
        print("")
        print("[3.5/4] Evaluating regression models ...")
        try:
            reg_results = trainer.evaluate_regression(X_test, y_test_reg)
        except Exception as exc:
            print("Regression evaluation failed: {}".format(exc))
            reg_results = {}
    else:
        reg_results = {}

    print("")
    print("[4/4] Saving artifacts ...")
    try:
        trainer.save_models()
    except Exception as exc:
        print("Saving models failed: {}".format(exc))

    # Save categorical column encoders so the server can translate
    # user-friendly string values (e.g. "Male") into the numeric codes used
    # by the trained models.
    try:
        import json
        encoders = preprocessor.export_encoder_map()
        encoder_path = os.path.join(config.MODEL_DIR, "feature_encoders.json")
        with open(encoder_path, "w", encoding="utf-8") as f:
            json.dump(encoders, f, ensure_ascii=False, indent=2)
        if encoders:
            print("  Saved {} categorical encoder maps to {}".format(
                len(encoders), encoder_path))
        else:
            print("  No categorical columns found - encoder map empty.")
    except Exception as exc:
        print("  Encoder map save skipped: {}".format(exc))

    # Save the ordered feature column names so the server always uses the
    # same column order as the training pipeline.
    try:
        import json
        meta_path = os.path.join(config.MODEL_DIR, "feature_columns.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"feature_columns": list(feature_cols)}, f,
                      ensure_ascii=False, indent=2)
        print("  Saved feature column metadata: {}".format(meta_path))
    except Exception as exc:
        print("  Feature column metadata save skipped: {}".format(exc))

    os.makedirs(config.FEATURE_IMPORTANCE_DIR, exist_ok=True)
    for name in list(trainer.models.keys()):
        try:
            fi = trainer.get_feature_importance(name, feature_cols, top_n=15)
            if fi is not None and not fi.empty:
                out_path = os.path.join(
                    config.FEATURE_IMPORTANCE_DIR, name + ".csv"
                )
                fi.to_csv(out_path, index=False)
                print("  Saved feature importance: {}".format(out_path))
        except Exception as exc:
            print("  Feature importance failed for {}: {}".format(name, exc))

    print("")
    print("=" * 60)
    header = "{:<20}{:>10}{:>10}{:>10}{:>10}".format(
        "Model", "Accuracy", "F1", "AUC", "Time(s)"
    )
    print(header)
    print("-" * 60)
    for name in config.MODEL_NAMES:
        if name not in trainer.models:
            continue
        r = results.get(name, {"accuracy": float("nan"), "f1": float("nan"), "auc": float("nan")})
        label = config.MODEL_LABELS.get(name, name)
        t = trainer.train_times.get(name, 0.0)
        print(
            "{:<20}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.2f}".format(
                label, r["accuracy"], r["f1"], r["auc"], t
            )
        )
    print("=" * 60)

    if reg_results:
        print("")
        print("=" * 60)
        header_r = "{:<20}{:>12}{:>12}{:>12}{:>12}".format(
            "Model(reg)", "MSE", "MAE", "RMSE", "R2"
        )
        print(header_r)
        print("-" * 60)
        for name in config.MODEL_NAMES:
            if name not in trainer.reg_models:
                continue
            rr = reg_results.get(name, {})
            label = config.MODEL_LABELS.get(name, name)
            print(
                "{:<20}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}".format(
                    label,
                    rr.get("mse", float("nan")),
                    rr.get("mae", float("nan")),
                    rr.get("rmse", float("nan")),
                    rr.get("r2", float("nan")),
                )
            )
        print("=" * 60)
    print("")
    print("Done! Now run: python app/server.py")


if __name__ == "__main__":
    main()
