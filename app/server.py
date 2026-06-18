"""Flask backend server for the College Student Academic Risk Early Warning System.

This module wires up a Flask web application that exposes:

- HTML pages (dashboard, predict, models, about) for human users.
- JSON APIs for status, model listing, prediction, and feature importance.

On startup the application loads every trained model it can find in the
``models/`` directory through :class:`src.model_manager.ModelManager`, and
reads the reference dataset from ``data/`` in order to compute feature
statistics (min / max / mean / median) that are used to fill missing input
values and to produce a simple human readable explanation for each
prediction.

Run with::

    python app/server.py

Then browse http://localhost:5000/
"""

import os
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config  # noqa: E402
from src.model_manager import ModelManager  # noqa: E402


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

model_manager = None
reference_df = None
feature_cols = []
feature_stats = {}
categorical_encoders = {}
RISK_LEVEL_MAP = {0: "低风险", 1: "中风险", 2: "高风险"}
TARGET_COL_REG_IN = getattr(config, "TARGET_REG", "Final_Academic_Score")

FEATURE_ZH = {
    "Age": "年龄",
    "Gender": "性别",
    "Course_Year": "年级",
    "Attendance_Percentage": "出勤率",
    "Assignment_Submission_Rate": "作业提交率",
    "Quiz_Average_Score": "小测平均分",
    "Midterm_Score": "期中成绩",
    "Previous_GPA": "过往GPA",
    "Online_Learning_Hours": "在线学习时长",
    "Course_Participation_Score": "课程参与度",
    "Forum_Activity_Count": "论坛活动次数",
    "Login_Frequency_Per_Week": "每周登录频率",
    "Learning_Resource_Access_Count": "学习资源访问次数",
    "Late_Submission_Count": "延迟提交次数",
    "Missing_Assignment_Count": "缺交作业次数",
    "Study_Plan_Adherence": "学习计划遵守度",
    "Academic_Counseling_Received": "是否接受学业咨询",
    "Tutoring_Support_Received": "是否接受辅导",
    "Adaptive_Learning_Resource_Used": "是否使用自适应学习资源",
}


def _ensure_initialized():
    """Lazy initialization called on first request."""
    global model_manager, reference_df, feature_cols, feature_stats
    if model_manager is None:
        init_application()


def _find_data_file():
    candidates = []
    if getattr(config, "DATA_FILE", None):
        candidates.append(config.DATA_FILE)
    data_dir = os.path.join(_PROJECT_ROOT, "data")
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.lower().endswith(".csv"):
                candidates.append(os.path.join(data_dir, name))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _load_reference_data():
    global reference_df, feature_cols, feature_stats, categorical_encoders
    data_path = _find_data_file()
    if data_path is None:
        print("[server] No reference CSV found - running in degraded mode.")
        reference_df = None
        feature_cols = []
        feature_stats = {}
        categorical_encoders = {}
        return
    try:
        reference_df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        reference_df = pd.read_csv(data_path, encoding="latin-1")
    exclude = set(getattr(config, "EXCLUDE_COLS", [])) | {
        getattr(config, "TARGET_CLASS", "Academic_Risk_Class"),
        getattr(config, "TARGET_REG", "Final_Academic_Score"),
    }
    feature_cols = [c for c in reference_df.columns if c not in exclude]

    # Override feature_cols with the training-time order (if available) so
    # that the column order matches what each model was fitted on.
    try:
        import json
        meta_path = os.path.join(
            getattr(config, "MODEL_DIR", os.path.join(_PROJECT_ROOT, "models")),
            "feature_columns.json",
        )
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            saved_cols = meta.get("feature_columns", [])
            if saved_cols:
                present = {c for c in reference_df.columns}
                feature_cols = [c for c in saved_cols if c in present]
                print("[server] Using training column order ({} cols).".format(
                    len(feature_cols)))
    except Exception:
        pass

    # Load categorical-encoder maps saved by the training pipeline. These
    # map column-name -> list of class labels (index = encoded value).
    try:
        import json
        enc_path = os.path.join(
            getattr(config, "MODEL_DIR", os.path.join(_PROJECT_ROOT, "models")),
            "feature_encoders.json",
        )
        if os.path.isfile(enc_path):
            with open(enc_path, "r", encoding="utf-8") as f:
                categorical_encoders = json.load(f) or {}
            if categorical_encoders:
                print("[server] Loaded {} categorical encoder maps: {}".format(
                    len(categorical_encoders),
                    ", ".join(sorted(categorical_encoders.keys())),
                ))
    except Exception:
        categorical_encoders = {}

    stats = {}
    for col in feature_cols:
        series = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        if series.empty:
            stats[col] = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        else:
            stats[col] = {
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
            }
    # Also keep regression-target stats for score-range display
    _reg_target = getattr(config, "TARGET_REG", "Final_Academic_Score")
    if _reg_target in reference_df.columns:
        series = pd.to_numeric(reference_df[_reg_target], errors="coerce").dropna()
        if not series.empty:
            stats[_reg_target] = {
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "std": round(float(series.std()), 2),
            }
    feature_stats = stats
    print("[server] Loaded reference dataset with {n} features.".format(n=len(feature_cols)))


def _init_model_manager():
    global model_manager
    manager = ModelManager()
    manager.load_all()
    model_manager = manager


def init_application():
    _load_reference_data()
    _init_model_manager()


def _coerce_features(input_values):
    coerced = {}
    for col in feature_cols:
        raw = input_values.get(col) if input_values else None
        if col in categorical_encoders:
            # Categorical column: map from label (e.g. "Male") to the
            # numeric encoding used during training (e.g. 0).
            classes = categorical_encoders[col]
            found = None
            if raw not in (None, ""):
                raw_str = str(raw).strip()
                for i, label in enumerate(classes):
                    if str(label) == raw_str or label == raw_str:
                        found = float(i)
                        break
            if found is not None:
                coerced[col] = found
            else:
                stats = feature_stats.get(col, {})
                coerced[col] = float(stats.get("mean", 0.0))
        else:
            if raw not in (None, ""):
                try:
                    coerced[col] = float(raw)
                except (TypeError, ValueError):
                    stats = feature_stats.get(col, {})
                    coerced[col] = float(stats.get("mean", 0.0))
            else:
                stats = feature_stats.get(col, {})
                coerced[col] = float(stats.get("mean", 0.0))
    ordered = [coerced[col] for col in feature_cols]
    return coerced, ordered


def _build_explanation(coerced, threshold=0.5):
    factors = []
    if reference_df is None:
        return factors
    for col in feature_cols:
        stats = feature_stats.get(col)
        if not stats:
            continue
        mean = stats["mean"]
        series = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        std = float(series.std()) if len(series) > 1 else 1.0
        value = coerced[col]
        deviation = (value - mean) / std if std else 0.0
        if abs(deviation) < threshold:
            continue
        factors.append({
            "feature": col,
            "value": round(value, 2),
            "mean": round(mean, 2),
            "deviation": round(deviation, 3),
            "direction": "高于均值" if deviation > 0 else "低于均值",
        })
    factors.sort(key=lambda f: abs(f["deviation"]), reverse=True)
    return factors[:5]


def _feature_importance_for(model_name):
    if model_manager is None:
        return [], []
    estimator = model_manager.get_model(model_name)
    if estimator is None:
        return [], []
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return [], []
    cols = feature_cols or ["f{}".format(i) for i in range(len(importances))]
    cols = cols[: len(importances)]
    values = [float(v) for v in importances[: len(cols)]]
    return cols, values


@app.route("/")
@app.route("/dashboard")
def dashboard():
    _ensure_initialized()
    available = model_manager.list_available() if model_manager else []
    return render_template(
        "dashboard.html",
        models_available=len(available),
        features_count=len(feature_cols),
        risk_labels=RISK_LEVEL_MAP,
    )


@app.route("/predict")
def predict_page():
    _ensure_initialized()
    # Build feature_zh is applied server-side so that if a column name has no
    # known Chinese label we fall back to the original English name.
    zh = {c: FEATURE_ZH.get(c, c) for c in feature_cols}
    return render_template(
        "predict.html",
        feature_cols=feature_cols,
        feature_stats=feature_stats,
        risk_labels=RISK_LEVEL_MAP,
        categorical=dict(categorical_encoders),
        feature_zh=zh,
    )


@app.route("/models")
def models_page():
    _ensure_initialized()
    available = model_manager.list_available() if model_manager else []
    labels = getattr(config, "MODEL_LABELS", {})
    all_models = getattr(config, "MODEL_NAMES", [])
    zh = {c: FEATURE_ZH.get(c, c) for c in feature_cols}
    return render_template(
        "models.html",
        available=available,
        all_models=all_models,
        labels=labels,
        feature_zh=zh,
    )


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/api/status")
def api_status():
    _ensure_initialized()
    available = model_manager.list_available() if model_manager else []
    current = available[0] if available else None
    return jsonify({
        "status": "ok",
        "models_available": len(available),
        "features_count": len(feature_cols),
        "current_model": current,
    })


@app.route("/api/models")
def api_models():
    _ensure_initialized()
    available = model_manager.list_available() if model_manager else []
    labels = dict(getattr(config, "MODEL_LABELS", {}))
    all_models = list(getattr(config, "MODEL_NAMES", []))
    current = available[0] if available else None
    return jsonify({
        "available": available,
        "all_models": all_models,
        "labels": labels,
        "current": current,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    _ensure_initialized()
    if model_manager is None:
        return jsonify({
            "success": False,
            "error": "服务器上没有可用的已训练模型。",
        }), 503
    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model")
    if not model_name:
        available = model_manager.list_available()
        model_name = available[0] if available else None
    if not model_name or model_name not in model_manager.list_available():
        return jsonify({
            "success": False,
            "error": "请求的模型不可用：{}".format(model_name),
        }), 400
    input_values = payload.get("features") or {}
    if not feature_cols:
        return jsonify({
            "success": False,
            "error": "特征元数据不可用，无法构造输入向量。",
        }), 503
    coerced, ordered = _coerce_features(input_values)
    X_df = pd.DataFrame([ordered], columns=list(feature_cols))
    estimator = model_manager.get_model(model_name)
    try:
        pred_raw = estimator.predict(X_df)
        arr = np.asarray(pred_raw).ravel()
        prediction = int(arr[0])
    except Exception as exc:
        return jsonify({"success": False, "error": "预测失败：{}".format(exc)}), 500
    probabilities = []
    if hasattr(estimator, "predict_proba"):
        try:
            proba_raw = estimator.predict_proba(X_df)
            proba = np.asarray(proba_raw).ravel()
            probabilities = [float(p) for p in proba]
        except Exception:
            probabilities = []
    # Regression score prediction (lazy - load regressor from disk only)
    predicted_score = None
    score_min = None
    score_max = None
    reg_estimator = model_manager.get_reg_model(model_name)
    if reg_estimator is not None:
        try:
            reg_pred = reg_estimator.predict(X_df)
            predicted_score = float(np.asarray(reg_pred).ravel()[0])
            if feature_stats and (TARGET_COL_REG_IN in feature_stats):
                stats = feature_stats[TARGET_COL_REG_IN]
                mean_val = stats.get("mean")
                std_val = stats.get("std")
                if mean_val is not None and std_val is not None:
                    score_min = float(round(mean_val - 2.0 * std_val, 2))
                    score_max = float(round(mean_val + 2.0 * std_val, 2))
                else:
                    score_min = float(round(predicted_score * 0.85, 2))
                    score_max = float(round(predicted_score * 1.15, 2))
            else:
                score_min = float(round(predicted_score * 0.85, 2))
                score_max = float(round(predicted_score * 1.15, 2))
            predicted_score = float(round(predicted_score, 2))
        except Exception:
            predicted_score = None
            score_min = None
            score_max = None
    explanation = {
        "key_factors": _build_explanation(coerced),
        "note": "关键因素用于标出输入值与参考数据集均值存在显著偏差的特征。",
    }
    labels = dict(getattr(config, "MODEL_LABELS", {}))
    return jsonify({
        "success": True,
        "risk_level": prediction,
        "risk_label": RISK_LEVEL_MAP.get(prediction, "Unknown"),
        "probabilities": probabilities,
        "explanation": explanation,
        "model_label": labels.get(model_name, model_name),
        "feature_names": list(feature_cols),
        "feature_values": coerced,
        "predicted_score": predicted_score,
        "score_min": score_min,
        "score_max": score_max,
    })


@app.route("/api/feature_importance")
def api_feature_importance():
    if model_manager is None:
        return jsonify({"success": False, "error": "Model manager is not initialized."}), 503
    model_name = request.args.get("model")
    if not model_name:
        available = model_manager.list_available()
        if not available:
            return jsonify({"success": False, "error": "No models are available."}), 400
        model_name = available[0]
    if model_name not in model_manager.models:
        return jsonify({"success": False, "error": "Model not found: {}".format(model_name)}), 404
    features, importances = _feature_importance_for(model_name)
    # Attach Chinese labels so the frontend can display a readable chart.
    zh_labels = [FEATURE_ZH.get(f, f) for f in features]
    return jsonify({
        "success": True,
        "features": features,
        "zh": zh_labels,
        "importances": importances,
        "model": model_name,
    })


def main():
    init_application()
    app.run(debug=True, host="0.0.0.0", port=5000)


# NOTE: We no longer auto-init at module import time. gunicorn will import
# `app.server:app` and on first request _ensure_initialized() will set up
# model_manager + data. This keeps memory footprint low until actually needed.

if __name__ == "__main__":
    init_application()
    app.run(debug=True, host="0.0.0.0", port=5000)
