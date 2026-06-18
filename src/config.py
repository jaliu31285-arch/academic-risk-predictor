"""Project configuration module.

Contains directory paths, target columns, model names, and constant settings
used across the ML pipeline modules.
"""

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'College_Students_Academic_Performance_Dataset.csv')

MODEL_DIR = os.path.join(BASE_DIR, 'models')

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
CONFUSION_MATRIX_DIR = os.path.join(OUTPUT_DIR, 'confusion_matrix')
SHAP_SUMMARY_DIR = os.path.join(OUTPUT_DIR, 'shap_summary')
MODEL_COMPARISON_DIR = os.path.join(OUTPUT_DIR, 'model_comparison')
FEATURE_IMPORTANCE_DIR = os.path.join(OUTPUT_DIR, 'feature_importance')

TARGET_CLASS = 'Academic_Risk_Class'
TARGET_REG = 'Final_Academic_Score'

EXCLUDE_COLS = [
    'Student_ID',
    'Academic_Risk_Level',
    'Intervention_Recommendation',
    'Performance_Score',
    'Final_Academic_Score',
]

MODEL_NAMES = ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'voting']

MODEL_LABELS = {
    'random_forest': '随机森林',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'catboost': 'CatBoost',
    'voting': '软投票',
}

RISK_LEVEL_MAP = {0: '低风险', 1: '中风险', 2: '高风险'}

TEST_SIZE = 0.2
RANDOM_STATE = 42


def _ensure_dirs():
    for directory in [
        DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        CONFUSION_MATRIX_DIR,
        SHAP_SUMMARY_DIR,
        MODEL_COMPARISON_DIR,
        FEATURE_IMPORTANCE_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)


_ensure_dirs()
