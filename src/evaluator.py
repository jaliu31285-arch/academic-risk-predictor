"""Model evaluation module.

Computes accuracy, weighted F1, and multi-class ROC-AUC for each trained
model on the held-out test set and prints a concise comparison report.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import config


def evaluate_all(models, X_test, y_test):
    results = {}
    classes = np.unique(y_test)
    for name in config.MODEL_NAMES:
        if name not in models:
            continue
        model = models[name]
        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average='weighted',
                            zero_division=0))
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                auc = float(roc_auc_score(
                    y_test, y_proba, multi_class='ovr', labels=classes))
            else:
                auc = float('nan')
        except Exception:
            auc = float('nan')
        results[name] = {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
        }
    _print_report(results)
    return results


def _print_report(results):
    print('=' * 55)
    print('Model Comparison on Test Set')
    print('=' * 55)
    header = '{:<20}{:>10}{:>10}{:>10}'.format('Model', 'Acc', 'F1', 'AUC')
    print(header)
    print('-' * 55)
    for name in config.MODEL_NAMES:
        if name not in results:
            continue
        r = results[name]
        label = config.MODEL_LABELS.get(name, name)
        print('{:<20}{:>10.4f}{:>10.4f}{:>10.4f}'.format(
            label, r['accuracy'], r['f1'], r['auc']))
    print('=' * 55)

