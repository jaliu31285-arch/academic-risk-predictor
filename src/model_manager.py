"""Model manager module.

Loads persisted models from disk, exposes predict for a named model, and
lists which models are currently available.
"""

import os

import joblib
import numpy as np

from src import config


class ModelManager:
    """Loads and manages previously trained classification and regression
    models from disk.

    Classification models are saved as ``<name>.pkl``.
    Regression models are saved as ``<name>_reg.pkl``.
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = config.MODEL_DIR
        self.model_dir = model_dir
        self.models = {}           # classifiers
        self.reg_models = {}       # regressors

    def load_all(self):
        if not os.path.isdir(self.model_dir):
            print('Model directory does not exist: ' + self.model_dir)
            return self.models
        # Load classifiers
        for name in config.MODEL_NAMES:
            path = os.path.join(self.model_dir, name + '.pkl')
            if os.path.isfile(path):
                self.models[name] = joblib.load(path)
        # Load regressors (if present)
        for name in config.MODEL_NAMES:
            path = os.path.join(self.model_dir, name + '_reg.pkl')
            if os.path.isfile(path):
                self.reg_models[name] = joblib.load(path)
        print('Loaded {} classification models, {} regression models'.format(
            len(self.models), len(self.reg_models)))
        return self.models

    def predict(self, name, X):
        if not self.models:
            self.load_all()
        if name not in self.models:
            raise ValueError('Model not available: ' + name)
        return self.models[name].predict(X)

    def predict_proba(self, name, X):
        """Return class probabilities for a classification model."""
        if not self.models:
            self.load_all()
        if name not in self.models:
            raise ValueError('Model not available: ' + name)
        model = self.models[name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        # Fallback: one-hot style probabilities from predicted class.
        classes = getattr(model, 'classes_', [0, 1, 2])
        preds = np.asarray(model.predict(X)).ravel()
        result = np.zeros((len(preds), len(classes)), dtype=float)
        for i, cls in enumerate(classes):
            result[preds == cls, i] = 1.0
        return result

    def predict_score(self, name, X):
        """Return a numeric score prediction from the regression model."""
        if not self.reg_models:
            # Try lazy-load from disk
            for mname in config.MODEL_NAMES:
                path = os.path.join(self.model_dir, mname + '_reg.pkl')
                if os.path.isfile(path):
                    self.reg_models[mname] = joblib.load(path)
        if name not in self.reg_models:
            # Fallback to the same-name classifier's prediction (not ideal,
            # but prevents crashes when regression models are missing).
            raise ValueError('Regression model not available: ' + name)
        result = self.reg_models[name].predict(X)
        return np.asarray(result, dtype=float).ravel()

    def list_available(self):
        available = []
        if os.path.isdir(self.model_dir):
            for name in config.MODEL_NAMES:
                if os.path.isfile(
                        os.path.join(self.model_dir, name + '.pkl')):
                    available.append(name)
        return available

    def list_available_regression(self):
        available = []
        if os.path.isdir(self.model_dir):
            for name in config.MODEL_NAMES:
                if os.path.isfile(
                        os.path.join(self.model_dir, name + '_reg.pkl')):
                    available.append(name)
        return available

