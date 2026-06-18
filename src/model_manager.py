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
        self.models = {}           # classifiers  (lazy-loaded)
        self.reg_models = {}       # regressors   (lazy-loaded)
        print('[ModelManager] lazy mode: models loaded on first use.')

    # ---------------------------------------------------------------
    # Lazy-loading helpers
    # ---------------------------------------------------------------
    def _classifier_path(self, name):
        return os.path.join(self.model_dir, name + '.pkl')

    def _regressor_path(self, name):
        return os.path.join(self.model_dir, name + '_reg.pkl')

    def get_model(self, name):
        """Return a classifier, loading it from disk on first access."""
        if name not in self.models:
            path = self._classifier_path(name)
            if not os.path.isfile(path):
                return None
            self.models[name] = joblib.load(path)
            print('[ModelManager] Loaded classifier: ' + name)
        return self.models[name]

    def get_reg_model(self, name):
        """Return a regressor, loading it from disk on first access."""
        if name not in self.reg_models:
            path = self._regressor_path(name)
            if not os.path.isfile(path):
                return None
            self.reg_models[name] = joblib.load(path)
            print('[ModelManager] Loaded regressor: ' + name)
        return self.reg_models[name]

    def unload(self, name):
        """Drop a model from memory to reclaim RAM."""
        self.models.pop(name, None)
        self.reg_models.pop(name, None)

    # ---------------------------------------------------------------
    # load_all kept for backwards compatibility - it is now a NO-OP.
    # ---------------------------------------------------------------
    def load_all(self):
        return self.models

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------
    def predict(self, name, X):
        model = self.get_model(name)
        if model is None:
            raise ValueError('Model not available: ' + name)
        return model.predict(X)

    def predict_proba(self, name, X):
        """Return class probabilities for a classification model."""
        model = self.get_model(name)
        if model is None:
            raise ValueError('Model not available: ' + name)
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        classes = getattr(model, 'classes_', [0, 1, 2])
        preds = np.asarray(model.predict(X)).ravel()
        result = np.zeros((len(preds), len(classes)), dtype=float)
        for i, cls in enumerate(classes):
            result[preds == cls, i] = 1.0
        return result

    def predict_score(self, name, X):
        """Return a numeric score prediction from the regression model."""
        reg = self.get_reg_model(name)
        if reg is None:
            raise ValueError('Regression model not available: ' + name)
        result = reg.predict(X)
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

