"""Model training module.

Builds and trains multiple classifier models including Random Forest,
XGBoost, LightGBM, CatBoost, and a soft-voting ensemble. Models are
persisted to disk via joblib.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from src import config


class ModelTrainer:
    """Builds, trains, and saves classification and regression models."""

    def __init__(self):
        self.models = {}           # classifiers
        self.reg_models = {}       # regressors
        self.train_times = {}
        self.reg_train_times = {}

    def build_model(self, name):
        name = name.lower()
        if name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        if name == 'xgboost':
            if XGBClassifier is None:
                raise ImportError('xgboost is not installed')
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0,
            )
        if name == 'lightgbm':
            if LGBMClassifier is None:
                raise ImportError('lightgbm is not installed')
            return LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        if name == 'catboost':
            if CatBoostClassifier is None:
                raise ImportError('catboost is not installed')
            return CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                verbose=False,
            )
        if name == 'voting':
            base_names = [n for n in config.MODEL_NAMES if n != 'voting']
            estimators = []
            for base in base_names:
                if base in self.models:
                    estimators.append((base, self.models[base]))
            if not estimators:
                raise RuntimeError(
                    'Base models must be trained before building voting model')
            return VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1,
            )
        raise ValueError('Unknown model name: ' + name)

    def build_regressor(self, name):
        """Build a regression model for predicting numeric score."""
        name = name.lower()
        if name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        if name == 'xgboost':
            if XGBRegressor is None:
                raise ImportError('xgboost is not installed')
            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        if name == 'lightgbm':
            if LGBMRegressor is None:
                raise ImportError('lightgbm is not installed')
            return LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        if name == 'catboost':
            if CatBoostRegressor is None:
                raise ImportError('catboost is not installed')
            return CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                verbose=False,
            )
        if name == 'voting':
            # For regression we use a simple Ridge regression as the "voting"
            # meta-learner since sklearn's VotingRegressor requires base models.
            return Ridge(alpha=1.0, random_state=config.RANDOM_STATE)
        raise ValueError('Unknown model name: ' + name)

    def train_all(self, X_train, y_train):
        base_names = [n for n in config.MODEL_NAMES if n != 'voting']
        for name in base_names:
            print('Training {} (classification) ...'.format(name))
            start = time.time()
            model = self.build_model(name)
            model.fit(X_train, y_train)
            elapsed = time.time() - start
            self.models[name] = model
            self.train_times[name] = elapsed
            print('  {} done in {:.2f}s'.format(name, elapsed))
        if 'voting' in config.MODEL_NAMES and 'voting' not in self.models:
            print('Training voting ensemble (classification) ...')
            start = time.time()
            voting_model = self.build_model('voting')
            voting_model.fit(X_train, y_train)
            elapsed = time.time() - start
            self.models['voting'] = voting_model
            self.train_times['voting'] = elapsed
            print('  voting done in {:.2f}s'.format(elapsed))
        return self.models

    def train_all_regression(self, X_train, y_train_reg):
        """Train regression models to predict numeric score."""
        base_names = [n for n in config.MODEL_NAMES if n != 'voting']
        for name in base_names:
            print('Training {} (regression) ...'.format(name))
            start = time.time()
            try:
                model = self.build_regressor(name)
                model.fit(X_train, y_train_reg)
                elapsed = time.time() - start
                self.reg_models[name] = model
                self.reg_train_times[name] = elapsed
                print('  {} done in {:.2f}s'.format(name, elapsed))
            except Exception as e:
                print('  {} failed: {}'.format(name, str(e)))
        # For regression "voting" fallback, use best trained regressor
        # or a Ridge baseline if nothing else is available.
        if 'voting' in config.MODEL_NAMES:
            if self.reg_models:
                # Use the first available regressor as placeholder for "voting"
                first_name = list(self.reg_models.keys())[0]
                self.reg_models['voting'] = self.reg_models[first_name]
                self.reg_train_times['voting'] = self.reg_train_times.get(
                    first_name, 0.0)
                print('  voting (regression) -> using {} as fallback'.format(
                    first_name))
            else:
                # Fallback: Ridge regression on raw features
                try:
                    ridge = Ridge(alpha=1.0, random_state=config.RANDOM_STATE)
                    ridge.fit(X_train, y_train_reg)
                    self.reg_models['voting'] = ridge
                    print('  voting (regression) -> Ridge fallback')
                except Exception as e:
                    print('  voting regression failed: {}'.format(str(e)))
        return self.reg_models

    def save_models(self):
        if not self.models:
            raise RuntimeError('No models have been trained yet')
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        saved = []
        for name, model in self.models.items():
            path = os.path.join(config.MODEL_DIR, name + '.pkl')
            joblib.dump(model, path)
            saved.append(path)
        print('Saved {} classification models to {}'.format(
            len(saved), config.MODEL_DIR))
        if self.reg_models:
            saved_reg = []
            for name, model in self.reg_models.items():
                path = os.path.join(
                    config.MODEL_DIR, name + '_reg.pkl')
                joblib.dump(model, path)
                saved_reg.append(path)
            print('Saved {} regression models to {}'.format(
                len(saved_reg), config.MODEL_DIR))
        return saved

    def evaluate_regression(self, X_test, y_test):
        """Compute standard regression metrics for every trained regressor.

        Returns a dict mapping model name to metrics dict with keys:
        mse (mean squared error), mae (mean absolute error),
        rmse (root mean squared error), r2 (R^2 coefficient of
        determination)."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        results = {}
        if not self.reg_models:
            print('  No regression models available - skipping evaluation')
            return results
        for name, model in self.reg_models.items():
            try:
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                rmse = float(np.sqrt(mse))
                r2 = r2_score(y_test, preds)
                results[name] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': rmse,
                    'r2': float(r2),
                }
                print('  {} regression -> MSE={:.4f}, MAE={:.4f}, R^2={:.4f}'.format(
                    name, mse, mae, r2))
            except Exception as exc:
                print('  {} regression evaluation failed: {}'.format(name, exc))
        return results

    def get_feature_importance(self, name, feature_cols, top_n=15):
        if name not in self.models:
            raise ValueError('Model not trained yet: ' + name)
        model = self.models[name]
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif name == 'voting' and hasattr(model, 'estimators_'):
            values = []
            for est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    values.append(est.feature_importances_)
            if values:
                importance = np.mean(np.array(values), axis=0)
        if importance is None:
            return pd.DataFrame(
                columns=['feature', 'importance'])
        df = pd.DataFrame({
            'feature': list(feature_cols),
            'importance': list(importance),
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        return df.head(top_n)

