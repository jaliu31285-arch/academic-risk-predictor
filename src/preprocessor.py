"""Data preprocessing module.

Loads the raw CSV dataset, handles missing values, encodes categorical
columns, and produces train/test splits ready for model training.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config


class DataPreprocessor:
    """Encapsulates all data preparation steps."""

    def __init__(self, data_path=None):
        if data_path is None:
            data_path = config.DATA_FILE
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_cols = None
        self._encoders = {}

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError('Data file not found: ' + self.data_path)
        self.data = pd.read_csv(self.data_path)
        print('Loaded dataset: {} samples, {} features'.format(
            self.data.shape[0], self.data.shape[1]))
        return self.data

    def handle_missing(self):
        if self.data is None:
            self.load_data()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.data[col].isnull().any():
                self.data[col] = self.data[col].fillna(self.data[col].median())
        object_cols = self.data.select_dtypes(include=['object']).columns
        for col in object_cols:
            if self.data[col].isnull().any():
                mode_value = self.data[col].mode().iloc[0]
                self.data[col] = self.data[col].fillna(mode_value)
        print('Missing values handled.')
        return self.data

    def encode_categorical(self):
        if self.data is None:
            self.handle_missing()
        target_cols = {config.TARGET_CLASS, config.TARGET_REG}
        # Use select_dtypes to catch all non-numeric columns regardless of
        # whether they are "object" dtype (pandas legacy) or "string" dtype
        # (newer pandas / pyarrow-backed data).
        cat_cols = [c for c in self.data.select_dtypes(exclude=[np.number]).columns
                    if c not in target_cols]
        encoded_any = False
        for col in cat_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self._encoders[col] = le
            encoded_any = True
        if encoded_any:
            print('Encoded {} categorical column(s): {}'.format(
                len(cat_cols), ', '.join(cat_cols)))
        else:
            print('No categorical columns to encode.')
        return self.data

    def split_features_target(self):
        if self.data is None:
            self.encode_categorical()
        exclude = list(config.EXCLUDE_COLS)
        has_reg_target = (
            config.TARGET_REG in self.data.columns
            and config.TARGET_REG != config.TARGET_CLASS
        )
        if has_reg_target:
            exclude.append(config.TARGET_REG)
        feature_cols = [c for c in self.data.columns if c not in exclude]
        if config.TARGET_CLASS in feature_cols:
            feature_cols.remove(config.TARGET_CLASS)
        self.feature_cols = feature_cols
        self.X = self.data[feature_cols].copy()
        self.y = self.data[config.TARGET_CLASS].copy()
        if has_reg_target:
            self.y_reg = self.data[config.TARGET_REG].copy()
        else:
            # Fallback: use class label as the regression target if no
            # dedicated numeric target exists in the dataset.
            self.y_reg = self.data[config.TARGET_CLASS].copy()
        print('Feature matrix shape: {}, targets: class={}, reg={}'.format(
            self.X.shape, self.y.name, self.y_reg.name))
        return self.X, self.y, self.y_reg, feature_cols

    def preprocess_all(self):
        self.load_data()
        self.handle_missing()
        self.encode_categorical()
        X, y_cls, y_reg, feature_cols = self.split_features_target()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cls,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y_cls,
        )
        _, _, y_train_reg, y_test_reg = train_test_split(
            X, y_reg,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )
        print('Train size: {}, Test size: {}'.format(
            X_train.shape[0], X_test.shape[0]))
        return X_train, X_test, y_train, y_test, y_train_reg, y_test_reg, feature_cols

    def export_encoder_map(self):
        """Return a dict mapping categorical column names to class label
        lists, suitable for JSON serialization. This map is consumed by the
        Flask server to translate user input (e.g. "Male") into the numeric
        codes used during training (e.g. 0)."""
        mapping = {}
        for col, le in self._encoders.items():
            try:
                mapping[col] = [str(x) for x in list(le.classes_)]
            except Exception:
                continue
        return mapping
