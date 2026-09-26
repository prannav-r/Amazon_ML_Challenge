"""
Supervised Matching Model Module (Phase 5)
Amazon ML Challenge 2026: Business Entity Resolution

Provides model wrappers and training pipelines for:
- Model A: Regularized Logistic Regression (interpretable linear baseline with imputer/scaler)
- Model B: Gradient-Boosted Trees (XGBoost 3.2.0 with conservative tabular hyperparameters)
- Model C: Gradient-Boosted Trees with class-weighting (scale_pos_weight) or Random Forest
- Optional probability calibration via isotonic / Platt sigmoid scaling.
"""

import time
import os
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.feature_schema import FEATURE_NAMES, get_feature_names


class EntityMatcherModel:
    """
    Standardized supervised model interface for pairwise business entity resolution.
    
    Guarantees:
    - Consistent feature ordering using canonical 65-feature schema.
    - Deterministic random seeds.
    - Profiling of training and inference runtimes.
    - Native support for feature importance extraction.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self.params = params or {}
        self.model = None
        self.pipeline = None
        self.is_fitted = False
        self.training_time_sec = 0.0
        self.inference_time_sec = 0.0

        self._build_model()

    def _build_model(self):
        """Initializes the underlying estimator based on model_type."""
        if self.model_type == "logistic_regression":
            lr_params = {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 500,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            lr_params.update(self.params)
            self.model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**lr_params)),
            ])

        elif self.model_type in ("xgboost", "xgboost_weighted"):
            # Conservative tabular hyperparameters
            xgb_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "eval_metric": "logloss",
                "random_state": self.random_state,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            if self.model_type == "xgboost_weighted":
                # scale_pos_weight will be set or passed in params
                xgb_params["scale_pos_weight"] = 5.2  # approximate sqrt(27)
            xgb_params.update(self.params)
            self.model = XGBClassifier(**xgb_params)

        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            rf_params = {
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            rf_params.update(self.params)
            self.model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(**rf_params)),
            ])
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Choose 'logistic_regression', 'xgboost', 'xgboost_weighted', or 'random_forest'.")

    def _prepare_matrix(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Ensures input matrix has exactly the schema feature columns in order."""
        if isinstance(X, pd.DataFrame):
            # Select schema features in exact order
            cols = [c for c in self.feature_names if c in X.columns]
            if len(cols) != len(self.feature_names):
                missing = set(self.feature_names) - set(X.columns)
                raise ValueError(f"Input DataFrame missing required features: {missing}")
            return X[self.feature_names].values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
            return X.astype(np.float32)
        else:
            raise TypeError(f"Unsupported type for X: {type(X)}")

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: Optional[int] = 30,
        verbose: bool = False,
    ) -> "EntityMatcherModel":
        """Fits the model on training data with optional validation evaluation."""
        X_tr = self._prepare_matrix(X_train)
        y_tr = np.asarray(y_train, dtype=np.int32)

        t0 = time.time()

        if self.model_type in ("xgboost", "xgboost_weighted") and X_val is not None and y_val is not None:
            X_v = self._prepare_matrix(X_val)
            y_v = np.asarray(y_val, dtype=np.int32)

            if early_stopping_rounds is not None and early_stopping_rounds > 0:
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)

            self.model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_v, y_v)],
                verbose=verbose,
            )
        else:
            self.model.fit(X_tr, y_tr)

        self.training_time_sec = time.time() - t0
        self.is_fitted = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts positive class match probabilities P(y=1)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_mat = self._prepare_matrix(X)

        t0 = time.time()
        probs = self.model.predict_proba(X_mat)[:, 1]
        self.inference_time_sec = time.time() - t0
        return probs

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Predicts binary match labels using given threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int32)

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns DataFrame of feature names and importances sorted descending."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if self.model_type in ("xgboost", "xgboost_weighted"):
            importances = self.model.feature_importances_
            metric_name = "gain_importance"
        elif self.model_type == "logistic_regression":
            coefs = self.model.named_steps["classifier"].coef_[0]
            importances = np.abs(coefs)
            metric_name = "abs_coefficient"
        elif self.model_type == "random_forest":
            importances = self.model.named_steps["classifier"].feature_importances_
            metric_name = "gini_importance"
        else:
            importances = np.zeros(len(self.feature_names))
            metric_name = "importance"

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            metric_name: importances,
        }).sort_values(by=metric_name, ascending=False).reset_index(drop=True)
        return df_imp

    def calibrate(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        method: str = "sigmoid",
    ) -> "EntityMatcherModel":
        """
        Calibrates model output probabilities using held-out validation set.
        Method: 'sigmoid' (Platt scaling) or 'isotonic'.
        """
        if not self.is_fitted:
            raise RuntimeError("Fit model before calibrating.")

        X_v = self._prepare_matrix(X_val)
        y_v = np.asarray(y_val, dtype=np.int32)

        calibrated = CalibratedClassifierCV(estimator=self.model, method=method, cv="prefit")
        calibrated.fit(X_v, y_v)
        self.model = calibrated
        return self

    def save(self, filepath: str):
        """Saves model artifact to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "params": self.params,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "training_time_sec": self.training_time_sec,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, filepath: str) -> "EntityMatcherModel":
        """Loads model artifact from disk."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            model_type=payload["model_type"],
            params=payload["params"],
            random_state=payload["random_state"],
            feature_names=payload["feature_names"],
        )
        instance.model = payload["model"]
        instance.is_fitted = payload["is_fitted"]
        instance.training_time_sec = payload["training_time_sec"]
        return instance
