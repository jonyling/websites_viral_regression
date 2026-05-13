import logging
from pathlib import Path
from typing import Dict

import joblib
import lightgbm as lgb # type: ignore
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


LOGGER = logging.getLogger(__name__)


class ModelTraining:
    """Train and evaluate the regression models developed in the EDA notebook."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        self.preprocessor = preprocessor
        self.random_state = random_state
        self.use_gpu = use_gpu

    def build_baseline_models(self) -> Dict[str, Pipeline]:
        lgb_params = self._lgb_params(n_estimators=100, learning_rate=0.1, max_depth=5)
        return {
            "linear_regression": Pipeline(
                [
                    ("preprocessor", clone(self.preprocessor)),
                    ("regressor", LinearRegression()),
                ]
            ),
            "ridge": Pipeline(
                [
                    ("preprocessor", clone(self.preprocessor)),
                    ("regressor", Ridge(alpha=1.0)),
                ]
            ),
            "xgboost": Pipeline(
                [
                    ("preprocessor", clone(self.preprocessor)),
                    (
                        "regressor",
                        xgb.XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "lightgbm": Pipeline(
                [
                    ("preprocessor", clone(self.preprocessor)),
                    ("regressor", lgb.LGBMRegressor(**lgb_params)),
                ]
            ),
            "random_forest": Pipeline(
                [
                    ("preprocessor", clone(self.preprocessor)),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=100,
                            max_depth=5,
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        }

    def train_baselines(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        models = self.build_baseline_models()
        metrics = {}
        for name, model in models.items():
            LOGGER.info("Training %s.", name)
            model.fit(X_train, y_train)
            metrics[name] = self.evaluate_log_scale(model, X_val, y_val)
            LOGGER.info("%s validation metrics: %s", name, metrics[name])
        return models, metrics

    def tune_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_iter: int = 30,
        cv: int = 5,
    ) -> tuple[Pipeline, Dict[str, float]]:
        """Tune LightGBM with the notebook's RandomizedSearchCV search space."""
        pipeline = Pipeline(
            [
                ("preprocessor", clone(self.preprocessor)),
                ("regressor", lgb.LGBMRegressor(**self._lgb_params())),
            ]
        )
        param_dist = {
            "regressor__n_estimators": [200, 300, 400, 500, 600, 700],
            "regressor__learning_rate": [0.01, 0.0166, 0.03, 0.05, 0.08, 0.1],
            "regressor__max_depth": [5, 7, 8, 9, 11, -1],
            "regressor__num_leaves": [31, 63, 127, 223, 255],
            "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
            "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "regressor__reg_alpha": [0, 0.1, 1, 1.4463, 5],
            "regressor__reg_lambda": [0, 0.1, 1, 1.7608, 5],
            "regressor__min_child_samples": [5, 10, 11, 20, 30],
        }
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )
        LOGGER.info("Starting LightGBM tuning: n_iter=%s, cv=%s.", n_iter, cv)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        metrics = self.evaluate_log_scale(best_model, X_val, y_val)
        LOGGER.info("Best LightGBM params: %s", search.best_params_)
        LOGGER.info("Tuned LightGBM validation metrics: %s", metrics)
        return best_model, metrics # type: ignore

    def build_notebook_lightgbm(self) -> Pipeline:
        """Build the final Optuna-style LightGBM from the notebook's last cells."""
        return Pipeline(
            [
                ("preprocessor", clone(self.preprocessor)),
                (
                    "regressor",
                    lgb.LGBMRegressor(
                        **self._lgb_params(
                            n_estimators=700,
                            learning_rate=0.016600250862150365,
                            max_depth=8,
                            num_leaves=223,
                            subsample=0.7049040912171848,
                            colsample_bytree=0.7088913513342914,
                            reg_alpha=1.4463103990967312,
                            reg_lambda=1.7607633224319064,
                            min_child_samples=11,
                        )
                    ),
                ),
            ]
        )

    def build_stacking_model(self, best_lgb: Pipeline) -> StackingRegressor:
        estimators = [
            ("lgb", best_lgb),
            (
                "xgb",
                Pipeline(
                    [
                        ("preprocessor", clone(self.preprocessor)),
                        (
                            "regressor",
                            xgb.XGBRegressor(
                                objective="reg:squarederror",
                                n_estimators=300,
                                learning_rate=0.05,
                                max_depth=7,
                                random_state=self.random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "ridge",
                Pipeline(
                    [
                        ("preprocessor", clone(self.preprocessor)),
                        ("regressor", Ridge(alpha=1.0)),
                    ]
                ),
            ),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=0.5),
            cv=5,
            n_jobs=-1,
        )

    def select_best_model(
        self,
        candidates: Dict[str, object],
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[str, object, Dict[str, Dict[str, float]]]:
        metrics = {
            name: self.evaluate_log_scale(model, X_val, y_val)
            for name, model in candidates.items()
        }
        best_name = max(metrics, key=lambda name: metrics[name]["r2"])
        return best_name, candidates[best_name], metrics

    def evaluate_log_scale(
        self, model: object, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        y_pred = model.predict(X) # type: ignore
        return {
            "mae": mean_absolute_error(y, y_pred),
            "mse": mean_squared_error(y, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": r2_score(y, y_pred),
        }

    def evaluate_original_scale(
        self, model: object, X: pd.DataFrame, y_log: pd.Series
    ) -> Dict[str, float]:
        y_pred_log = model.predict(X) # type: ignore
        y_pred = np.clip(np.expm1(y_pred_log), 0, None)
        y_true = np.expm1(y_log)
        absolute_percentage_error = (
            np.abs(y_true - y_pred) / np.maximum(y_true, 1e-8)
        ) * 100
        return {
            "mae_shares": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "median_absolute_percentage_error": float(np.median(absolute_percentage_error)),
            "r2_log": r2_score(y_log, y_pred_log),
        }

    @staticmethod
    def save_model(model: object, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

    def _lgb_params(self, **overrides) -> dict:
        params = {
            "objective": "regression",
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        if self.use_gpu:
            params["device"] = "gpu"
        params.update(overrides)
        return params
