import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

class ModelTraining:
    """
    A class used to train and evaluate machine learning models to predict the number of 'shares' of articles on mashable.com.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocessor

    def train_and_evaluate_for_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")

        pipelines = {}
        metrics = {}

        model_name = ["ridge", "lasso", "stacking"]

        # Model-1 -> Ridge Regression
        ridge_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor), 
            ('regressor', Ridge(alpha=1))
        ])
        ridge_pipeline.fit(X_train, y_train)
        metrics["ridge"] = self._evaluate_model(ridge_pipeline, X_val, y_val, model_name="ridge")
                
        # Model-2 -> Lasso Regression
        lasso_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor), 
            ('regressor', Ridge(alpha=1))
        ])
        lasso_pipeline.fit(X_train, y_train)
        metrics["lasso"] = self._evaluate_model(lasso_pipeline, X_val, y_val, model_name="lasso")

        # Model-3 -> Stacking Regression
        # Define base models for StackingRegressor
        lasso_pipeline1 = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', Lasso(random_state=42))])
        rf_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', RandomForestRegressor(random_state=42))])
        lgb_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', lgb.LGBMRegressor(random_state=42))])

        base_models = [
            ('lasso', lasso_pipeline1),
            ('rf', rf_pipeline),
            ('lgb', lgb_pipeline)
        ]

        meta_model = Ridge()

        # Create stacking regressor
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3,  # Number of cross-validation folds for base predictions
            n_jobs=-1,  # Parallelize fitting of base models
            passthrough=False  # Do not pass original features to meta-model
        )

        # Fit the stacking regressor
        stacking_regressor.fit(X_train, y_train)

        metrics["stacking"] = self._evaluate_model(stacking_regressor, X_val, y_val, model_name="stacking")
                        
        return pipelines, metrics
    
    def train_and_evaluate_for_tuned_models(
    self,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning for Ridge, Lasso, and Stacking models and evaluate them.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}

        # Define parameter grids for each model
        ridge_lasso_param_grid = self.config["RandomizedSearch_param"]["param_grid"]  # From your config.yaml
        stacking_param_grid = {
            'final_estimator__alpha': [0.01, 0.1, 1.0, 10.0],  # Ridge meta-model
            'lasso__regressor__alpha': [0.01, 0.1, 1.0],      # Lasso base model
            'rf__regressor__max_depth': [3, 5, 10],           # Random Forest base model
            'lgb__regressor__learning_rate': [0.01, 0.1, 0.2],  # LightGBM base model
            'xgb__regressor__max_depth': [3, 5, 10],  # Added for XGBoost
            'cat__regressor__depth': [4, 6, 8]  # Added for CatBoost
        }

        # Define base models for StackingRegressor
        lasso_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', Lasso(random_state=42))])
        rf_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', RandomForestRegressor(random_state=42))])
        lgb_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', lgb.LGBMRegressor(random_state=42))])
        xgb_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', XGBRegressor(random_state=42))])  # Added XGBoost
        cat_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', CatBoostRegressor(random_state=42, verbose=0))])  # Added CatBoost
        
        base_models = [
            ('lasso', lasso_pipeline),
            ('rf', rf_pipeline),
            ('lgb', lgb_pipeline), 
            ('xgb', xgb_pipeline),  # Added for diversity
            ('cat', cat_pipeline)   # Added for diversity
        ]

        # Define models
        models = {
            "ridge": Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', Ridge())]),
            "lasso": Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', Lasso(random_state=42))]),
            "stacking": StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(),
                cv=3,
                n_jobs=-1
            )
        }

        # Hyperparameter tuning settings
        cv = self.config["RandomizedSearch_param"]["cv"]  # e.g., 3
        scoring = self.config["RandomizedSearch_param"]["scoring"]  # e.g., 'r2'
        n_jobs = self.config["RandomizedSearch_param"]["n_jobs"]  # e.g., -1
        n_iter = self.config["RandomizedSearch_param"]["n_iter"]  # Reduced to 5for faster runtime

        for model_name, model in models.items():
            # Use appropriate param_grid
            param_grid = ridge_lasso_param_grid if model_name in ["ridge", "lasso"] else stacking_param_grid

            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=42,
                verbose=2
            )
            search.fit(X_train, y_train)
            tuned_models[model_name] = search.best_estimator_
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name + " (tuned)"
            )
        
        # Weighted VotingRegressor based on baseline R²
        # Assume baseline R² from previous evaluations (replace with actual values)
        baseline_r2 = {'lasso': 0.342, 'rf': 0.3, 'lgb': 0.35, 'xgb': 0.34, 'cat': 0.33}  # Example; calculate from baselines
        weights = [baseline_r2[name.split('__')[0]] for name, _ in base_models]  # Weights based on baseline R²
        voting = VotingRegressor(estimators=base_models, weights=weights, n_jobs=-1)
        voting.fit(X_train, y_train)
        tuned_models["weighted_voting"] = voting
        tuned_metrics["weighted_voting"] = self._evaluate_model(voting, X_val, y_val, "weighted_voting")

        logging.info("Hyperparameter tuning completed.")
        return tuned_models, tuned_metrics

    def _evaluate_model(self, model, X_val, y_val, model_name: str) -> Dict[str, float]:
        """
        Evaluate a model on validation data and return metrics.

        Args:
            model: Trained model or pipeline.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.
            model_name (str): Name of the model for logging.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics (R², MAE, MSE, RMSE).
        """
        y_pred = model.predict(X_val)
        metrics = {
            'r2': r2_score(y_val, y_pred),
            'mae': mean_absolute_error(y_val, y_pred),
            'mse': mean_squared_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred))
        }
        logging.info(f"Metrics for {model_name}: {metrics}")
        return metrics

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, X: pd.DataFrame, y: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_pred = model.predict(X_test)
        # Define custom scorers for RMSE and MAE (negative for cross_val_score to maximize)
        def rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))

        rmse_scorer = make_scorer(rmse, greater_is_better=False)  # Negative RMSE for minimization
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)  # Negative MAE
        r2_scorer = make_scorer(r2_score)

        # Example: Your model pipeline (replace with your StackingRegressor or other model)
        # model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', StackingRegressor(...))])

        # Perform cross-validation with multiple metrics
        cv_folds = 5
        scores_rmse = cross_val_score(model, X, y, cv=cv_folds, scoring=rmse_scorer)
        scores_mae = cross_val_score(model, X, y, cv=cv_folds, scoring=mae_scorer)
        scores_r2 = cross_val_score(model, X, y, cv=cv_folds, scoring=r2_scorer)

        # Print mean and std for robust validation
        print("Robust Cross-Validation Results:")
        print(f"RMSE: Mean = {-np.mean(scores_rmse):.4f}, Std = {np.std(scores_rmse):.4f}")
        print(f"MAE: Mean = {-np.mean(scores_mae):.4f}, Std = {np.std(scores_mae):.4f}")
        print(f"R²: Mean = {np.mean(scores_r2):.4f}, Std = {np.std(scores_r2):.4f}")

        # Optional: Fit on train and evaluate on test with all metrics
        model.fit(X_train, y_train)
        
        print("\nTest Metrics (Prioritizing RMSE/MAE):")
        print(f"RMSE: {rmse(y_test, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"R²: {r2_score(y_test, y_pred):.4f}")

        y_test_pred = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": root_mean_squared_error(y_test, y_test_pred),
            "r2": r2_score(y_test, y_test_pred),
        }
        logging.info(f"Final Test Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics

        
