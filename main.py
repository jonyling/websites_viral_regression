import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_preparation import DataPreparation, FeatureConfig
from src.model_training import ModelTraining


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Mashable shares regression model from the EDA notebook."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data") / "mini_project_1_data.csv",
        help="Path to the Mashable CSV file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("best_lgb_model.pkl"),
        help="Where to save the final trained model.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use LightGBM GPU training, matching the optional GPU cells in the EDA.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV for LightGBM instead of using notebook parameters.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of RandomizedSearchCV iterations when --tune is used.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Cross-validation folds when --tune is used.",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="Also train the optional stacking ensemble from the notebook.",
    )
    parser.add_argument(
        "--no-poly",
        action="store_true",
        help="Disable the optional polynomial interaction features.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_path = args.data_path if args.data_path.is_absolute() else project_dir / args.data_path
    model_path = args.model_path if args.model_path.is_absolute() else project_dir / args.model_path

    LOGGER.info("Loading data from %s.", data_path)
    df = pd.read_csv(data_path)

    data_prep = DataPreparation(
        FeatureConfig(use_polynomial_features=not args.no_poly)
    )
    cleaned_df = data_prep.clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep.split_data(cleaned_df)
    preprocessor = data_prep.create_preprocessor()

    trainer = ModelTraining(preprocessor=preprocessor, use_gpu=args.use_gpu)
    baseline_models, baseline_metrics = trainer.train_baselines(
        X_train, y_train, X_val, y_val
    )

    if args.tune:
        lgb_model, lgb_metrics = trainer.tune_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            n_iter=args.n_iter,
            cv=args.cv,
        )
    else:
        lgb_model = trainer.build_notebook_lightgbm()
        LOGGER.info("Training notebook-parameter LightGBM.")
        lgb_model.fit(X_train, y_train)
        lgb_metrics = trainer.evaluate_log_scale(lgb_model, X_val, y_val)

    candidates = {**baseline_models, "notebook_lightgbm": lgb_model}

    if args.stack:
        LOGGER.info("Training optional stacking ensemble.")
        stack = trainer.build_stacking_model(lgb_model)
        stack.fit(X_train, y_train)
        candidates["stacking"] = stack # type: ignore

    best_name, best_model, validation_metrics = trainer.select_best_model(
        candidates, X_val, y_val # type: ignore
    )
    test_log_metrics = trainer.evaluate_log_scale(best_model, X_test, y_test)
    test_original_metrics = trainer.evaluate_original_scale(best_model, X_test, y_test)
    trainer.save_model(best_model, model_path)

    print("\nValidation metrics:")
    for name, metrics in validation_metrics.items():
        print(f"{name}: {metrics}")

    print(f"\nBest model: {best_name}")
    print(f"Best LightGBM validation metrics: {lgb_metrics}")
    print(f"Test metrics (log shares): {test_log_metrics}")
    print(f"Test metrics (original shares): {test_original_metrics}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
