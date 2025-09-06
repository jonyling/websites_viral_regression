# Standard library imports
import logging

# Third-party imports
import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings
from pathlib import Path 

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO)

@ignore_warnings(category=Warning)
def main():

    # Configuration file path
    config_path = "./src/config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
 
    # Load CSV file into a DataFrame
    main_dir = Path(__file__).parent
    data_path = main_dir / "data/mini_project_1_data.csv"
    try:
        df = pd.read_csv(data_path)
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"Datafile cannot be loaded: {str(e)}")
        
    # Initialize and run data preparation
    data_prep = DataPreparation(config, df)
    cleaned_df = data_prep.clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep.split_data(cleaned_df)
    X = cleaned_df.drop(columns=config["dropped_features"])
    y = df[config["target_column"]]
    preprocessor = data_prep.create_preprocessor(cleaned_df)
    
    # Initialize model training with the created preprocessor
    model_training = ModelTraining(config, preprocessor)

    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_for_baseline_models(
            X_train, y_train, X_val, y_val
        )
    )

    # Train and evaluate tuned models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_for_tuned_models(
        X_train, y_train, X_val, y_val
    )

    # Combine all models and their metrics into dictionaries
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Find the best model based on R² score
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["r2"])
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    # Evaluate the best model on the test set
    final_metrics = model_training.evaluate_final_model(
        best_model, X_test, y_test, X, y, X_train, y_train, best_model_name
    )

    # Print or use results
    print("Final Test Metrics:", final_metrics)

if __name__ == "__main__":
    main()

