# Mashable Shares Regression Mini Project

This project predicts the number of social shares for Mashable articles. The production pipeline follows the final direction of `eda.ipynb`: clean missing values, log-transform the highly skewed target, engineer the strongest EDA features, then train and evaluate regression models with LightGBM as the main final model.

## Repository Structure

```text
jonyling-RegressionMiniProject/
├── data/
│   └── mini_project_1_data.csv
├── src/
│   ├── data_preparation.py
│   └── model_training.py
├── eda.ipynb
├── main.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
python main.py
```

The default run saves the best validation model to `best_lgb_model.pkl`.

Useful options:

```bash
python main.py --tune --n-iter 30 --cv 5
python main.py --use-gpu
python main.py --stack
python main.py --no-poly
```

`--use-gpu` enables the LightGBM GPU option used in the notebook. Leave it off on machines without a working GPU/OpenCL LightGBM setup.

## Pipeline

1. Load `data/mini_project_1_data.csv`.
2. Clean EDA-identified missing or strange values:
   - median-fill link, image, video, and self-reference share columns;
   - replace zero non-stop-word ratios with non-zero medians;
   - mode-fill missing `data_channel`;
   - replace strange `kw_min_min` values `{-1, 0, 4, 217}` with the mean of valid values.
3. Engineer notebook features:
   - `shares_logged` as the log-transformed target;
   - logged content, unique-token, comment, and keyword metrics;
   - `is_weekend`;
   - `kw_engagement_ratio_logged`;
   - `content_keyword_interaction`.
4. Split data into 80% train, 10% validation, and 10% test.
5. Preprocess with median imputation, scaling, one-hot encoding for `weekday` and `data_channel`, and optional polynomial interactions for the notebook's top numerical features.
6. Train baseline models and a notebook-parameter LightGBM model. Optional flags run LightGBM tuning and stacking.
7. Select the best validation model, evaluate it on the test set, and save it.

## Models

The script trains these baselines:

- Linear Regression
- Ridge Regression
- XGBoost Regressor
- LightGBM Regressor
- Random Forest Regressor

The main final model is LightGBM, matching the EDA's final modeling direction. The included tuned-parameter LightGBM uses the notebook's Optuna-style values. `--tune` runs a RandomizedSearchCV search over the LightGBM space explored in the notebook, and `--stack` trains the optional LightGBM/XGBoost/Ridge stacking ensemble.

## Metrics

The model is evaluated on log shares using:

- MAE
- MSE
- RMSE
- R2

For business interpretation, predictions are also back-transformed to the original shares scale and reported with:

- MAE in shares
- MAPE
- median absolute percentage error
- log-scale R2

## Notes

The original `shares` target is extremely right-skewed, so the pipeline predicts `log1p(shares)` and converts predictions back with `expm1` only for final interpretation. This keeps training more stable while still giving share-scale metrics.
