# jonyling-RegressionMiniProject
# 'shares' Prediction - ML model to predict the number of sharings of online articles

A machine learning project to predict redict the number of sharings of online articles on mashable.com, using data preprocessing and model training pipelines. Built with Python, this project leverages libraries like Pandas, Scikit-learn, and YAML for configuration.

Overview
This repository contains the deliverables for the AIAP Foundation Classification Mini Project, addressing the objectives of predicting the number of shares (popularity) on social media platforms for the news articles using the dataset provided, and also to evaluate at least 3 suitable models for predicting the number of shares. 

Repository Structure

RegressionMiniProject
├── .github/                    # GitHub Actions scripts (provided in template)
├── src/                        # Python modules for ML pipeline
│   ├── data_preparation.py     # Data loading and preprocessing
│   ├── model_training.py       # Model training and evaluation
│   └── config.yaml             # Configuration file for pipeline parameters
├── data/                       # Data folder (not uploaded, contains gas_monitoring.db)
├── eda.ipynb                   # Jupyter notebook for Task 1 (EDA)
├── requirements.txt            # Python dependencies
├── main.py                     # Main module execute the pipeline
└── README.md                   # This file

## Table of Contents
- [Pipeline Execution Instructions](##PipelineExecutionInstructions)
- [PipelineLogicalFlow](#PipelineLogicalFlow)
- [Key_EDA_Findings_and_Pipeline_Choices](#Key_EDA_Findings_and_Pipeline_Choices)
- [Model_Selection_and_Justification](#Model_Selection_and_Justification)
- [Model_Evaluation](#Model_Evaluation)
- [Configuration](#configuration)
- [MLmodel_results](#MLmodel_results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## PipelineExecutionInstructions
Setup:
- Ensure Python 3.11+ is installed.
- Place mini_project_1_data.csv in the data/ folder (relative path: data/mini_project_1_data.csv).
- Install dependencies: pip install -r requirements.txt.
- Execute the main.py: This runs data_preparation.py to clean data, split data and create the preprocessor. This is followed running model_training.py to train and evaluate for base models, then tuned models, and finally the final model.
- Modify Parameters: Edit src/config.yaml to adjust model hyperparameters, feature selections, or preprocessing steps. Example: Change regressor__alpha for RandomizedSearch. 

## PipelineLogicalFlow
- Data Loading: Load data from data/mini_project_1_data.csv.
- Data Preprocessing:
    - KNN-impute missing values: 'kw_avg_min', 'n_non_stop_unique_tokens', 'data_channel'.
    - Median-impute missing values: 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos', 'self_reference_max_shares', 'self_reference_avg_shares'. 
    - Clip into {0, 50000}: 'kw_avg_min'
- Feature engineering: 
    - Bin into 'low', 'medium', 'high': 'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg'
    - Drop features: 'ID', 'URL', 'shares'.
- Split Data:
    - Split data with 80/10/10 proportion for training, validation and test sets.
- Build Pipelines
    - Numerical features =  ['timedelta', 'n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos', 'n_comments', 
                              'average_token_length', 'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_shares', 'num_keywords']
    - Ordinal features = ['kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg']
    - Nominal features = ['data_channel', 'weekday']
- Baseline Models Training: 
    - Train three models: Ridge, Lasso, Stacking (using Lasso, RandomForest, LightGBM).
- Tuned Models Training: 
    - Train five models: Ridge, Lasso and Stacking (using Lasso, RandomForest, LightGBM, XGB and CatBoost) and tune hyperparameters with RandomizedSearchCV. Lastly, VotingRegressor for an additional model. 
- Evaluation:
    - Firstly, evaluate models using MAE, MSE RMSE and R2, then secondly, evaluate models with a focus on RMSE and MAE (less sensitive to outliers in skewed 'shares') alongside R². It uses cross_val_score from scikit-learn for robust validation across folds. 
    - Final Test Metrics: {'MAE': 2285.3942557115906, 'MSE': 42685559.999009736, 'RMSE': 6533.418706849404, 'r2': 0.3287087982196899}
- Visualization: Below is a simplified flowchart of the pipeline:
    graph TD
        A[Load Data] --> B[Preprocess Data]
        B --> C[Feature Engineering]
        C --> D[Split Data]
        D --> E[Build Pipelines]
        E --> F[Train Baseline Models]
        F --> G[Train Tuned Models]
        G --> H[Evaluation ]

## Key_EDA_Findings_and_Pipeline_Choices
- Data Quality: Missing values: 
    num_hrefs                      721
    num_self_hrefs                 721
    self_reference_min_shares      721
    self_reference_max_shares      721
    self_reference_avg_shares      721
    num_imgs                      1419
    num_videos                   16755
    data_channel                  5462

    It is likely that the features with 721 NaN values are highly related to one another, and are missing together during data input. 

    '0' value, which is not logical given the nature of the features: n_non_stop_words, n_non_stop_unique_tokens, average_token_length. It is likely that the features with 721 NaN values are highly related to one another, and are missing together during data input but input as 0.
- 'ID': There are 3553 rows where there is a gap size of more than 1 between 'ID' and the previous 'ID'. Suspicion that there are nearly 4000 missing rows of data on top of the 35,680 existing. 
- Both 'URL' from which the date of article publishing and timedelta indicate that there is a fairly consistent rate of 30 articles every month. 
- Target variable 'shares': Values range from 4 to 843,300; extremely wide range. 
- Feature Engineering:
    - 'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg' were binned into 'low', 'medium', 'high' buckets, because the maximum values were extremely high and binning the values would mitigate the huge gaps in values.    

## Model_Selection_and_Justification
- 10 different models are used in the EDA and the best results, in terms of r2, of each are:
    Linear Regression           0.273
    Ridge Regression            0.345
    Lasso Regression            0.344
    XGBooster                   0.302
    LightGBM                    0.231
    Elastic Net                 0.273
    Stacking Regressor          0.324
    Random Forest               0.236
    Gradient Boosting           0.273
    Neural Network              0.273

    It is obvious that the top 3 models are Ridge, Lasso and Stacking regressors. 
- It might be due to the following reasons:
    - Ridge and Lasso work best due to their regularization, which mitigates overfitting and handles multicollinearity/noisy features in your data. 
    - Stacking outperforms by combining these strengths with non-linear models, optimizing via a meta-learner—ideal for the dataset’s mixed signal. 
    - Other models:
        - RandomForest: While strong for non-linear data, it might overfit on the dataset’s noise (e.g., outlier 'shares' >843k) without sufficient tuning or pruning. Its R² might lag behind Stacking because it doesn’t benefit from the meta-learner’s optimization or Lasso’s feature selection.
        - LGBM/XGBoost Alone: These gradient-boosting models excel with large datasets or heavy tuning, but the ~35k rows and moderate feature set might not fully exploit their capacity. Without extensive hyperparameter tuning (e.g., num_leaves, max_depth), they could underperform compared to Stacking, which harnesses their strengths alongside others. Early stopping helps, but the ensemble effect dominates.
        - Neural Network likely underperforms due to limited data, a noisy/skewed target, and lack of tailored tuning, making them less effective than Ridge (regularization), Lasso (feature selection), and Stacking (ensemble diversity) for the dataset. With significant effort in preprocessing and tuning, an NN could approach Stacking’s R², but it’s currently less practical given the setup.

## Model_Evaluation
- Using the following metrics to evaluate the models:
    - Mean Absolute Error (MAE) 
    - Mean Squared Error (MSE) 
    - Root Mean Squared Error (RMSE) 
    - R-Squared (r2)

## Configuration
The project uses a config.yaml file for configuration. Key parameters include:
    file_path: Path to the dataset (e.g., mini_project_1_data.csv).
    target_column: The column to predict (e.g., shares).
    val_test_size: Validate and Test set size (e.g., 0.2).
    param_grid: Hyperparameter grid for Ridge and Lasso (e.g., alpha: [0.01, 0.1, 1, 10]).
    scoring: Evaluation metric (e.g., r2).
    numerical_features, nominal_features, ordinal_features: Lists of feature types for preprocessing.
Edit config.yaml to customize the pipeline for your dataset.

## MLmodel_results
- Metrics for ridge (tuned): {'r2': 0.27048155131467566, 'mae': 2399.844412168484, 'mse': 43860621.1804115, 'rmse': 6622.7351736583505}
- Metrics for lasso (tuned): {'r2': 0.27294601143994734, 'mae': 2393.989056560649, 'mse': 43712451.1757138, 'rmse': 6611.539244057604}
- Metrics for stacking (tuned): {'r2': 0.2730990507099713, 'mae': 2320.5490913096314, 'mse': 43703250.04110186, 'rmse': 6610.843368368506}
- Metrics for weighted_voting: {'r2': 0.23502345954316606, 'mae': 2295.6145125379408, 'mse': 45992457.50857171, 'rmse': 6781.77392048509}
- Final Test Metrics for Stacking Regressor (found to be the model yielding the best results)
    - Mean Absolute Error (MAE) = 2285.3942557115906 
    - Mean Squared Error (MSE) = 42685559.999009736
    - Root Mean Squared Error (RMSE) = 6533.418706849404
    - R-Squared (r2) = 0.3287087982196899 
- Slight drop of r2 of Final Model compared to EDA is probably due to using RandomizedSearch instead of GridSearch. RandomizedSearch is used to be more efficient in compute time but with a slight drop in performance. 

## Contributing
We welcome contributions! To contribute:
    Fork the repository.
    Create a new branch (git checkout -b feature/your-feature-name).
    Commit your changes (git commit -m 'Add your feature').
    Push to the branch (git push origin feature/your-feature-name).
    Open a Pull Request.
Please follow our Code of Conduct (CODE_OF_CONDUCT.md) and ensure code adheres to PEP 8 style guidelines.

## License
This project is licensed under the MIT License (LICENSE).

## Contact 
For questions or feedback, reach out to:
    Email: jonyling@hotmail.com
    GitHub: jonyling
    X: @JonyLing1

This README is concise yet informative, tailored to your project's structure. Adjust the repository URL, contact details, and dataset path as needed. Let me know if you'd like to expand any section!
