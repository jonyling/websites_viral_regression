import logging
from typing import Any, Dict, Tuple
import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

# In data_preparation.py
class DataPreparation:
    def __init__(self, config: Dict[str, Any], df):
        """
        Initialize DataPreparation with configuration and DataFrame.

        Args:
            config (dict): Configuration dictionary.
            df (pd.DataFrame): Input DataFrame.
        """        
        self.config = config
        self.df = df
        self.preprocessor = self.create_preprocessor(self.df)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data cleaning.")
        
        # 'n_non_stop_unique_tokens' - KNN-Impute the 1061 zero values
        ###############################################################################################################################################################################################################
        imputation_features = [
            'timedelta', 'n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos', 'n_comments', 
            'average_token_length', 'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_shares', 'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 
            'kw_min_avg', 'kw_max_avg', 'kw_avg_avg'
            ]

        # Create a copy of the dataframe with selected features
        df_impute = df[imputation_features].copy()

        # Initialize and fit KNN imputer
        imputer = KNNImputer(n_neighbors=7, weights='uniform')  # You can adjust n_neighbors
        df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns)

        # Replace the original column with imputed values
        df['n_non_stop_unique_tokens'] = df_imputed['n_non_stop_unique_tokens']
        ###############################################################################################################################################################################################################

        # Fill the 721 NaN 'num_hrefs'.
        ###############################################################################################################################################################################################################
        num_hrefs_median_value = df['num_hrefs'].median()
        df['num_hrefs'] = df['num_hrefs'].fillna(num_hrefs_median_value)
        ###############################################################################################################################################################################################################

        # Fill the 721 NaN 'num_self_hrefs'.
        ###############################################################################################################################################################################################################
        num_self_hrefs_median_value = df['num_self_hrefs'].median()
        df['num_self_hrefs'] = df['num_self_hrefs'].fillna(num_self_hrefs_median_value)
        ###############################################################################################################################################################################################################

        # Fill the 698 NaN 'num_imgs'.
        ###############################################################################################################################################################################################################
        num_imgs_median_value = df['num_imgs'].median()
        df['num_imgs'] = df['num_imgs'].fillna(num_imgs_median_value)
        ###############################################################################################################################################################################################################

        # Fill the 16,755 NaN 'num_videos'.
        ###############################################################################################################################################################################################################
        num_videos_median_value = df['num_videos'].median()
        df['num_videos'] = df['num_videos'].fillna(num_videos_median_value)
        ###############################################################################################################################################################################################################

        # 'data_channel' - KNN-Impute the 5462 missing values
        ###############################################################################################################################################################################################################
        channel_mapping = {'world': 0, 'technology': 1, 'entertainment': 2, 'business': 3, 'social_media': 4, 'lifestyle': 5}
        df['data_channel_enum'] = df['data_channel'].map(channel_mapping)
        imputation_features = [
            'timedelta', 'n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos', 'n_comments', 
            'average_token_length', 'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_shares', 'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 
            'kw_min_avg', 'kw_max_avg', 'kw_avg_avg', 'data_channel_enum'
            ]
        df_impute = df[imputation_features].copy()
        imputer = KNNImputer(n_neighbors=7, weights='uniform')  # You can adjust n_neighbors
        df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns)
        df['data_channel_enum'] = df_imputed['data_channel_enum']
        df['data_channel_enum'] = np.round(df['data_channel_enum']).astype(int)
        df['data_channel_enum'] = df['data_channel_enum'].clip(lower=0, upper=5)
        mapping_channel = {0: 'world', 1: 'technology', 2: 'entertainment', 3: 'business', 4: 'social_media', 5: 'lifestyle'}
        df['data_channel'] = df['data_channel_enum'].map(mapping_channel)
        ###############################################################################################################################################################################################################

        # 'self_reference_max_shares' and 'self_reference_avg_shares' - Fill the 721 NaN values.
        ###############################################################################################################################################################################################################
        self_reference_min_shares_median_value = df['self_reference_min_shares'].median()
        df['self_reference_min_shares'] = df['self_reference_min_shares'].fillna(self_reference_min_shares_median_value)
        self_reference_max_shares_median_value = df['self_reference_max_shares'].median()
        df['self_reference_max_shares'] = df['self_reference_max_shares'].fillna(self_reference_max_shares_median_value)
        self_reference_avg_shares_median_value = df['self_reference_avg_shares'].median()
        df['self_reference_avg_shares'] = df['self_reference_avg_shares'].fillna(self_reference_avg_shares_median_value)
        ###############################################################################################################################################################################################################

        # 'kw_min_min' - KNN-impute strange values {-1, 4, 217, 0} .
        ###############################################################################################################################################################################################################
        df['kw_avg_min'] = df['kw_avg_min'].clip(lower=0, upper=50000) # 50000 is a reasonable value upon examining max @ 42827.
        # There are too many entries with very strange value. I would KNN-impute this column.
        strange_values = {-1, 4, 217, 0}

        # Replace strange values with NaN in the kw_min_min column
        df['kw_min_min'] = df['kw_min_min'].replace(strange_values, np.nan)

        # Create a copy of the dataframe with selected features
        df_impute = df[imputation_features].copy()

        # Initialize and fit KNN imputer
        imputer = KNNImputer(n_neighbors=7, weights='uniform')  # You can adjust n_neighbors
        df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns)

        # Replace the original column with imputed values
        df['kw_min_min'] = df_imputed['kw_min_min']

        # Round to the nearest integer and clip to valid categories
        df['kw_min_min'] = np.round(df['kw_min_min']).astype(int)
        df['kw_min_min'] = df['kw_min_min'].clip(lower=0, upper=50000) # 50000 is a reasonable value. 
        ###############################################################################################################################################################################################################

        # 'kw_avg_min' - Fill the 644 -1 values.
        ###############################################################################################################################################################################################################
        df['kw_avg_min'] = df['kw_avg_min'].clip(lower=0, upper=50000) # 50000 is a reasonable value upon examining max @ 42827.
        ###############################################################################################################################################################################################################        

        # Binning the used kw metrics into 'low', 'medium', 'high' buckets. 
        ###############################################################################################################################################################################################################
        def bin_feature(series):
            # Clean data
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            series = series.clip(lower=0, upper=1e6)
            # Compute quantiles manually
            quantiles = series.quantile([0, 1/3, 2/3, 1])
            # Ensure unique bin edges
            quantiles = quantiles.unique()  # Remove duplicates if any
            if len(quantiles) < 4:  # Need at least 4 edges for 3 bins
                quantiles = np.linspace(series.min(), series.max(), 4)  # Fallback to linear spacing
            # Apply binning
            bins = pd.cut(series, bins=quantiles, labels=['low', 'medium', 'high'], include_lowest=True)
            return bins
        # Apply to keyword features
        keyword_features = ['kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg']
        for feature in keyword_features:
            if feature in df.columns:
                df[f'{feature}_binned'] = bin_feature(df[feature])
            else:
                print(f"Warning: {feature} not found in df")
        ###############################################################################################################################################################################################################
   
        logging.info("Data cleaning completed.")
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        logging.info("Splitting data started.")
        # Separate the data into features and target. 
        X = df.drop(columns=self.config["dropped_features"]) 
        y = df[self.config["target_column"]]
        # Split the data into training (80%) and test-validation (20%) sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        # Split the test-validation set (20%) into validation (10%) and test (10%) sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Create a preprocessing pipeline.

        Args:
            df (pd.DataFrame): Input DataFrame to determine column types.

        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        numerical_features = ['timedelta', 'n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos', 'n_comments', 
                              'average_token_length', 'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_shares', 'num_keywords']
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        nominal_features = ['data_channel', 'weekday']
        nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        ordinal_features = ['kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg']
        kw_min_min_binned_order = ['low', 'medium', 'high']
        kw_max_min_binned_order = ['low', 'medium', 'high']
        kw_avg_min_binned_order = ['low', 'medium', 'high']
        kw_min_max_binned_order = ['low', 'medium', 'high']
        kw_max_max_binned_order = ['low', 'medium', 'high']
        kw_avg_max_binned_order = ['low', 'medium', 'high']
        kw_min_avg_binned_order = ['low', 'medium', 'high']
        kw_max_avg_binned_order = ['low', 'medium', 'high']
        kw_avg_avg_binned_order = ['low', 'medium', 'high']
        category_orders = [kw_min_min_binned_order, kw_max_min_binned_order, kw_avg_min_binned_order, kw_min_max_binned_order, kw_max_max_binned_order, kw_avg_max_binned_order, kw_min_avg_binned_order, kw_max_avg_binned_order, 
                           kw_avg_avg_binned_order]
        ordinal_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder(categories=category_orders))])

        passthrough_features = []

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('nom', nominal_transformer, nominal_features),
                ('ord', ordinal_transformer, ordinal_features),
                ('pass', 'passthrough', passthrough_features)
            ],
            remainder='drop',  # Drop unused features
            n_jobs=-1)
        return preprocessor
