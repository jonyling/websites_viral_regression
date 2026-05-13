import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    target_column: str = "shares_logged"
    test_size: float = 0.20
    validation_size: float = 0.50
    random_state: int = 42
    use_polynomial_features: bool = True


class DataPreparation:
    """Prepare Mashable article data according to the final EDA notebook."""

    selected_columns = [
        "weekday",
        "data_channel",
        "is_weekend",
        "n_comments_logged",
        "n_tokens_content_logged",
        "n_unique_tokens_logged",
        "average_token_length",
        "num_hrefs",
        "num_self_hrefs",
        "num_imgs",
        "self_reference_min_shares",
        "self_reference_max_shares",
        "self_reference_avg_shares",
        "kw_min_min_logged",
        "kw_max_min_logged",
        "kw_avg_min_logged",
        "kw_min_max_logged",
        "kw_max_max_logged",
        "kw_avg_max_logged",
        "kw_min_avg_logged",
        "kw_max_avg_logged",
        "kw_avg_avg_logged",
        "kw_engagement_ratio_logged",
        "content_keyword_interaction",
    ]

    numerical_features = [
        "is_weekend",
        "n_comments_logged",
        "n_tokens_content_logged",
        "n_unique_tokens_logged",
        "average_token_length",
        "num_hrefs",
        "num_self_hrefs",
        "num_imgs",
        "self_reference_min_shares",
        "self_reference_max_shares",
        "self_reference_avg_shares",
        "kw_min_min_logged",
        "kw_max_min_logged",
        "kw_avg_min_logged",
        "kw_min_max_logged",
        "kw_max_max_logged",
        "kw_avg_max_logged",
        "kw_min_avg_logged",
        "kw_max_avg_logged",
        "kw_avg_avg_logged",
        "kw_engagement_ratio_logged",
        "content_keyword_interaction",
    ]

    nominal_features = ["weekday", "data_channel"]

    polynomial_features = [
        "kw_avg_avg_logged",
        "self_reference_avg_shares",
        "n_comments_logged",
        "kw_engagement_ratio_logged",
        "n_tokens_content_logged",
    ]

    keyword_features = [
        "kw_min_min",
        "kw_max_min",
        "kw_avg_min",
        "kw_min_max",
        "kw_max_max",
        "kw_avg_max",
        "kw_min_avg",
        "kw_max_avg",
        "kw_avg_avg",
    ]

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean missing values and create the engineered features used by the EDA."""
        LOGGER.info("Starting data preparation.")
        cleaned = df.copy()

        if "weekday" in cleaned:
            cleaned["weekday"] = cleaned["weekday"].astype("string").str.lower()

        self._fill_numeric_medians(
            cleaned,
            [
                "num_hrefs",
                "num_self_hrefs",
                "num_imgs",
                "num_videos",
                "self_reference_min_shares",
                "self_reference_max_shares",
                "self_reference_avg_shares",
            ],
        )

        for col in ["n_non_stop_words", "n_non_stop_unique_tokens"]:
            if col in cleaned:
                median_value = cleaned.loc[cleaned[col] > 0, col].median()
                cleaned[col] = cleaned[col].replace(0, median_value)

        if "data_channel" in cleaned:
            mode_value = cleaned["data_channel"].mode(dropna=True)
            fallback_channel = mode_value.iloc[0] if not mode_value.empty else "world"
            cleaned["data_channel"] = cleaned["data_channel"].fillna(fallback_channel)

        if "kw_min_min" in cleaned:
            strange_values = {-1, 0, 4, 217}
            replacement = cleaned["kw_min_min"].replace(strange_values, np.nan).mean() # type: ignore
            cleaned["kw_min_min"] = cleaned["kw_min_min"].replace(strange_values, replacement) # type: ignore

        # The notebook uses log1p throughout to reduce the heavy right skew.
        cleaned["shares_logged"] = self._safe_log1p(cleaned["shares"])
        cleaned["n_tokens_content_logged"] = self._safe_log1p(cleaned["n_tokens_content"])
        cleaned["n_unique_tokens_logged"] = self._safe_log1p(cleaned["n_unique_tokens"])
        cleaned["n_comments_logged"] = self._safe_log1p(cleaned["n_comments"])

        for col in self.keyword_features:
            cleaned[f"{col}_logged"] = self._safe_log1p(cleaned[col])

        cleaned["is_weekend"] = cleaned["weekday"].isin(["saturday", "sunday"]).astype(int)
        cleaned["kw_engagement_ratio"] = cleaned["kw_avg_avg"] / (cleaned["kw_max_avg"] + 1e-6)
        cleaned["kw_engagement_ratio_logged"] = self._safe_log1p(cleaned["kw_engagement_ratio"])
        cleaned["content_keyword_interaction"] = (
            cleaned["n_tokens_content_logged"] * cleaned["num_keywords"]
        )

        cleaned = self._replace_inf_with_nan(cleaned)
        LOGGER.info("Data preparation completed with %s rows.", len(cleaned))
        return cleaned

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        X = df[self.selected_columns].copy()
        y = df[self.config.target_column].copy()

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_preprocessor(self) -> ColumnTransformer:
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        nominal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        transformers = [
            ("num", numerical_transformer, self.numerical_features),
            ("nom", nominal_transformer, self.nominal_features),
        ]

        if self.config.use_polynomial_features:
            poly_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=2, interaction_only=True, include_bias=False
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("poly", poly_transformer, self.polynomial_features))

        return ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=-1)

    @staticmethod
    def _fill_numeric_medians(df: pd.DataFrame, columns: list[str]) -> None:
        for col in columns:
            if col in df:
                df[col] = df[col].fillna(df[col].median())

    @staticmethod
    def _safe_log1p(series: pd.Series) -> pd.Series:
        return np.log1p(pd.to_numeric(series, errors="coerce").clip(lower=0)) # type: ignore

    @staticmethod
    def _replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
        return out
