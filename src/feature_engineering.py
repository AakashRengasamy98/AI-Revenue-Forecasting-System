import pandas as pd
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Production-grade feature engineering pipeline
    """

    # -----------------------------
    # DATE FEATURES
    # -----------------------------
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating date features...")

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

        # Seasonality flags
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

        return df

    # -----------------------------
    # LAG FEATURES (VERY IMPORTANT)
    # -----------------------------
    def create_lag_features(
        self,
        df: pd.DataFrame,
        lags=[1, 3, 7, 14, 30]
    ) -> pd.DataFrame:
        logger.info("Creating lag features...")

        for lag in lags:
            df[f"lag_{lag}"] = (
                df.groupby(["store_nbr", "family"])["sales"]
                .shift(lag)
            )

        return df

    # -----------------------------
    # ROLLING FEATURES (VERY IMPORTANT)
    # -----------------------------
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating rolling features...")

        group = df.groupby(["store_nbr", "family"])["sales"]

        df["rolling_mean_7"] = group.transform(
            lambda x: x.shift(1).rolling(7).mean()
        )

        df["rolling_mean_14"] = group.transform(
            lambda x: x.shift(1).rolling(14).mean()
        )

        df["rolling_std_7"] = group.transform(
            lambda x: x.shift(1).rolling(7).std()
        )

        return df

    # -----------------------------
    # TREND FEATURE
    # -----------------------------
    def create_trend_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating trend feature...")

        df = df.sort_values("date")
        df["trend"] = range(len(df))

        return df

    # -----------------------------
    # ENCODE CATEGORICAL
    # -----------------------------
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding categorical features...")

        df = pd.get_dummies(
            df,
            columns=["family", "holiday_type"],
            drop_first=True
        )

        return df

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def run_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering pipeline...")

        df = df.sort_values(["store_nbr", "family", "date"])

        df = self.create_date_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_trend_feature(df)

        df = self.encode_categorical(df)

        # Drop NaNs from lag/rolling
        df = df.dropna()

        logger.info("Feature engineering completed.")

        return df