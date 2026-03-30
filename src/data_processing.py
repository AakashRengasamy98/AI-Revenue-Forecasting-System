import pandas as pd
import os

from src.config import (
    TRAIN_PATH,
    STORES_PATH,
    TRANSACTIONS_PATH,
    HOLIDAYS_PATH,
    OIL_PATH,
    PROCESSED_DATA_PATH
)
from src.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Production-grade data processing pipeline
    """

    def __init__(self):
        self.train = None
        self.stores = None
        self.transactions = None
        self.holidays = None
        self.oil = None

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    def load_data(self):
        try:
            logger.info("Loading datasets...")

            self.train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
            self.stores = pd.read_csv(STORES_PATH)
            self.transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])
            self.holidays = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])
            self.oil = pd.read_csv(OIL_PATH, parse_dates=["date"])

            logger.info("Datasets loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    def clean_data(self):
        logger.info("Cleaning data...")

        # Oil price missing values
        self.oil["dcoilwtico"] = self.oil["dcoilwtico"].ffill().bfill()

        # Remove negative sales
        self.train = self.train[self.train["sales"] >= 0]

        #  OUTLIER CAPPING (BEST PRACTICE)
        upper_limit = self.train["sales"].quantile(0.99)
        self.train["sales"] = self.train["sales"].clip(upper=upper_limit)

        # Fill missing promotions
        self.train["onpromotion"] = self.train["onpromotion"].fillna(0)

        logger.info("Data cleaning completed.")

    # -----------------------------
    # MERGE DATA
    # -----------------------------
    def merge_data(self):
        logger.info("Merging datasets...")

        df = self.train.merge(self.stores, on="store_nbr", how="left")
        df = df.merge(self.transactions, on=["date", "store_nbr"], how="left")
        df = df.merge(self.oil, on="date", how="left")

        # -----------------------------
        # HOLIDAY PROCESSING
        # -----------------------------
        try:
            self.holidays = self.holidays[self.holidays["transferred"] == False]

            if "type" in self.holidays.columns:
                holidays_df = self.holidays[["date", "type"]].copy()
                holidays_df.rename(columns={"type": "holiday_type"}, inplace=True)

                df = df.merge(holidays_df, on="date", how="left")
            else:
                logger.warning("Column 'type' not found in holidays dataset")

        except Exception as e:
            logger.warning(f"Holiday processing issue: {e}")

        # Ensure column always exists
        if "holiday_type" not in df.columns:
            df["holiday_type"] = "None"

        df["holiday_type"] = df["holiday_type"].fillna("None")

        # Fill oil again after merge
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

        logger.info("Merging completed.")
        return df

    # -----------------------------
    # HANDLE MISSING
    # -----------------------------
    def handle_missing(self, df):
        logger.info("Handling missing values...")

        df["transactions"] = df["transactions"].fillna(0)
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

        return df

    # -----------------------------
    # SAVE DATA
    # -----------------------------
    def save_processed_data(self, df):
        logger.info("Saving processed data...")

        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)

        logger.info(f"Processed data saved at {PROCESSED_DATA_PATH}")

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def run_pipeline(self):
        self.load_data()
        self.clean_data()

        df = self.merge_data()
        df = self.handle_missing(df)

        self.save_processed_data(df)

        logger.info("Data pipeline completed successfully.")
        return df






