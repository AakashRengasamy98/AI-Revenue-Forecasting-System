import pandas as pd
import joblib

from src.config import MODEL_PATH
from src.logger import get_logger

logger = get_logger(__name__)


class ScenarioEngine:
    """
    Scenario simulation engine for business what-if analysis
    """

    def __init__(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully for scenario engine.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    # -----------------------------
    # APPLY SCENARIO CHANGES
    # -----------------------------
    def apply_scenario(
        self,
        df: pd.DataFrame,
        promo_increase_pct: float = 0.0,
        demand_shift_pct: float = 0.0,
    ) -> pd.DataFrame:
        """
        Apply business scenario adjustments to dataset
        """

        logger.info("Applying scenario adjustments...")

        df = df.copy()

        # Promotion impact
        if "onpromotion" in df.columns:
            df["onpromotion"] = df["onpromotion"] * (1 + promo_increase_pct)

        # Demand shift multiplier
        df["scenario_multiplier"] = 1 + demand_shift_pct

        return df

    # -----------------------------
    # CLEAN FEATURE NAMES (IMPORTANT)
    # -----------------------------
    def clean_feature_names(self, df):
        df.columns = (
            df.columns
            .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
        )
        return df

    # -----------------------------
    # PREPARE FEATURES FOR MODEL
    # -----------------------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data exactly like training pipeline
        """

        X = df.drop(columns=["sales", "date"], errors="ignore")

        # Encode
        X = pd.get_dummies(X)

        # Clean column names
        X = self.clean_feature_names(X)

        # Align with model features
        try:
            model_features = self.model.feature_names_in_
            X = X.reindex(columns=model_features, fill_value=0)
        except Exception:
            logger.warning("Model feature names not found. Skipping alignment.")

        return X

    # -----------------------------
    # PREDICT
    # -----------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions from model
        """

        logger.info("Generating predictions...")

        X = self.prepare_features(df)

        preds = self.model.predict(X)

        df["prediction"] = preds

        # Apply scenario multiplier
        if "scenario_multiplier" in df.columns:
            df["prediction"] = df["prediction"] * df["scenario_multiplier"]

        return df

    # -----------------------------
    # MAIN SCENARIO PIPELINE
    # -----------------------------
    def run_scenario(
        self,
        df: pd.DataFrame,
        promo_increase_pct: float = 0.0,
        demand_shift_pct: float = 0.0,
    ) -> pd.DataFrame:
        """
        Run full scenario simulation pipeline
        """

        logger.info("Running scenario simulation...")

        df = self.apply_scenario(
            df,
            promo_increase_pct=promo_increase_pct,
            demand_shift_pct=demand_shift_pct,
        )

        df = self.predict(df)

        logger.info("Scenario simulation completed.")

        return df