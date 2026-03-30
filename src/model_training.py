import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.config import PROCESSED_DATA_PATH, MODEL_PATH
from src.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Final production-grade model training pipeline
    """

    def __init__(self):
        self.models = {}
        self.best_model = None

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    def load_data(self):
        logger.info("Loading processed dataset...")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        return df

    # -----------------------------
    # CLEAN FEATURE NAMES
    # -----------------------------
    def clean_feature_names(self, df):
        df.columns = (
            df.columns
            .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
        )
        return df

    # -----------------------------
    # SPLIT + PREPROCESS
    # -----------------------------
    def time_series_split(self, df, split_date="2016-06-01"):
        logger.info("Performing time-series split...")

        df["date"] = pd.to_datetime(df["date"])

        train = df[df["date"] < split_date].copy()
        val = df[df["date"] >= split_date].copy()

        # Remove NaNs (from lag/rolling features)
        train = train.dropna()
        val = val.dropna()

        # Split features and target
        X_train = train.drop(columns=["sales", "date", "id"])
        y_train = train["sales"]

        X_val = val.drop(columns=["sales", "date", "id"])
        y_val = val["sales"]

        # -----------------------------
        # ENCODING
        # -----------------------------
        logger.info("Encoding categorical features...")

        X_train = pd.get_dummies(X_train)
        X_val = pd.get_dummies(X_val)

        # Align columns
        X_train, X_val = X_train.align(
            X_val, join="left", axis=1, fill_value=0
        )

        # Clean feature names (LightGBM fix)
        X_train = self.clean_feature_names(X_train)
        X_val = self.clean_feature_names(X_val)

        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Validation shape: {X_val.shape}")

        return X_train, X_val, y_train, y_val

    # -----------------------------
    # TRAIN MODELS
    # -----------------------------
    def train_models(self, X_train, y_train):
        logger.info("Training models...")

        self.models = {
            "linear": LinearRegression(),

            "rf": RandomForestRegressor(
                n_estimators=150,
                n_jobs=-1,
                random_state=42
            ),

            "xgb": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),

            "lgbm": LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=-1,
                random_state=42
            ),

            "cat": CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                verbose=0
            )
        }

        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_train, y_train)

    # -----------------------------
    # EVALUATE
    # -----------------------------
    def evaluate_models(self, X_val, y_val):
        logger.info("Evaluating models...")

        results = {}

        for name, model in self.models.items():
            preds = model.predict(X_val)

            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))

            results[name] = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2)
            }

            logger.info(f"{name} | MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        return results

    # -----------------------------
    # SELECT BEST MODEL
    # -----------------------------
    def select_best_model(self, results):
        logger.info("Selecting best model based on RMSE...")

        best_name = min(results, key=lambda x: results[x]["RMSE"])
        self.best_model = self.models[best_name]

        logger.info(f"Best model selected: {best_name}")

        return best_name

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    def save_model(self):
        logger.info("Saving best model...")

        joblib.dump(self.best_model, MODEL_PATH)

        logger.info(f"Model saved at: {MODEL_PATH}")

    def get_feature_importance(self, feature_names):
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
                }).sort_values(by="importance", ascending=False)
            print("\nTop Features:")
            print(importance_df.head(10))
            return importance_df
            
    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def run_training_pipeline(self):
        logger.info("Starting training pipeline...")

        df = self.load_data()

        X_train, X_val, y_train, y_val = self.time_series_split(df)

        self.train_models(X_train, y_train)

        results = self.evaluate_models(X_val, y_val)

        best_model = self.select_best_model(results)

        self.get_feature_importance(X_train.columns)

        self.save_model()

        logger.info("Training pipeline completed successfully.")

        return results, best_model