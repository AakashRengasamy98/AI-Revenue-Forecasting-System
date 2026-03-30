from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer

from src.logger import get_logger

logger = get_logger(__name__)


def main():
    try:
        print("🚀 Running Data Pipeline...")

        # -----------------------------
        # STEP 1: DATA PROCESSING
        # -----------------------------
        processor = DataProcessor()
        df = processor.run_pipeline()

        print("⚙️ Feature Engineering...")

        # -----------------------------
        # STEP 2: FEATURE ENGINEERING
        # -----------------------------
        fe = FeatureEngineer()
        df_features = fe.run_feature_pipeline(df)

        # Save updated features (optional but useful)
        df_features.to_csv("data/processed/final_features.csv", index=False)

        print("🧠 Training Models...")

        # -----------------------------
        # STEP 3: MODEL TRAINING
        # -----------------------------
        trainer = ModelTrainer()
        results, best_model = trainer.run_training_pipeline()

        print("✅ Training Complete!")
        print(f"Best Model: {best_model}")
        print("Results:", results)

        logger.info("Full pipeline executed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print("❌ Pipeline failed. Check logs for details.")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()