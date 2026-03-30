import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

TRAIN_PATH = os.path.join(DATA_RAW, "train.csv")
TEST_PATH = os.path.join(DATA_RAW, "test.csv")
STORES_PATH = os.path.join(DATA_RAW, "stores.csv")
TRANSACTIONS_PATH = os.path.join(DATA_RAW, "transactions.csv")
HOLIDAYS_PATH = os.path.join(DATA_RAW, "holidays_events.csv")
OIL_PATH = os.path.join(DATA_RAW, "oil.csv")

PROCESSED_DATA_PATH = os.path.join(DATA_PROCESSED, "final_dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

LOG_PATH = os.path.join(BASE_DIR, "logs", "app.log")