import pandas as pd
import numpy as np
import joblib
from src.config import MODEL_PATH


class Forecaster:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict_future(self, df, days=365):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        predictions = []

        for i in range(days):

            last_row = df.iloc[-1:].copy()
            next_date = last_row["date"].values[0] + np.timedelta64(1, 'D')

            new_row = last_row.copy()
            new_row["date"] = next_date

            # update time features
            new_row["dayofweek"] = pd.to_datetime(next_date).dayofweek
            new_row["month"] = pd.to_datetime(next_date).month

            #  IMPORTANT: lag features
            new_row["lag_1"] = df["sales"].iloc[-1]
            new_row["lag_7"] = df["sales"].iloc[-7]

            # rolling
            new_row["rolling_mean_7"] = df["sales"].tail(7).mean()

            # drop target
            X = new_row.drop(columns=["sales", "date"], errors="ignore")

            # encoding
            X = pd.get_dummies(X)
            X = X.reindex(columns=self.model.feature_names_in_, fill_value=0)

            pred = self.model.predict(X)[0]

            new_row["sales"] = pred

            df = pd.concat([df, new_row], ignore_index=True)

            predictions.append({
                "date": next_date,
                "prediction": pred
            })

        return pd.DataFrame(predictions)