#  AI-Powered Revenue Forecasting System

##  Overview

This project is a production-grade machine learning system designed to forecast revenue using historical retail data.

It includes an end-to-end pipeline from data ingestion to forecasting and scenario-based simulation.

---

##  Features

###  Data Pipeline

* Data cleaning & preprocessing
* Handling missing values
* Merging multiple datasets

###  Feature Engineering

* Date features (month, year, etc.)
* Lag features
* Rolling statistics

###  Model Training

* Linear Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost

###  Model Evaluation

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* Automatic best model selection

###  Forecasting

* Predict future revenue trends
* Supports long-term forecasting (up to 1 year)

###  Scenario Simulation

* Adjust sales %
* Modify promotion impact
* Simulate business decisions

---

## Best Model Performance

| Model         | RMSE            |
| ------------- | --------------- |
| Random Forest | ~211–220   Best |
| LightGBM      | ~222            |
| XGBoost       | ~234            |
| Linear        | ~500+           |

---

##  Project Structure

```
project/
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── forecasting.py
│   ├── scenario_engine.py
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## 
How to Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

---

## Dataset

Dataset used:
https://www.kaggle.com/competitions/store-sales-time-series-forecasting

Note: Data is not included due to GitHub size limits.

---

##  Future Improvements

* Streamlit dashboard
* API deployment
* Real-time forecasting
* Cloud integration (AWS/GCP)

---

##  Use Case

This system can be used by:

* Retail businesses
* E-commerce platforms
* Financial forecasting teams

---

##  Author

Aakash Rengasamy
