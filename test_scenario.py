import pandas as pd
from src.scenario_engine import ScenarioEngine
from src.config import PROCESSED_DATA_PATH

# Load processed data
df = pd.read_csv(PROCESSED_DATA_PATH)

# Initialize scenario engine
engine = ScenarioEngine()

# Run scenario
result = engine.run_scenario(
    df,
    promo_increase_pct=0.2,   # +20% promotions
    demand_shift_pct=0.1      # +10% demand
)

# Show results
print(result[["sales", "prediction"]].head())