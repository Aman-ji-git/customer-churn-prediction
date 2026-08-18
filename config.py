import os

DATA_PATH = os.getenv("CHURN_DATA_PATH", "data/customer_churn.csv")
MODEL_PATH = os.getenv("CHURN_MODEL_PATH", "models/churn_model.joblib")

# PostgreSQL example:
# postgresql+psycopg2://postgres:password@localhost:5432/churn_db
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///churn_predictions.db",
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
