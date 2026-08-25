"""Centralized configuration, paths, and constants."""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "StudentPerformanceFactors.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.parquet"
MODEL_DIR = BASE_DIR / "models_store"

# ML Constants
RANDOM_STATE = 42
TARGET_COLUMN = "Exam_Score"
TEST_SIZE = 0.2
CV_FOLDS = 5

# Risk Thresholds
# Below 60 = High Risk, 60-75 = Moderate Risk, Above 75 = Low Risk
RISK_THRESHOLDS = {
    "high_risk_max": 60,
    "moderate_risk_max": 75
}