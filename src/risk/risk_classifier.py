"""Classifies risk based on predicted score."""
import pandas as pd
import numpy as np
from src import config

def classify_risk(predicted_score: float) -> str:
    """
    Classifies a student's risk level based on their predicted exam score.
    
    Thresholds defined in config:
    - Below 60: High Risk (Likely to fail, needs immediate intervention)
    - 60 to 75: Moderate Risk (Passing, but vulnerable)
    - Above 75: Low Risk (On track for success)
    """
    if predicted_score < config.RISK_THRESHOLDS["high_risk_max"]:
        return "High Risk"
    elif predicted_score <= config.RISK_THRESHOLDS["moderate_risk_max"]:
        return "Moderate Risk"
    else:
        return "Low Risk"

def classify_risk_batch(predicted_scores) -> pd.Series:
    """Vectorized version of classify_risk for processing arrays/series."""
    scores = pd.Series(predicted_scores)
    conditions = [
        scores < config.RISK_THRESHOLDS["high_risk_max"],
        (scores >= config.RISK_THRESHOLDS["high_risk_max"]) & (scores <= config.RISK_THRESHOLDS["moderate_risk_max"])
    ]
    choices = ["High Risk", "Moderate Risk"]
    return pd.Series(np.select(conditions, choices, default="Low Risk"))