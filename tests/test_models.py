"""Tests for model and risk modules."""
from src import config
from src.risk.risk_classifier import classify_risk
from src.recommendations.recommend import generate_recommendations

def test_classify_risk():
    """Test that risk classification aligns perfectly with config thresholds."""
    # Using actual config values dynamically so tests don't break if thresholds change
    high_threshold = config.RISK_THRESHOLDS["high_risk_max"]
    mod_threshold = config.RISK_THRESHOLDS["moderate_risk_max"]
    
    # 1 point below High Risk threshold
    assert classify_risk(high_threshold - 1) == "High Risk"
    # Exact High Risk threshold should fall into Moderate Risk (since condition is <)
    assert classify_risk(high_threshold) == "Moderate Risk"
    
    # Moderate risk bounds
    assert classify_risk(mod_threshold - 5) == "Moderate Risk"
    
    # 1 point above Moderate Risk threshold
    assert classify_risk(mod_threshold + 1) == "Low Risk"
    
    # Hardcoded fallback tests just for sanity checks
    assert classify_risk(50) == "High Risk"
    assert classify_risk(70) == "Moderate Risk"
    assert classify_risk(90) == "Low Risk"

def test_generate_recommendations():
    """Test that the recommender returns relevant string advice based on SHAP mock data."""
    # Mocking a student with bad attendance and low study hours
    mock_top_features = [
        {"feature": "num__Attendance", "value": 65, "shap_contribution": -3.5},
        {"feature": "num__Hours_Studied", "value": 5, "shap_contribution": -1.2},
        {"feature": "cat__Gender_Male", "value": 1, "shap_contribution": 0.1} # Shouldn't trigger a rule
    ]
    
    recs = generate_recommendations(mock_top_features, risk_category="High Risk")
    
    # Assertions
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert all(isinstance(r, str) for r in recs)
    
    # Check if the specific rules fired
    recs_text = " ".join(recs).lower()
    assert "attendance" in recs_text
    assert "study hours" in recs_text
    assert "tutoring" in recs_text # Triggered by the High Risk fallback