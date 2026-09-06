"""Flask API and Application."""
import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify, send_from_directory

from src import config
from src.risk.risk_classifier import CostSensitiveRiskModel
from src.explainability.shap_explain import get_shap_explainer, explain_single_prediction
from src.recommendations.recommend import generate_recommendations
from src.data.eda_generator import generate_full_eda

app = Flask(__name__)

# Initialize the Temporal & Cost-Sensitive Risk Engine
risk_engine = CostSensitiveRiskModel()

# Auto-generate EDA files on startup if they don't exist
json_check = config.BASE_DIR / "src" / "app" / "static" / "inferences.json"
if not json_check.exists():
    print("[*] Generating EDA figures and inferences automatically...")
    generate_full_eda()

# Load ML Artifacts
best_model = joblib.load(config.MODEL_DIR / "best_model.pkl")
preprocessor = joblib.load(config.MODEL_DIR / "preprocessor.pkl")
processed_df = pd.read_csv(config.PROCESSED_DATA_PATH.with_suffix('.csv'))
X_train = processed_df.drop(columns=[config.TARGET_COLUMN])
X_train_transformed = preprocessor.transform(X_train)
feature_names = preprocessor.get_feature_names_out()
explainer = get_shap_explainer(best_model, X_train_transformed)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/reports/<path:filename>")
def serve_reports(filename):
    """Serve EDA images and metric CSVs to the frontend."""
    return send_from_directory(config.BASE_DIR / "reports", filename)

@app.route("/api/model-stats", methods=["GET"])
def get_model_stats():
    csv_path = config.BASE_DIR / "reports" / "metrics" / "model_comparison.csv"
    df = pd.read_csv(csv_path)
    return jsonify({
        "labels": df["Model"].tolist(),
        "rmse": df["RMSE"].tolist()
    })

@app.route("/api/model-stats-full", methods=["GET"])
def get_model_stats_full():
    """Returns all metrics (RMSE, R2, MAE, MSE) for all models."""
    csv_path = config.BASE_DIR / "reports" / "metrics" / "model_comparison.csv"
    df = pd.read_csv(csv_path)
    return jsonify({
        "labels": df["Model"].tolist(),
        "rmse": df["RMSE"].tolist(),
        "r2": df["R2"].tolist(),
        "mae": df["MAE"].tolist(),
        "mse": df["MSE"].tolist()
    })

@app.route("/api/download-stats", methods=["GET"])
def download_stats():
    directory = config.BASE_DIR / "reports" / "metrics"
    return send_from_directory(directory, "model_comparison.csv", as_attachment=True)

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.json
    numeric_fields = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
    
    for field in numeric_fields:
        if field in form_data:
            form_data[field] = float(form_data[field])

    input_df = pd.DataFrame([form_data])
    input_transformed = preprocessor.transform(input_df)
    
    # 1. Base Score Prediction
    predicted_score = round(best_model.predict(input_transformed)[0], 1)
    
    # 2. Inject Defaults for Missing Features
    default_features = {
        "Motivation_Level": "Medium", "Access_to_Resources": "Medium",
        "Family_Income": "Medium", "Teacher_Quality": "Medium",
        "Peer_Influence": "Neutral", "School_Type": "Public",
        "Internet_Access": "Yes", "Extracurricular_Activities": "No",
        "Learning_Disabilities": "No", "Parental_Education_Level": "High School",
        "Distance_from_Home": "Near", "Gender": "Male",
        "Parental_Involvement": "Medium"
    }
    risk_input_data = {**default_features, **form_data}
    
    # 3. Cost-Sensitive Risk & Uncertainty Inference
    try:
        risk_results = risk_engine.predict_risk(risk_input_data)
        risk_category = risk_results['status']
    except Exception as e:
        print(f"Risk Engine Error: {e}")
        risk_category = "Unknown"
        risk_results = {"is_uncertain": False, "risk_flag": False, "uncertainty_score": 0.0}

    # 4. Base SHAP Explanations
    top_features = explain_single_prediction(explainer, input_transformed, feature_names)
    recommendations = generate_recommendations(top_features, risk_category)
    
    # 5. Recommendations Injection
    if risk_results.get("is_uncertain"):
        recommendations.insert(0, "⚠️ MODEL UNCERTAINTY HIGH: The student's profile shows conflicting temporal signals. A human educator must manually review this case.")
    elif risk_results.get("risk_flag"):
        recommendations.insert(0, "🚨 HIGH RISK DETECTED: Cost-sensitive analysis flags this student for immediate early intervention.")
        if form_data.get('Attendance', 100) < 75:
            recommendations.insert(1, "Action: Phase 1 Engagement is critically low. Prioritize attendance recovery.")
    else:
        recommendations.insert(0, "✅ On Track: Student demonstrates stable progression across temporal phases.")

    # Format SHAP Features
    clean_features = []
    for f in top_features:
        clean_name = f['feature'].replace('num__', '').replace('cat__', '').replace('_', ' ')
        clean_features.append({
            "name": clean_name,
            "value": round(f['value'], 2),
            "contribution": round(f['shap_contribution'], 2)
        })
        
    # Format Temporal & Uncertainty Features for the New Dedicated Box
    temp_df = risk_engine._engineer_temporal_features(pd.DataFrame([risk_input_data]))
    risk_features = [
        {"name": "Phase 1 Early Engagement", "value": round(float(temp_df['Phase1_Engagement'].iloc[0]), 2), "desc": "Evaluates attendance weighted by motivation trajectory."},
        {"name": "Phase 2 Midterm Consistency", "value": round(float(temp_df['Phase2_Consistency'].iloc[0]), 2), "desc": "Combines previous scores and tutoring progression."},
        {"name": "Phase 3 Late Term Fatigue", "value": round(float(temp_df['Phase3_Fatigue'].iloc[0]), 2), "desc": "Measures study-to-sleep ratios and physical activity balance."},
        {"name": "Prediction Uncertainty (Entropy)", "value": risk_results.get("uncertainty_score", 0.0), "desc": "Shannon Entropy measuring model confidence spread."}
    ]
    
    return jsonify({
        "score": predicted_score,
        "risk": risk_category,
        "features": clean_features,
        "risk_features": risk_features,
        "recommendations": recommendations
    })

@app.route("/api/dataset-inferences", methods=["GET"])
def get_inferences():
    import json
    json_path = config.BASE_DIR / "src" / "app" / "static" / "inferences.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, port=5000)