"""Flask API and Application."""
import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify, send_from_directory

from src import config
from src.risk.risk_classifier import classify_risk
from src.explainability.shap_explain import get_shap_explainer, explain_single_prediction
from src.recommendations.recommend import generate_recommendations
from src.data.eda_generator import generate_full_eda

app = Flask(__name__)

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
    
    predicted_score = round(best_model.predict(input_transformed)[0], 1)
    risk_category = classify_risk(predicted_score)
    top_features = explain_single_prediction(explainer, input_transformed, feature_names)
    recommendations = generate_recommendations(top_features, risk_category)
    
    clean_features = []
    for f in top_features:
        clean_name = f['feature'].replace('num__', '').replace('cat__', '').replace('_', ' ')
        clean_features.append({
            "name": clean_name,
            "value": round(f['value'], 2),
            "contribution": round(f['shap_contribution'], 2)
        })
    
    return jsonify({
        "score": predicted_score,
        "risk": risk_category,
        "features": clean_features,
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