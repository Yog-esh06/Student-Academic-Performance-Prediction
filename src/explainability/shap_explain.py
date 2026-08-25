"""SHAP values and feature importance."""
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import config
from src.data.load_data import load_raw_data
from src.data.clean_data import clean_data
from src.data.preprocess import get_train_test_split

def get_shap_explainer(model, X_train):
    """Returns the appropriate SHAP explainer based on model type."""
    tree_models = ['RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor', 'DecisionTreeRegressor']
    model_name = type(model).__name__
    
    if model_name in tree_models:
        return shap.TreeExplainer(model)
    else:
        # LinearExplainer works perfectly for our winning Linear Regression model
        return shap.LinearExplainer(model, X_train)

def get_global_feature_importance(model, X_train, feature_names):
    """Calculates global feature importance and saves the SHAP summary plot."""
    explainer = get_shap_explainer(model, X_train)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_train)
    
    # Calculate mean absolute SHAP values per feature
    vals = np.abs(shap_values).mean(0)
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": vals
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    # Save to CSV
    csv_path = config.BASE_DIR / "reports" / "metrics" / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"[*] Global feature importance saved to {csv_path}")
    
    # Generate and save plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plot_path = config.BASE_DIR / "reports" / "figures" / "shap_summary.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"[*] SHAP summary plot saved to {plot_path}")
    
    return importance_df

def explain_single_prediction(explainer, X_row, feature_names):
    """Returns the top 5 features driving an individual prediction."""
    shap_values = explainer.shap_values(X_row)
    
    # Extract values for the single instance
    contributions = shap_values[0]
    actual_values = X_row[0]
    
    feature_contributions = []
    for i in range(len(feature_names)):
        feature_contributions.append({
            "feature": feature_names[i],
            "value": actual_values[i],
            "shap_contribution": contributions[i]
        })
        
    # Sort by absolute contribution, descending to find the biggest drivers
    feature_contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
    return feature_contributions[:5]

if __name__ == "__main__":
    print("--- SHAP Explainability Pipeline ---")
    
    # 1. Load Data & Preprocessor
    df = clean_data(load_raw_data())
    X_train, _, _, _ = get_train_test_split(df)
    
    preprocessor = joblib.load(config.MODEL_DIR / "preprocessor.pkl")
    X_train_transformed = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    
    # 2. Load Best Model
    best_model = joblib.load(config.MODEL_DIR / "best_model.pkl")
    print(f"\n[*] Loaded best model: {type(best_model).__name__}")
    
    # 3. Global Importance
    print("[*] Calculating global feature importance...")
    importance_df = get_global_feature_importance(best_model, X_train_transformed, feature_names)
    
    print("\n--- Top 10 Global Features ---")
    print(importance_df.head(10).to_string(index=False))