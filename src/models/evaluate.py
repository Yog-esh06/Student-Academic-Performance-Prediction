"""Evaluates models and compares metrics."""
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.data.load_data import load_raw_data
from src.data.clean_data import clean_data
from src.data.preprocess import get_train_test_split

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Calculates evaluation metrics for a single model."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def evaluate_all(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Evaluates all models and returns a DataFrame sorted by RMSE."""
    results = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        
    df_results = pd.DataFrame(results)
    # Reorder columns to put Model first
    df_results = df_results[["Model", "MAE", "MSE", "RMSE", "R2"]]
    # Sort by RMSE (lower is better)
    df_results = df_results.sort_values(by="RMSE").reset_index(drop=True)
    return df_results

def save_metrics_report(comparison_df: pd.DataFrame):
    """Saves the comparison table and prints it."""
    save_path = config.BASE_DIR / "reports" / "metrics" / "model_comparison.csv"
    comparison_df.to_csv(save_path, index=False)
    print(f"\n[*] Metrics report saved to: {save_path}\n")
    print("--- Model Comparison Report ---")
    print(comparison_df.to_string(index=False))

def select_best_model(comparison_df: pd.DataFrame, models_dict: dict):
    """Returns the name and object of the best model (lowest RMSE)."""
    best_model_name = comparison_df.iloc[0]["Model"]
    return best_model_name, models_dict[best_model_name]

if __name__ == "__main__":
    print("--- Model Evaluation Pipeline ---")
    
    # 1. Load test data
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    # 2. Load preprocessor and transform test data
    preprocessor = joblib.load(config.MODEL_DIR / "preprocessor.pkl")
    X_test_transformed = preprocessor.transform(X_test)
    
    # 3. Load saved models
    model_names = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]
    loaded_models = {}
    for name in model_names:
        path = config.MODEL_DIR / f"{name.replace(' ', '_')}.pkl"
        loaded_models[name] = joblib.load(path)
        
    # 4. Evaluate and compare
    comparison_df = evaluate_all(loaded_models, X_test_transformed, y_test)
    save_metrics_report(comparison_df)
    
    # 5. Identify the best model
    best_name, best_model = select_best_model(comparison_df, loaded_models)
    print(f"\n[*] Selected Best Model: {best_name} (Lowest RMSE)")
    
    # Explicitly save the best model as 'best_model.pkl' for the API/Dashboard to find easily
    joblib.dump(best_model, config.MODEL_DIR / "best_model.pkl")
    print("[*] Saved 'best_model.pkl' for downstream use.")