"""Trains regressors and performs CV."""
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score

from src import config
from src.data.load_data import load_raw_data
from src.data.clean_data import clean_data
from src.data.preprocess import get_train_test_split, build_preprocessing_pipeline

def get_models() -> dict:
    """Returns a dictionary of the 5 untrained regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=config.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=config.RANDOM_STATE)
    }

def train_and_cv_all(models_dict: dict, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Runs 5-fold CV for each model and returns their mean RMSE scores."""
    cv_results = {}
    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    print("\n[*] Running 5-Fold Cross Validation...")
    for name, model in models_dict.items():
        # Using neg_mean_squared_error, so we multiply by -1 before taking sqrt
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        rmse_scores = np.sqrt(-scores)
        cv_results[name] = rmse_scores
        print(f"    - {name}: Mean CV RMSE = {rmse_scores.mean():.4f} (std: {rmse_scores.std():.4f})")
        
    return cv_results

def fit_final_models(models_dict: dict, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Fits all models on the full training dataset."""
    print("\n[*] Fitting final models on full training set...")
    fitted_models = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        print(f"    - {name} fitted.")
    return fitted_models

def save_model(model, name: str):
    """Saves a fitted model to the models_store directory."""
    path = config.MODEL_DIR / f"{name.replace(' ', '_')}.pkl"
    joblib.dump(model, path)

if __name__ == "__main__":
    print("--- Model Training Pipeline ---")
    
    # 1. Load, clean, and split data
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    # 2. Fit and transform training data
    preprocessor = build_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Save the preprocessor again just to be safe
    joblib.dump(preprocessor, config.MODEL_DIR / "preprocessor.pkl")
    
    # 3. Initialize models
    models = get_models()
    
    # 4. Cross-validation
    cv_scores = train_and_cv_all(models, X_train_transformed, y_train)
    
    # 5. Fit on full training data
    fitted_models = fit_final_models(models, X_train_transformed, y_train)
    
    # 6. Save all models
    print("\n[*] Saving models to disk...")
    for name, model in fitted_models.items():
        save_model(model, name)
    print("[*] All models saved successfully.")