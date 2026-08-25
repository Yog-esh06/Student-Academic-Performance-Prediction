"""Encodes, scales, and splits data."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from src import config
from src.data.load_data import load_raw_data
from src.data.clean_data import clean_data

# Hardcoded from the EDA output to ensure consistency
CATEGORICAL_COLS = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

NUMERIC_COLS = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
    'Tutoring_Sessions', 'Physical_Activity'
]

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Returns a scikit-learn ColumnTransformer that scales numeric 
    features and one-hot encodes categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ],
        remainder='drop' # Drops any columns not explicitly defined above (like the target)
    )
    return preprocessor

def get_train_test_split(df: pd.DataFrame):
    """
    Splits the dataframe into X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    
    return train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )

def save_processed_data(df: pd.DataFrame):
    """
    Saves the cleaned, pre-split dataframe to the processed data directory.
    (Saved as CSV since pyarrow/parquet wasn't explicitly added to requirements)
    """
    save_path = config.PROCESSED_DATA_PATH.with_suffix('.csv')
    df.to_csv(save_path, index=False)
    print(f"[*] Saved cleaned data to {save_path}")

if __name__ == "__main__":
    print("--- Starting Preprocessing ---")
    
    # 1. Load and Clean
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    
    # 2. Save cleaned dataset
    save_processed_data(df_clean)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = get_train_test_split(df_clean)
    
    # 4. Build and Fit Preprocessor (Fit ONLY on X_train to prevent data leakage)
    preprocessor = build_preprocessing_pipeline()
    
    print("\n[*] Fitting preprocessor on X_train...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Extract the new column names after One-Hot Encoding
    feature_names = preprocessor.get_feature_names_out()
    
    # 5. Save the fitted preprocessor
    preprocessor_path = config.MODEL_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[*] Saved fitted preprocessor to {preprocessor_path}")
    
    print("\n--- Preprocessing Results ---")
    print(f"X_train shape: {X_train.shape} -> Transformed: {X_train_transformed.shape}")
    print(f"X_test shape:  {X_test.shape} -> Transformed: {X_test_transformed.shape}")
    print(f"Total features after encoding: {len(feature_names)}")
    print("\nSample feature names:")
    print(list(feature_names)[:10] + ["..."])