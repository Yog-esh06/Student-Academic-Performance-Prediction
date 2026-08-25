"""Loads raw data from disk."""
import pandas as pd
from src import config
from src.data.clean_data import clean_data

def load_raw_data() -> pd.DataFrame:
    """
    Loads raw data from the configured path and validates its structure.
    
    Returns:
        pd.DataFrame: The raw dataframe.
        
    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If the target column is missing.
    """
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at: {config.RAW_DATA_PATH}")
    
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' missing from data.")
        
    return df

if __name__ == "__main__":
    print("--- Loading Raw Data ---")
    try:
        raw_df = load_raw_data()
        print(f"Raw data shape: {raw_df.shape}")
        
        print("\n--- Cleaning Data ---")
        cleaned_df = clean_data(raw_df)
        
        print("\n--- Cleaned Data Info ---")
        cleaned_df.info()
        
        print("\n--- Cleaned Data Describe ---")
        print(cleaned_df.describe(include='all'))
        
    except Exception as e:
        print(f"Error: {e}")