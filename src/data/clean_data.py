"""Cleans data (missing values, duplicates, outliers)."""
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe by handling missing values, duplicates, and flagging outliers.
    
    Args:
        df (pd.DataFrame): The raw dataframe.
        
    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    df_clean = df.copy()
    
    # 1. Handle Duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_dupes = initial_rows - len(df_clean)
    if dropped_dupes > 0:
        print(f"[*] Dropped {dropped_dupes} duplicate rows.")
    else:
        print("[*] No duplicate rows found.")
        
    # 2. Handle Missing Values
    missing_summary = df_clean.isnull().sum()
    cols_with_missing = missing_summary[missing_summary > 0]
    
    if not cols_with_missing.empty:
        print("[*] Imputing missing values:")
        for col in cols_with_missing.index:
            missing_count = cols_with_missing[col]
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"    - {col} (Numeric): Imputed {missing_count} rows with median ({median_val:.2f}).")
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                print(f"    - {col} (Categorical): Imputed {missing_count} rows with mode ('{mode_val}').")
    else:
        print("[*] No missing values found.")
        
    # 3. Detect Outliers (IQR Method)
    print("[*] Outlier Detection (IQR Method) - Flagging only:")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    outliers_found = False
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        if not outliers.empty:
            outliers_found = True
            print(f"    - {col}: Found {len(outliers)} outliers (Bounds: {lower_bound:.2f} to {upper_bound:.2f}).")
            
    if not outliers_found:
        print("    - No significant outliers detected in numeric columns.")
        
    return df_clean