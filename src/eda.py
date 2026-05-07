"""
Exploratory Data Analysis (EDA) module for phishing detection project.

This module provides functions to load and clean the phishing dataset.
"""

import pandas as pd
from config import LABEL_MAP, LABEL_NAMES


def load_data(path: str) -> pd.DataFrame:
    """
    Load and clean the phishing dataset from a CSV file.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Clean dataframe with columns "URL" and "label" (numeric).
                     Nulls have been removed.
    """
    df = pd.read_csv(path)
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    
    if dropped > 0:
        print(f"Dropped {dropped} rows with null values.")
    
    # Rename Label to label (lowercase) and map using LABEL_MAP
    df = df.rename(columns={"Label": "label"})
    df["label"] = df["label"].map(LABEL_MAP)
    
    # Keep only URL and label columns
    df = df[["URL", "label"]]
    
    # Print statistics
    total_rows = len(df)
    class_counts = df["label"].value_counts().sort_index()
    
    print(f"\n--- Dataset Summary ---")
    print(f"Total rows: {total_rows}")
    print(f"\nClass Distribution:")
    for label_id, count in class_counts.items():
        label_name = LABEL_NAMES[label_id]
        percentage = (count / total_rows) * 100
        print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    return df
