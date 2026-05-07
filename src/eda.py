"""
Exploratory Data Analysis (EDA) module for phishing detection project.

This module provides functions to load, explore, and visualize the phishing dataset.
It is used during the training pipeline to understand data characteristics and distributions.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import DATA_PATH, OUTPUT_DIR, LABEL_MAP, LABEL_NAMES


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


def print_samples(df: pd.DataFrame) -> None:
    """
    Print sample URLs from each class for quick inspection.
    
    Args:
        df (pd.DataFrame): Dataframe with "URL" and "label" columns.
    """
    print("\n--- Sample Legitimate URLs ---")
    legitimate_samples = df[df["label"] == 0]["URL"].head(3)
    for idx, url in enumerate(legitimate_samples, 1):
        try:
            truncated = url[:80] if len(url) > 80 else url
            print(f"  {idx}. {truncated}...")
        except Exception:
            print(f"  {idx}. [URL with special characters]")
    
    print("\n--- Sample Phishing URLs ---")
    phishing_samples = df[df["label"] == 1]["URL"].head(3)
    for idx, url in enumerate(phishing_samples, 1):
        try:
            truncated = url[:80] if len(url) > 80 else url
            print(f"  {idx}. {truncated}...")
        except Exception:
            print(f"  {idx}. [URL with special characters]")


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Create and save a bar plot of class distribution.
    
    Args:
        df (pd.DataFrame): Dataframe with "URL" and "label" columns.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    class_counts = df["label"].value_counts().sort_index()
    class_names = [LABEL_NAMES[label_id] for label_id in class_counts.index]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_names, y=class_counts.values, palette="Set2")
    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    print(f"\nClass distribution plot saved to {output_path}")


def run_eda(df: pd.DataFrame) -> None:
    """
    Run the complete exploratory data analysis pipeline.
    
    This is the main entry point for EDA. It calls all exploration functions
    in sequence and generates a comprehensive data overview.
    
    Args:
        df (pd.DataFrame): Dataframe with "URL" and "label" columns.
    """
    print_samples(df)
    plot_class_distribution(df)
