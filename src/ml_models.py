"""
ML models module for phishing detection.

This module handles training, evaluation, saving, and loading of all machine learning
models. It is completely configuration-driven and uses paths from config.py.
"""

import os
import time
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from config import RANDOM_STATE, MODEL_DIR

# Define all models to train
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
}


def train_all(X_train, y_train) -> dict:
    """
    Train all ML models in the MODELS dictionary.
    
    Trains each model on the provided training data and tracks execution time
    for performance monitoring.
    
    Args:
        X_train: Feature matrix for training (from train_test_split)
        y_train: Label vector for training (from train_test_split)
        
    Returns:
        dict: Mapping of model_name to fitted model object
    """
    trained_models = {}
    
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    
    for name, model in MODELS.items():
        print(f"\nTraining {name}...", end=" ", flush=True)
        start_time = time.time()
        
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        elapsed = time.time() - start_time
        print(f"Done ({elapsed:.2f}s)")
    
    return trained_models


def evaluate_model(name: str, model, X_test, y_test, output_dir: str) -> dict:
    """
    Evaluate a single model and generate metrics and visualizations.
    
    Computes accuracy, precision, recall, and F1-score. Generates a confusion
    matrix heatmap and saves it to disk.
    
    Args:
        name (str): Name of the model (used for headers and filenames)
        model: Fitted model object with predict() method
        X_test: Feature matrix for testing
        y_test: True labels for testing
        output_dir (str): Directory to save confusion matrix plot
        
    Returns:
        dict: Keys are model_name, accuracy, precision, recall, f1
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, 
                                        target_names=["Legitimate", "Phishing"],
                                        output_dict=True)
    
    # Extract weighted averages
    precision = report_dict["weighted avg"]["precision"]
    recall = report_dict["weighted avg"]["recall"]
    f1 = report_dict["weighted avg"]["f1-score"]
    
    # Print results
    print("\n" + "="*60)
    print(f"{name.upper()}")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\n" + classification_report(y_test, y_pred, 
                                      target_names=["Legitimate", "Phishing"]))
    
    # Save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    safe_name = name.lower().replace(" ", "_")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.title(name, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{safe_name}_confusion_matrix.png")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")
    
    return {
        "model_name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_all(trained_models: dict, X_test, y_test, output_dir: str) -> pd.DataFrame:
    """
    Evaluate all trained models and generate a comparison table.
    
    Evaluates each model independently and produces a formatted summary table
    for easy comparison of performance metrics.
    
    Args:
        trained_models (dict): Mapping of model_name to fitted model object
        X_test: Feature matrix for testing
        y_test: True labels for testing
        output_dir (str): Directory to save confusion matrix plots
        
    Returns:
        pd.DataFrame: Comparison table with metrics for all models
    """
    results = []
    
    for name, model in trained_models.items():
        result = evaluate_model(name, model, X_test, y_test, output_dir)
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print("\n")
    
    # Round to 4 decimal places and display
    display_df = results_df.copy()
    display_df["Accuracy"] = display_df["accuracy"].round(4)
    display_df["Precision"] = display_df["precision"].round(4)
    display_df["Recall"] = display_df["recall"].round(4)
    display_df["F1-Score"] = display_df["f1"].round(4)
    
    summary = display_df[["model_name", "Accuracy", "Precision", "Recall", "F1-Score"]]
    summary.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    print(summary.to_string(index=False))
    print()
    
    return results_df


def get_best_model(trained_models: dict, results_df: pd.DataFrame) -> tuple[str, object]:
    """
    Identify and return the best performing model.
    
    Selects the model with the highest accuracy from the evaluation results.
    
    Args:
        trained_models (dict): Mapping of model_name to fitted model object
        results_df (pd.DataFrame): Results dataframe from evaluate_all()
        
    Returns:
        tuple: (best_model_name, best_model_object)
    """
    best_idx = results_df["accuracy"].idxmax()
    best_name = results_df.loc[best_idx, "model_name"]
    best_accuracy = results_df.loc[best_idx, "accuracy"]
    
    print(f"Best Model: {best_name} with accuracy {best_accuracy:.4f}")
    
    best_model = trained_models[best_name]
    return best_name, best_model


def save_model(model, name: str, model_dir: str) -> None:
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: Fitted model object to save
        name (str): Name of the model (used in filename)
        model_dir (str): Directory to save the model file
    """
    os.makedirs(model_dir, exist_ok=True)
    
    safe_name = name.lower().replace(" ", "_")
    output_path = os.path.join(model_dir, f"{safe_name}.pkl")
    
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


def load_model(name: str, model_dir: str):
    """
    Load a trained model from disk using joblib.
    
    Args:
        name (str): Name of the model (used in filename)
        model_dir (str): Directory where the model file is stored
        
    Returns:
        Fitted model object
        
    Raises:
        FileNotFoundError: If the model file does not exist
    """
    safe_name = name.lower().replace(" ", "_")
    model_path = os.path.join(model_dir, f"{safe_name}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{name}' not found at {model_path}. Run train.py first."
        )
    
    model = joblib.load(model_path)
    return model
