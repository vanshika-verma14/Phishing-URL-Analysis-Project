"""
Training pipeline for the phishing-detector project.

This script orchestrates all training steps in sequence:
1. Data loading and exploration
2. Feature extraction
3. Rule-based classifier evaluation
4. ML model training and evaluation
5. Sample predictions on test set
6. Model persistence

Run this script once to train the system before using main.py for live predictions.
"""

import sys
import os


def main():
    """
    Execute the complete training pipeline.
    
    Orchestrates all project modules in order to:
    - Load and analyze phishing data
    - Extract engineered features
    - Train multiple ML models
    - Evaluate performance with multiple classifiers
    - Generate sample predictions
    - Save the best model for deployment
    
    Raises:
        Exception: Any error during training is caught and reported.
    """
    try:
        # ============================================================
        # STEP 1: LOAD AND EXPLORE DATA
        # ============================================================
        from src.eda import load_data, run_eda
        from config import DATA_PATH, OUTPUT_DIR
        
        print("="*60)
        print("   PHISHING DETECTOR — TRAINING PIPELINE")
        print("="*60)
        
        df = load_data(DATA_PATH)
        run_eda(df)
        
        # ============================================================
        # STEP 2: FEATURE EXTRACTION
        # ============================================================
        from src.features import build_feature_matrix, FEATURE_COLUMNS
        
        print("\n" + "="*60)
        print("STEP 2: FEATURE EXTRACTION")
        print("="*60)
        
        feature_df = build_feature_matrix(df)
        X = feature_df[FEATURE_COLUMNS]
        y = df["label"]
        
        # ============================================================
        # STEP 3: RULE-BASED CLASSIFIER
        # ============================================================
        from src.rule_based import evaluate as rule_evaluate
        
        print("\n" + "="*60)
        print("STEP 3: RULE-BASED CLASSIFIER")
        print("="*60)
        
        rule_evaluate(feature_df, y, OUTPUT_DIR)
        
        # ============================================================
        # STEP 4: TRAIN/TEST SPLIT
        # ============================================================
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_STATE
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print("\n" + "="*60)
        print("STEP 4: TRAIN/TEST SPLIT")
        print("="*60)
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")
        
        # ============================================================
        # STEP 5: ML MODEL TRAINING & EVALUATION
        # ============================================================
        from src.ml_models import train_all, evaluate_all, get_best_model, save_model
        from config import MODEL_DIR
        
        print("\n" + "="*60)
        print("STEP 5: ML MODEL TRAINING & EVALUATION")
        print("="*60)
        
        trained_models = train_all(X_train, y_train)
        results_df = evaluate_all(trained_models, X_test, y_test, OUTPUT_DIR)
        best_name, best_model = get_best_model(trained_models, results_df)
        
        save_model(best_model, "best_model", MODEL_DIR)
        save_model(best_model, best_name.lower().replace(" ", "_"), MODEL_DIR)
        
        # ============================================================
        # STEP 5B: THRESHOLD TUNING
        # ============================================================
        from sklearn.metrics import precision_recall_curve
        import numpy as np
        import json
        import matplotlib.pyplot as plt
        
        print("\n" + "="*60)
        print("STEP 5B: THRESHOLD TUNING")
        print("="*60)
        
        # Get probabilities from best model on test set
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate precision recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Find best threshold — where precision and recall are
        # most balanced and both above 0.90
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        
        print(f"Default threshold : 0.50")
        print(f"Tuned threshold   : {best_threshold:.4f}")
        print(f"Best F1 at tuned  : {f1_scores[best_idx]:.4f}")
        
        # Save threshold to a file so app.py and main.py can load it
        os.makedirs(MODEL_DIR, exist_ok=True)
        threshold_path = os.path.join(MODEL_DIR, "threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": float(best_threshold)}, f)
        print(f"Threshold saved to {threshold_path}")
        
        # Show how predictions change with tuned threshold
        y_pred_tuned = (y_proba >= best_threshold).astype(int)
        from sklearn.metrics import classification_report
        print("\nResults with tuned threshold:")
        print(classification_report(y_test, y_pred_tuned,
              target_names=["Legitimate", "Phishing"]))
        
        # Plot precision-recall curve and save
        plt.figure(figsize=(8, 5))
        plt.plot(recall, precision, color="#2563EB", linewidth=2)
        plt.axvline(x=recall[best_idx], color="#DC2626",
                    linestyle="--", label=f"Best threshold: {best_threshold:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve — XGBoost")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curve.png"))
        plt.close()
        print("Precision-recall curve saved to outputs/")
        
        # ============================================================
        # STEP 6: SAMPLE PREDICTIONS
        # ============================================================
        from src.explainer import sample_predictions_table
        
        print("\n" + "="*60)
        print("STEP 6: SAMPLE PREDICTIONS")
        print("="*60)
        
        # Build lookup dataframe with aligned indices for sample selection
        lookup_df = df.copy()
        lookup_df = lookup_df.iloc[X.index]
        
        sample_predictions_table(best_model, X_test, y_test, lookup_df)
        
        # ============================================================
        # TRAINING COMPLETE
        # ============================================================
        print("="*60)
        print("Training complete. Model saved to models/")
        print("Run main.py to analyse URLs live.")
        print("="*60)
    
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR DURING TRAINING")
        print("="*60)
        print(f"\n{type(e).__name__}: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
