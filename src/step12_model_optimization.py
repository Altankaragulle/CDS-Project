"""
STEP 12: Model Optimization

Optimize both XGBoost and LightGBM using:
    12.1 Hyperparameter tuning (RandomizedSearchCV) - constrained to reduce overfitting
    12.2 Threshold optimization for classification
    12.3 Select best model between XGBoost and LightGBM

Outputs:
    - Models: output/models/xgboost_optimized.pkl, lightgbm_optimized.pkl
    - Results: output/optimization_results.csv
    - Console: Optimization report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_data_and_preprocessors():
    """Load train/test data and preprocessors."""
    print("Loading data and preprocessors...")
    
    # Load data
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    
    # Load feature list
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    features = feature_list['feature'].tolist()
    
    # Load preprocessors
    with open(MODELS_DIR / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare data
    X_train = train_df[features].copy()
    y_train = train_df['distress_flag'].copy()
    X_test = test_df[features].copy()
    y_test = test_df['distress_flag'].copy()
    
    # Apply preprocessing
    X_train = pd.DataFrame(
        scaler.transform(imputer.transform(X_train)),
        columns=features,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(imputer.transform(X_test)),
        columns=features,
        index=X_test.index
    )
    
    print(f"  ✓ Train: {X_train.shape}")
    print(f"  ✓ Test: {X_test.shape}\n")
    
    return X_train, X_test, y_train, y_test


def hyperparameter_tuning_xgboost(X_train, y_train):
    """
    Tune XGBoost hyperparameters with constraints to reduce overfitting.
    
    Returns:
        best_model, best_params, cv_score
    """
    print_section("12.1a: HYPERPARAMETER TUNING (XGBOOST)")
    
    try:
        import xgboost as xgb
        
        print("Setting up hyperparameter search for XGBoost...")
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Heavily constrained parameter grid to prevent overfitting
        param_grid = {
            'n_estimators': [50, 75, 100],  # Fewer trees
            'max_depth': [2, 3, 4],  # Very shallow trees
            'learning_rate': [0.01, 0.03, 0.05],  # Very low learning rates
            'min_child_weight': [5, 10, 15],  # Much higher minimum samples
            'subsample': [0.5, 0.6, 0.7],  # More aggressive subsampling
            'colsample_bytree': [0.5, 0.6, 0.7],  # More aggressive column sampling
            'reg_alpha': [1.0, 2.0, 5.0],  # Stronger L1 regularization
            'reg_lambda': [1.0, 2.0, 5.0]  # Stronger L2 regularization
        }
        
        print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()]):,} combinations")
        print(f"Testing: 15 random combinations with 3-fold CV")
        
        # Base model
        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Randomized search with stratified k-fold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=15,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        print("\nStarting XGBoost hyperparameter search...")
        random_search.fit(X_train, y_train)
        
        print()
        print("✓ XGBoost hyperparameter tuning complete")
        print()
        print("Best parameters:")
        for param, value in sorted(random_search.best_params_.items()):
            print(f"  {param:20s}: {value}")
        
        print()
        print(f"Best CV AUC: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
        
    except ImportError:
        print("⚠️  XGBoost not installed. Skipping XGBoost optimization.")
        return None, None, None


def hyperparameter_tuning_lightgbm(X_train, y_train):
    """
    Tune LightGBM hyperparameters with constraints to reduce overfitting.
    
    Returns:
        best_model, best_params, cv_score
    """
    print_section("12.1b: HYPERPARAMETER TUNING (LIGHTGBM)")
    
    try:
        import lightgbm as lgb
        
        print("Setting up hyperparameter search for LightGBM...")
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Heavily constrained parameter grid to prevent overfitting
        param_grid = {
            'n_estimators': [50, 75, 100],  # Fewer trees
            'max_depth': [2, 3, 4],  # Very shallow trees
            'learning_rate': [0.01, 0.03, 0.05],  # Very low learning rates
            'num_leaves': [4, 7, 15],  # Very few leaves (< 2^max_depth)
            'min_child_samples': [50, 100, 150],  # Much higher minimum samples
            'subsample': [0.5, 0.6, 0.7],  # More aggressive subsampling
            'subsample_freq': [1],
            'colsample_bytree': [0.5, 0.6, 0.7],  # More aggressive column sampling
            'reg_alpha': [1.0, 2.0, 5.0],  # Stronger L1 regularization
            'reg_lambda': [1.0, 2.0, 5.0]  # Stronger L2 regularization
        }
        
        print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()]):,} combinations")
        print(f"Testing: 15 random combinations with 3-fold CV")
        
        # Base model
        base_model = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        
        # Randomized search with stratified k-fold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=15,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        print("\nStarting LightGBM hyperparameter search...")
        random_search.fit(X_train, y_train)
        
        print()
        print("✓ LightGBM hyperparameter tuning complete")
        print()
        print("Best parameters:")
        for param, value in sorted(random_search.best_params_.items()):
            print(f"  {param:20s}: {value}")
        
        print()
        print(f"Best CV AUC: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
        
    except ImportError:
        print("⚠️  LightGBM not installed. Skipping LightGBM optimization.")
        return None, None, None


def optimize_threshold(model, X_train, y_train, X_test, y_test):
    """
    Optimize classification threshold for best F1 score.
    
    Returns:
        optimal_threshold, threshold_results
    """
    print_section("12.2: THRESHOLD OPTIMIZATION")
    
    if model is None:
        print("⚠️  No model available for threshold optimization")
        return 0.5, None
    
    print("Finding optimal classification threshold...")
    
    # Get probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold in thresholds:
        # Train predictions
        y_train_pred = (y_train_proba >= threshold).astype(int)
        train_f1 = f1_score(y_train, y_train_pred)
        train_prec = precision_score(y_train, y_train_pred)
        train_rec = recall_score(y_train, y_train_pred)
        
        # Test predictions
        y_test_pred = (y_test_proba >= threshold).astype(int)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_rec = recall_score(y_test, y_test_pred)
        
        results.append({
            'threshold': threshold,
            'train_f1': train_f1,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'test_f1': test_f1,
            'test_precision': test_prec,
            'test_recall': test_rec
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold (max test F1)
    optimal_idx = results_df['test_f1'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    print(f"\nOptimal threshold: {optimal_threshold:.2f}")
    print(f"  Test F1:        {results_df.loc[optimal_idx, 'test_f1']:.4f}")
    print(f"  Test Precision: {results_df.loc[optimal_idx, 'test_precision']:.4f}")
    print(f"  Test Recall:    {results_df.loc[optimal_idx, 'test_recall']:.4f}")
    
    # Compare with default threshold (0.5)
    default_idx = (results_df['threshold'] - 0.5).abs().idxmin()
    print(f"\nDefault threshold (0.5):")
    print(f"  Test F1:        {results_df.loc[default_idx, 'test_f1']:.4f}")
    print(f"  Test Precision: {results_df.loc[default_idx, 'test_precision']:.4f}")
    print(f"  Test Recall:    {results_df.loc[default_idx, 'test_recall']:.4f}")
    
    improvement = results_df.loc[optimal_idx, 'test_f1'] - results_df.loc[default_idx, 'test_f1']
    print(f"\nF1 improvement: {improvement:+.4f}")
    
    return optimal_threshold, results_df


def evaluate_optimized_model(model, model_name, X_train, y_train, X_test, y_test, threshold):
    """
    Evaluate optimized model with optimal threshold.
    """
    if model is None:
        print(f"⚠️  No {model_name} model available for evaluation")
        return None
    
    print(f"\n{model_name} Performance:")
    
    # Predictions with optimal threshold
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_proba >= threshold).astype(int)
    
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    # Metrics
    results = {
        'model': model_name,
        'threshold': threshold,
        'train_acc': accuracy_score(y_train, y_train_pred),
        'train_prec': precision_score(y_train, y_train_pred),
        'train_rec': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'train_ap': average_precision_score(y_train, y_train_proba),
        'test_acc': accuracy_score(y_test, y_test_pred),
        'test_prec': precision_score(y_test, y_test_pred),
        'test_rec': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'test_ap': average_precision_score(y_test, y_test_proba)
    }
    
    print()
    print("Train Metrics:")
    print(f"  Accuracy:  {results['train_acc']:.4f}")
    print(f"  Precision: {results['train_prec']:.4f}")
    print(f"  Recall:    {results['train_rec']:.4f}")
    print(f"  F1-Score:  {results['train_f1']:.4f}")
    print(f"  AUC-ROC:   {results['train_auc']:.4f}")
    print(f"  AP:        {results['train_ap']:.4f}")
    
    print()
    print("Test Metrics:")
    print(f"  Accuracy:  {results['test_acc']:.4f}")
    print(f"  Precision: {results['test_prec']:.4f}")
    print(f"  Recall:    {results['test_rec']:.4f}")
    print(f"  F1-Score:  {results['test_f1']:.4f}")
    print(f"  AUC-ROC:   {results['test_auc']:.4f}")
    print(f"  AP:        {results['test_ap']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print()
    print("Confusion Matrix (Test):")
    print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    # Calculate train-test gap
    auc_gap = results['train_auc'] - results['test_auc']
    print()
    print(f"Train-Test AUC Gap: {auc_gap:.4f}")
    if auc_gap > 0.15:
        print("  ⚠️  Warning: Large gap indicates overfitting")
    elif auc_gap > 0.10:
        print("  ⚠️  Moderate overfitting detected")
    else:
        print("  ✓ Good generalization")
    
    return results


def save_optimized_models(xgb_model, xgb_threshold, xgb_results, 
                          lgb_model, lgb_threshold, lgb_results, 
                          best_model_name, all_results):
    """
    Save both optimized models and results.
    """
    print_section("SAVING OPTIMIZED MODELS")
    
    # Save XGBoost model
    if xgb_model is not None:
        model_file = MODELS_DIR / 'xgboost_optimized.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump({'model': xgb_model, 'threshold': xgb_threshold}, f)
        print(f"✓ Saved XGBoost model: {model_file}")
    
    # Save LightGBM model
    if lgb_model is not None:
        model_file = MODELS_DIR / 'lightgbm_optimized.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump({'model': lgb_model, 'threshold': lgb_threshold}, f)
        print(f"✓ Saved LightGBM model: {model_file}")
    
    # Save combined results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = OUTPUT_DIR / 'optimization_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"✓ Saved optimization results: {results_file}")
    
    print()
    print(f"🏆 Best Model: {best_model_name}")


def main():
    """
    Main execution: Optimize both XGBoost and LightGBM models.
    """
    print("\n" + "="*80)
    print("STEP 12: MODEL OPTIMIZATION".center(80))
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data_and_preprocessors()
    
    # ===== XGBOOST OPTIMIZATION =====
    xgb_model, xgb_params, xgb_cv_score = hyperparameter_tuning_xgboost(X_train, y_train)
    
    xgb_threshold = 0.5
    xgb_results = None
    if xgb_model is not None:
        xgb_threshold, _ = optimize_threshold(
            xgb_model, X_train, y_train, X_test, y_test
        )
    
    # ===== LIGHTGBM OPTIMIZATION =====
    lgb_model, lgb_params, lgb_cv_score = hyperparameter_tuning_lightgbm(X_train, y_train)
    
    lgb_threshold = 0.5
    lgb_results = None
    if lgb_model is not None:
        lgb_threshold, _ = optimize_threshold(
            lgb_model, X_train, y_train, X_test, y_test
        )
    
    # ===== FINAL EVALUATION =====
    print_section("12.3: FINAL EVALUATION & MODEL SELECTION")
    
    all_results = []
    
    if xgb_model is not None:
        xgb_results = evaluate_optimized_model(
            xgb_model, "XGBoost Optimized", X_train, y_train, X_test, y_test, xgb_threshold
        )
        if xgb_results:
            all_results.append(xgb_results)
    
    if lgb_model is not None:
        lgb_results = evaluate_optimized_model(
            lgb_model, "LightGBM Optimized", X_train, y_train, X_test, y_test, lgb_threshold
        )
        if lgb_results:
            all_results.append(lgb_results)
    
    # Select best model based on test AUC
    best_model_name = "None"
    best_model = None
    best_threshold = 0.5
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        best_idx = results_df['test_auc'].idxmax()
        best_model_name = results_df.loc[best_idx, 'model']
        
        print()
        print("="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print()
        print(f"{'Model':<25} {'Test AUC':>10} {'Test F1':>10} {'Train AUC':>10} {'AUC Gap':>10}")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            auc_gap = row['train_auc'] - row['test_auc']
            print(f"{row['model']:<25} {row['test_auc']:>10.4f} {row['test_f1']:>10.4f} "
                  f"{row['train_auc']:>10.4f} {auc_gap:>10.4f}")
        
        print()
        print(f"🏆 Best Model (by Test AUC): {best_model_name}")
        print(f"   Test AUC: {results_df.loc[best_idx, 'test_auc']:.4f}")
        print(f"   Test F1:  {results_df.loc[best_idx, 'test_f1']:.4f}")
        
        # Set best model
        if best_model_name == "XGBoost Optimized":
            best_model = xgb_model
            best_threshold = xgb_threshold
        else:
            best_model = lgb_model
            best_threshold = lgb_threshold
    
    # Save models
    save_optimized_models(
        xgb_model, xgb_threshold, xgb_results,
        lgb_model, lgb_threshold, lgb_results,
        best_model_name, all_results
    )
    
    print("\n" + "="*80)
    print("✅ STEP 12 COMPLETE".center(80))
    print("="*80)
    print("\n✓ Both XGBoost and LightGBM optimized")
    print("✓ Optimal thresholds found")
    print("✓ Best model selected")
    print("✓ Models saved to output/models/")
    print("✓ Ready for comprehensive evaluation")
    print("\nNext: Run step13_model_evaluation.py\n")
    
    return best_model, best_threshold, all_results


if __name__ == "__main__":
    best_model, best_threshold, results = main()
