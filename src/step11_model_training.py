"""
STEP 11: Model Training

Train multiple ML models for corporate distress prediction:
    11.1 Prepare data (imputation, scaling)
    11.2 Train baseline models (Logistic Regression, Random Forest)
    11.3 Train advanced models (XGBoost, LightGBM)
    11.4 Compare model performance

Models:
    - Logistic Regression (interpretable baseline)
    - Random Forest (ensemble baseline)
    - XGBoost (gradient boosting)
    - LightGBM (fast gradient boosting)

Outputs:
    - Models: output/models/*.pkl
    - Results: output/model_results.csv
    - Console: Training report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_train_test_data():
    """Load train and test datasets."""
    print("Loading train and test datasets...")
    
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    
    # Load feature list
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    features = feature_list['feature'].tolist()
    
    print(f"  ✓ Train: {len(train_df):,} rows")
    print(f"  ✓ Test: {len(test_df):,} rows")
    print(f"  ✓ Features: {len(features)}\n")
    
    return train_df, test_df, features


def prepare_data(train_df, test_df, features):
    """
    Prepare data for modeling.
    
    Steps:
        - Extract X (features) and y (target)
        - Impute missing values (median strategy)
        - Scale features (standardization)
    
    Returns:
        X_train, X_test, y_train, y_test, imputer, scaler
    """
    print_section("11.1: PREPARING DATA (IMPUTATION & SCALING)")
    
    # Extract features and target
    X_train = train_df[features].copy()
    y_train = train_df['distress_flag'].copy()
    
    X_test = test_df[features].copy()
    y_test = test_df['distress_flag'].copy()
    
    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Check missing values
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    print(f"\nMissing values before imputation:")
    print(f"  Train: {train_missing:,}")
    print(f"  Test: {test_missing:,}")
    
    # Impute missing values (median strategy)
    print("\nImputing missing values (median strategy)...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_imputed, columns=features, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=features, index=X_test.index)
    
    print(f"  ✓ Imputation complete")
    
    # Scale features (standardization)
    print("\nScaling features (standardization)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    
    print(f"  ✓ Scaling complete")
    
    print()
    print("✓ Data preparation complete")
    
    return X_train, X_test, y_train, y_test, imputer, scaler


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression model.
    
    Returns:
        model, results_dict
    """
    print_section("11.2: TRAINING LOGISTIC REGRESSION")
    
    print("Training Logistic Regression with class weights...")
    
    # Train model with class weights and L2 regularization to reduce overfitting
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs',
        C=0.1,  # Strong L2 regularization (inverse of regularization strength)
        penalty='l2'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    results = evaluate_model("Logistic Regression", y_train, y_train_pred, y_train_proba,
                            y_test, y_test_pred, y_test_proba)
    
    # Save model
    model_file = MODELS_DIR / 'logistic_regression.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved: {model_file}")
    
    return model, results


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model.
    
    Returns:
        model, results_dict
    """
    print_section("11.3: TRAINING RANDOM FOREST")
    
    print("Training Random Forest with class weights...")
    
    # Train model with stronger constraints to reduce overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Reduced from 10 to prevent deep trees
        min_samples_split=50,  # Increased from 20
        min_samples_leaf=25,  # Increased from 10
        max_features='sqrt',  # Limit features per split
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    results = evaluate_model("Random Forest", y_train, y_train_pred, y_train_proba,
                            y_test, y_test_pred, y_test_proba)
    
    # Save model
    model_file = MODELS_DIR / 'random_forest.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved: {model_file}")
    
    return model, results


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model.
    
    Returns:
        model, results_dict
    """
    print_section("11.4: TRAINING XGBOOST")
    
    try:
        import xgboost as xgb
        
        print("Training XGBoost with scale_pos_weight...")
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model with very strong regularization to reduce overfitting
        model = xgb.XGBClassifier(
            n_estimators=75,  # Fewer trees
            max_depth=3,  # Shallow trees
            learning_rate=0.03,  # Very low learning rate
            min_child_weight=10,  # High minimum samples per leaf
            subsample=0.6,  # Aggressive row sampling
            colsample_bytree=0.6,  # Aggressive column sampling
            reg_alpha=2.0,  # Strong L1 regularization
            reg_lambda=2.0,  # Strong L2 regularization
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = evaluate_model("XGBoost", y_train, y_train_pred, y_train_proba,
                                y_test, y_test_pred, y_test_proba)
        
        # Save model
        model_file = MODELS_DIR / 'xgboost.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✓ Model saved: {model_file}")
        
        return model, results
        
    except ImportError:
        print("⚠️  XGBoost not installed. Skipping...")
        print("   Install with: pip install xgboost")
        return None, None


def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    Train LightGBM model.
    
    Returns:
        model, results_dict
    """
    print_section("11.5: TRAINING LIGHTGBM")
    
    try:
        import lightgbm as lgb
        
        print("Training LightGBM with class weights...")
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model with very strong regularization to reduce overfitting
        model = lgb.LGBMClassifier(
            n_estimators=75,  # Fewer trees
            max_depth=3,  # Shallow trees
            learning_rate=0.03,  # Very low learning rate
            num_leaves=7,  # Very few leaves (< 2^max_depth)
            min_child_samples=100,  # High minimum samples per leaf
            subsample=0.6,  # Aggressive row sampling
            subsample_freq=1,  # Apply subsample every iteration
            colsample_bytree=0.6,  # Aggressive column sampling
            reg_alpha=2.0,  # Strong L1 regularization
            reg_lambda=2.0,  # Strong L2 regularization
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = evaluate_model("LightGBM", y_train, y_train_pred, y_train_proba,
                                y_test, y_test_pred, y_test_proba)
        
        # Save model
        model_file = MODELS_DIR / 'lightgbm.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✓ Model saved: {model_file}")
        
        return model, results
        
    except ImportError:
        print("⚠️  LightGBM not installed. Skipping...")
        print("   Install with: pip install lightgbm")
        return None, None


def evaluate_model(model_name, y_train, y_train_pred, y_train_proba, 
                   y_test, y_test_pred, y_test_proba):
    """
    Evaluate model performance.
    
    Returns:
        Dictionary with metrics
    """
    # Train metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred)
    train_rec = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_ap = average_precision_score(y_train, y_train_proba)
    
    # Test metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_ap = average_precision_score(y_test, y_test_proba)
    
    # Confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"\nTrain Metrics:")
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall:    {train_rec:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    print(f"  AUC-ROC:   {train_auc:.4f}")
    print(f"  AP:        {train_ap:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  AUC-ROC:   {test_auc:.4f}")
    print(f"  AP:        {test_ap:.4f}")
    
    print(f"\nConfusion Matrix (Test):")
    print(f"  TN: {cm_test[0,0]:5d}  FP: {cm_test[0,1]:5d}")
    print(f"  FN: {cm_test[1,0]:5d}  TP: {cm_test[1,1]:5d}")
    
    return {
        'model': model_name,
        'train_acc': train_acc,
        'train_prec': train_prec,
        'train_rec': train_rec,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'train_ap': train_ap,
        'test_acc': test_acc,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'test_ap': test_ap
    }


def compare_models(all_results):
    """
    Compare all trained models.
    """
    print_section("MODEL COMPARISON")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Sort by test AUC
    results_df = results_df.sort_values('test_auc', ascending=False)
    
    print("Model Performance Summary (sorted by Test AUC):")
    print()
    print(f"{'Model':<20} {'Test AUC':>10} {'Test F1':>10} {'Test Prec':>10} {'Test Rec':>10}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:<20} {row['test_auc']:>10.4f} {row['test_f1']:>10.4f} "
              f"{row['test_prec']:>10.4f} {row['test_rec']:>10.4f}")
    
    # Save results
    results_file = OUTPUT_DIR / 'model_results.csv'
    results_df.to_csv(results_file, index=False)
    print()
    print(f"✓ Results saved: {results_file}")
    
    # Best model
    best_model = results_df.iloc[0]
    print()
    print(f"🏆 Best Model: {best_model['model']}")
    print(f"   Test AUC: {best_model['test_auc']:.4f}")
    print(f"   Test F1:  {best_model['test_f1']:.4f}")
    
    return results_df


def main():
    """
    Main execution: Train all models.
    """
    print("\n" + "="*80)
    print("STEP 11: MODEL TRAINING".center(80))
    print("="*80)
    
    # Load data
    train_df, test_df, features = load_train_test_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, imputer, scaler = prepare_data(
        train_df, test_df, features
    )
    
    # Save preprocessors
    with open(MODELS_DIR / 'imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train models
    all_results = []
    
    # Logistic Regression
    lr_model, lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    if lr_results:
        all_results.append(lr_results)
    
    # Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    if rf_results:
        all_results.append(rf_results)
    
    # XGBoost
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_results:
        all_results.append(xgb_results)
    
    # LightGBM
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_test, y_test)
    if lgb_results:
        all_results.append(lgb_results)
    
    # Compare models
    results_df = compare_models(all_results)
    
    print("\n" + "="*80)
    print("✅ STEP 11 COMPLETE".center(80))
    print("="*80)
    print(f"\n✓ Trained {len(all_results)} models successfully")
    print("✓ Models saved to output/models/")
    print("✓ Results saved to output/model_results.csv")
    print("✓ Ready for model optimization")
    print("\nNext: Run step12_model_optimization.py\n")
    
    return results_df


if __name__ == "__main__":
    results_df = main()
