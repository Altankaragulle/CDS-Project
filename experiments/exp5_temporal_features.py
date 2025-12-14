"""
EXPERIMENT 5: Add Temporal Change Features - Detect Early Deterioration

Goal: Boost recall by adding temporal change features that capture:
    - Structural deterioration (leverage increase, profitability decline)
    - Credit stress (CDS spread increases)
    - Financial distress signals (Altman Z decline)

These features are known to significantly improve early distress detection.

This requires RETRAINING the model with new features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
EXP_OUTPUT_DIR = OUTPUT_DIR / 'experiments'
EXP_MODELS_DIR = EXP_OUTPUT_DIR / 'models'
EXP_FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures' / 'experiments'

# Set style
sns.set_style('whitegrid')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_base_data():
    """Load original train/test data."""
    print("Loading original train/test data...")
    
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    
    print(f"  ✓ Train: {train_df.shape}")
    print(f"  ✓ Test: {test_df.shape}\n")
    
    return train_df, test_df


def create_temporal_change_features(df):
    """
    Create temporal change features to detect deterioration.
    
    Features created:
        - debt_to_assets_change_1y: Change in leverage
        - altman_z_change_1y: Change in distress score
        - roa_change_1y: Change in profitability
        - current_ratio_change_1y: Change in liquidity
        - cds_spread_change_1q: 1-quarter CDS change
        - cds_spread_change_1y: 1-year CDS change
        - equity_return_1q_change: Momentum shift
    """
    print_section("CREATING TEMPORAL CHANGE FEATURES")
    
    print("Computing temporal changes (grouped by firm)...")
    
    # Sort by firm and date
    df = df.sort_values(['gvkey', 'date']).copy()
    
    # Group by firm
    grouped = df.groupby('gvkey')
    
    new_features = []
    
    # 1. Debt-to-Assets change (1 year = 4 quarters)
    if 'debt_to_assets' in df.columns:
        df['debt_to_assets_change_1y'] = grouped['debt_to_assets'].diff(4)
        new_features.append('debt_to_assets_change_1y')
        print("  ✓ debt_to_assets_change_1y")
    
    # 2. Altman Z-Score change
    if 'altman_z_score' in df.columns:
        df['altman_z_change_1y'] = grouped['altman_z_score'].diff(4)
        new_features.append('altman_z_change_1y')
        print("  ✓ altman_z_change_1y")
    
    # 3. ROA change
    if 'roa' in df.columns:
        df['roa_change_1y'] = grouped['roa'].diff(4)
        new_features.append('roa_change_1y')
        print("  ✓ roa_change_1y")
    
    # 4. Current ratio change
    if 'current_ratio' in df.columns:
        df['current_ratio_change_1y'] = grouped['current_ratio'].diff(4)
        new_features.append('current_ratio_change_1y')
        print("  ✓ current_ratio_change_1y")
    
    # 5. CDS spread changes (already lagged, so safe to use)
    if 'cds_spread_lag1' in df.columns and 'cds_spread_lag4' in df.columns:
        # 1-quarter change (lag1 - lag2)
        df['cds_spread_lag2'] = grouped['cds_spread_lag1'].shift(1)
        df['cds_spread_change_1q'] = df['cds_spread_lag1'] - df['cds_spread_lag2']
        new_features.append('cds_spread_change_1q')
        print("  ✓ cds_spread_change_1q")
        
        # 1-year change (lag1 - lag4)
        df['cds_spread_change_1y'] = df['cds_spread_lag1'] - df['cds_spread_lag4']
        new_features.append('cds_spread_change_1y')
        print("  ✓ cds_spread_change_1y")
    
    # 6. Return momentum shift (using return_1m)
    if 'return_1m' in df.columns:
        df['return_1m_prev'] = grouped['return_1m'].shift(1)
        df['return_1m_change'] = df['return_1m'] - df['return_1m_prev']
        new_features.append('return_1m_change')
        print("  ✓ return_1m_change")
    
    # 7. Return lag change
    if 'return_lag1' in df.columns:
        df['return_lag1_change'] = grouped['return_lag1'].diff(1)
        new_features.append('return_lag1_change')
        print("  ✓ return_lag1_change")
    
    # 8. Volatility changes
    if 'volatility_3m' in df.columns:
        df['volatility_3m_change'] = grouped['volatility_3m'].diff(1)
        new_features.append('volatility_3m_change')
        print("  ✓ volatility_3m_change")
    
    if 'volatility_12m' in df.columns:
        df['volatility_12m_change'] = grouped['volatility_12m'].diff(4)
        new_features.append('volatility_12m_change')
        print("  ✓ volatility_12m_change")
    
    print(f"\n✓ Created {len(new_features)} temporal change features")
    
    return df, new_features


def analyze_feature_importance(new_features, train_df, test_df):
    """
    Quick analysis of new features' predictive power.
    """
    print_section("ANALYZING NEW FEATURES")
    
    print("Correlation with distress_flag:")
    print("-" * 50)
    
    correlations = []
    for feature in new_features:
        if feature in train_df.columns:
            corr = train_df[[feature, 'distress_flag']].corr().iloc[0, 1]
            correlations.append({
                'feature': feature,
                'correlation': abs(corr),
                'direction': 'positive' if corr > 0 else 'negative'
            })
            print(f"  {feature:<35} {corr:>8.4f}")
    
    print("-" * 50)
    
    return pd.DataFrame(correlations).sort_values('correlation', ascending=False)


def train_model_with_temporal_features(train_df, test_df, original_features, new_features):
    """
    Train LightGBM model with original + temporal features.
    """
    print_section("TRAINING MODEL WITH TEMPORAL FEATURES")
    
    try:
        import lightgbm as lgb
    except ImportError:
        print("❌ LightGBM not installed")
        return None, None
    
    # Combine features
    all_features = original_features + new_features
    
    # Load preprocessors
    with open(MODELS_DIR / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare data
    X_train = train_df[all_features].copy()
    y_train = train_df['distress_flag'].copy()
    X_test = test_df[all_features].copy()
    y_test = test_df['distress_flag'].copy()
    
    print(f"Training data: {X_train.shape}")
    print(f"Features: {len(all_features)} ({len(original_features)} original + {len(new_features)} temporal)")
    
    # Refit preprocessors on new feature set
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    imputer_new = SimpleImputer(strategy='median')
    scaler_new = StandardScaler()
    
    X_train_processed = scaler_new.fit_transform(imputer_new.fit_transform(X_train))
    X_test_processed = scaler_new.transform(imputer_new.transform(X_test))
    
    X_train_processed = pd.DataFrame(X_train_processed, columns=all_features, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=all_features, index=X_test.index)
    
    # Train model (Medium Regularization config)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print(f"\nTraining LightGBM (Medium Regularization + Temporal Features)...")
    
    model = lgb.LGBMClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=20,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=50,
        reg_alpha=0.3,
        reg_lambda=0.3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_processed)
    y_train_proba = model.predict_proba(X_train_processed)[:, 1]
    
    y_test_pred = model.predict(X_test_processed)
    y_test_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Metrics
    results = {
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred)
    }
    
    print(f"\n✓ Training complete")
    print(f"\nPerformance:")
    print(f"  Train AUC: {results['train_auc']:.4f} | Test AUC: {results['test_auc']:.4f}")
    print(f"  Train Recall: {results['train_recall']:.4f} | Test Recall: {results['test_recall']:.4f}")
    print(f"  Train F1: {results['train_f1']:.4f} | Test F1: {results['test_f1']:.4f}")
    
    # Save model
    model_file = EXP_MODELS_DIR / 'lightgbm_medium_regularization_temporal_features.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved: {model_file}")
    
    # Save preprocessors
    with open(EXP_MODELS_DIR / 'imputer_temporal.pkl', 'wb') as f:
        pickle.dump(imputer_new, f)
    with open(EXP_MODELS_DIR / 'scaler_temporal.pkl', 'wb') as f:
        pickle.dump(scaler_new, f)
    
    # Save feature list
    feature_df = pd.DataFrame({'feature': all_features})
    feature_df.to_csv(EXP_OUTPUT_DIR / 'ml_feature_list_temporal.csv', index=False)
    
    return model, results, X_test_processed, y_test, y_test_proba


def compare_models(original_results, temporal_results):
    """
    Compare original vs temporal feature models.
    """
    print_section("MODEL COMPARISON: ORIGINAL VS TEMPORAL FEATURES")
    
    print("PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Original':<15} {'+ Temporal':<15} {'Improvement':<15}")
    print("-" * 80)
    
    metrics = ['test_auc', 'test_recall', 'test_precision', 'test_f1']
    metric_names = ['AUC', 'Recall', 'Precision', 'F1-Score']
    
    for metric, name in zip(metrics, metric_names):
        orig = original_results[metric]
        temp = temporal_results[metric]
        improvement = temp - orig
        pct_change = (improvement / orig) * 100 if orig != 0 else 0
        
        marker = "✅" if improvement > 0 else "⚠️" if improvement == 0 else "❌"
        print(f"{marker} {name:<18} {orig:<15.4f} {temp:<15.4f} {improvement:>+7.4f} ({pct_change:+.1f}%)")
    
    print("-" * 80)
    
    # Highlight recall improvement
    recall_improvement = temporal_results['test_recall'] - original_results['test_recall']
    
    print(f"\n🎯 RECALL IMPROVEMENT: {recall_improvement:+.4f} ({recall_improvement/original_results['test_recall']*100:+.1f}%)")
    
    if temporal_results['test_recall'] >= 0.55:
        print(f"   ✅ TARGET ACHIEVED: Recall >= 0.55 ({temporal_results['test_recall']:.3f})")
    elif temporal_results['test_recall'] >= 0.50:
        print(f"   ⚠️  CLOSE: Recall = {temporal_results['test_recall']:.3f} (target: 0.55)")
    else:
        print(f"   ❌ Below target: Recall = {temporal_results['test_recall']:.3f}")


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance highlighting temporal features.
    """
    print_section("GENERATING FEATURE IMPORTANCE PLOT")
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)
    
    # Identify temporal features
    temporal_keywords = ['change', 'diff']
    feature_importance_df['is_temporal'] = feature_importance_df['feature'].apply(
        lambda x: any(kw in x.lower() for kw in temporal_keywords)
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['green' if is_temp else 'steelblue' 
              for is_temp in feature_importance_df['is_temporal']]
    
    bars = ax.barh(feature_importance_df['feature'], feature_importance_df['importance'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Feature Importance (Temporal Features in Green)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Temporal Change Features'),
        Patch(facecolor='steelblue', alpha=0.7, edgecolor='black', label='Original Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    output_file = EXP_FIGURES_DIR / 'feature_importance_temporal.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def main():
    """
    Main execution: Temporal features experiment.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 5: TEMPORAL CHANGE FEATURES".center(80))
    print("="*80)
    
    # Load data
    train_df, test_df = load_base_data()
    
    # Load original feature list
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    original_features = feature_list['feature'].tolist()
    
    print(f"Original features: {len(original_features)}")
    
    # Create temporal features
    train_df, new_features = create_temporal_change_features(train_df)
    test_df, _ = create_temporal_change_features(test_df)
    
    # Analyze new features
    feature_correlations = analyze_feature_importance(new_features, train_df, test_df)
    
    # Load original model results for comparison
    original_results = {
        'test_auc': 0.6431,
        'test_recall': 0.4641,
        'test_precision': 0.3422,
        'test_f1': 0.3940
    }
    
    # Train model with temporal features
    model, temporal_results, X_test, y_test, y_test_proba = train_model_with_temporal_features(
        train_df, test_df, original_features, new_features
    )
    
    if model is not None:
        # Compare models
        compare_models(original_results, temporal_results)
        
        # Plot feature importance
        plot_feature_importance(model, original_features + new_features)
        
        # Save results
        results_df = pd.DataFrame([
            {'model': 'Original (Medium Reg)', **original_results},
            {'model': 'Medium Reg + Temporal', **temporal_results}
        ])
        
        results_file = EXP_OUTPUT_DIR / 'temporal_features_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved: {results_file}")
    
    print("\n" + "="*80)
    print("✅ TEMPORAL FEATURES EXPERIMENT COMPLETE".center(80))
    print("="*80)
    print(f"\n✓ New features created: {len(new_features)}")
    print(f"✓ Model saved: {EXP_MODELS_DIR}")
    print(f"✓ Figures saved: {EXP_FIGURES_DIR}")
    print("\nNext: Combine with threshold optimization (exp4) for maximum recall!\n")


if __name__ == "__main__":
    main()
