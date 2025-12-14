"""
STEP 15: Model Explainability

Explain optimized model predictions (XGBoost & LightGBM) using:
    15.1 SHAP values (global and local explanations)
    15.2 Feature interactions
    15.3 Example predictions analysis
    15.4 Model comparison

Outputs:
    - Figures: report/figures/step15_*.png
    - Report: output/shap_values_*.csv
    - Console: Explainability analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures'

# Set style
sns.set_style('whitegrid')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_models_and_data():
    """Load both optimized models and test data."""
    print("Loading optimized models and test data...")
    
    # Load test data
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    
    # Load feature list
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    features = feature_list['feature'].tolist()
    
    # Load preprocessors
    with open(MODELS_DIR / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    models = {}
    
    # Load XGBoost Optimized
    try:
        with open(MODELS_DIR / 'xgboost_optimized.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            models['XGBoost'] = model_dict['model']
        print(f"  ✓ XGBoost Optimized loaded")
    except FileNotFoundError:
        print(f"  ⚠️  XGBoost Optimized not found, skipping...")
    
    # Load LightGBM Optimized
    try:
        with open(MODELS_DIR / 'lightgbm_optimized.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            models['LightGBM'] = model_dict['model']
        print(f"  ✓ LightGBM Optimized loaded")
    except FileNotFoundError:
        print(f"  ⚠️  LightGBM Optimized not found, skipping...")
    
    # Prepare test data
    X_test = test_df[features].copy()
    y_test = test_df['distress_flag'].copy()
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(imputer.transform(X_test)),
        columns=features,
        index=X_test.index
    )
    
    print(f"  ✓ Loaded {len(models)} optimized models")
    print(f"  ✓ Test data: {X_test_scaled.shape}\n")
    
    return models, X_test_scaled, y_test, test_df, features


def compute_shap_values(model, X_test, features, model_name='Model'):
    """
    Compute SHAP values for model explanations.
    
    Returns:
        shap_values, explainer, X_sample
    """
    try:
        import shap
        
        print(f"\nComputing SHAP values for {model_name}...")
        
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values on a sample (for speed)
        sample_size = min(1000, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        
        print(f"  Analyzing {sample_size} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, take the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        print(f"  ✓ SHAP values computed for {model_name}")
        print(f"    Shape: {shap_values.shape}")
        
        return shap_values, explainer, X_sample
        
    except ImportError:
        print(f"  ⚠️  SHAP not installed. Skipping SHAP analysis for {model_name}.")
        print("     Install with: pip install shap")
        return None, None, None
    except Exception as e:
        print(f"  ⚠️  Error computing SHAP for {model_name}: {e}")
        return None, None, None


def plot_shap_summary(shap_values, X_sample, features, model_name='Model'):
    """
    Create SHAP summary plots for a specific model.
    """
    if shap_values is None:
        print(f"  ⚠️  SHAP values not available for {model_name}")
        return
    
    try:
        import shap
        
        # 1. Summary plot (beeswarm)
        print(f"  Creating SHAP summary plot for {model_name}...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=features, show=False, max_display=20)
        plt.title(f'{model_name} - SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        model_suffix = model_name.lower().replace(' ', '_')
        output_file = FIGURES_DIR / f'step15_shap_summary_{model_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
        
        # 2. Bar plot (mean absolute SHAP values)
        print(f"  Creating SHAP importance plot for {model_name}...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', show=False, max_display=20)
        plt.title(f'{model_name} - SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        output_file = FIGURES_DIR / f'step15_shap_importance_{model_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"  ⚠️  Error creating SHAP plots for {model_name}: {e}")


def analyze_feature_interactions(shap_values, X_sample, features):
    """
    Analyze feature interactions using SHAP.
    """
    print_section("15.2: FEATURE INTERACTIONS")
    
    if shap_values is None:
        print("⚠️  SHAP values not available")
        return
    
    try:
        import shap
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top 5 features
        top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
        top_features = [features[i] for i in top_indices]
        
        print("Top 5 Most Important Features (by SHAP):")
        for i, (idx, feat) in enumerate(zip(top_indices, top_features), 1):
            print(f"  {i}. {feat:30s}: {mean_abs_shap[idx]:.4f}")
        
        # Create dependence plots for top 2 features
        print("\nCreating SHAP dependence plots for top features...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, (idx, feat) in enumerate(zip(top_indices[:2], top_features[:2])):
            ax = axes[i]
            
            # Scatter plot
            scatter = ax.scatter(X_sample.iloc[:, idx], shap_values[:, idx], 
                               c=shap_values[:, idx], cmap='RdYlGn_r', 
                               alpha=0.6, s=20)
            
            ax.set_xlabel(feat, fontsize=11, fontweight='bold')
            ax.set_ylabel('SHAP Value', fontsize=11, fontweight='bold')
            ax.set_title(f'SHAP Dependence: {feat}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='SHAP Value')
        
        plt.tight_layout()
        
        output_file = FIGURES_DIR / 'step15_shap_dependence.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Error analyzing interactions: {e}")


def analyze_example_predictions(model, X_test, y_test, test_df, features):
    """
    Analyze specific example predictions.
    """
    print_section("15.3: EXAMPLE PREDICTIONS ANALYSIS")
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Find interesting examples
    print("Finding interesting prediction examples...\n")
    
    # 1. True Positives (correctly predicted distress)
    tp_mask = (y_test == 1) & (y_pred == 1)
    tp_indices = y_test[tp_mask].index[:3]
    
    # 2. False Positives (incorrectly predicted distress)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_indices = y_test[fp_mask].index[:3]
    
    # 3. False Negatives (missed distress)
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = y_test[fn_mask].index[:3]
    
    # 4. High confidence correct
    correct_mask = (y_test == y_pred)
    high_conf_correct = y_test[correct_mask & (y_proba > 0.8)].index[:3]
    
    examples = []
    
    # Analyze each category
    categories = [
        ("True Positive (Correctly Predicted Distress)", tp_indices),
        ("False Positive (False Alarm)", fp_indices),
        ("False Negative (Missed Distress)", fn_indices),
        ("High Confidence Correct", high_conf_correct)
    ]
    
    for category, indices in categories:
        if len(indices) > 0:
            print(f"{category}:")
            print("-" * 80)
            
            for idx in indices[:2]:  # Show 2 examples per category
                if idx in test_df.index:
                    row = test_df.loc[idx]
                    
                    # Get top features for this prediction
                    x_values = X_test.loc[idx]
                    top_features_idx = np.argsort(np.abs(x_values))[-5:][::-1]
                    
                    example = {
                        'category': category,
                        'company': row.get('company_name', 'Unknown'),
                        'date': row.get('date', 'Unknown'),
                        'true_label': int(y_test.loc[idx]),
                        'predicted_prob': float(y_proba[y_test.index.get_loc(idx)]),
                        'predicted_label': int(y_pred[y_test.index.get_loc(idx)])
                    }
                    
                    print(f"\n  Company: {example['company']}")
                    print(f"  Date: {example['date']}")
                    print(f"  True Label: {example['true_label']} | Predicted: {example['predicted_label']} (prob: {example['predicted_prob']:.3f})")
                    print(f"  Top 5 Feature Values:")
                    
                    for i, feat_idx in enumerate(top_features_idx, 1):
                        feat_name = features[feat_idx]
                        feat_value = x_values.iloc[feat_idx]
                        print(f"    {i}. {feat_name:25s}: {feat_value:8.3f}")
                    
                    examples.append(example)
            
            print()
    
    # Save examples
    if examples:
        examples_df = pd.DataFrame(examples)
        output_file = OUTPUT_DIR / 'example_predictions.csv'
        examples_df.to_csv(output_file, index=False)
        print(f"✓ Saved examples: {output_file}")


def create_explainability_summary(shap_values, features, model_name='Model'):
    """
    Create summary of explainability analysis for a specific model.
    """
    if shap_values is None:
        print(f"  ⚠️  SHAP values not available for {model_name}")
        return
    
    # Calculate feature importance from SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Save
    model_suffix = model_name.lower().replace(' ', '_')
    output_file = OUTPUT_DIR / f'shap_values_{model_suffix}.csv'
    summary.to_csv(output_file, index=False)
    print(f"\n{model_name} - Top 10 Features by SHAP Importance:")
    print()
    print(f"{'Rank':<6} {'Feature':<30} {'Mean |SHAP|':>12} {'Mean SHAP':>12}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(summary.head(10).iterrows(), 1):
        print(f"{i:<6} {row['feature']:<30} {row['mean_abs_shap']:>12.4f} {row['mean_shap']:>12.4f}")
    
    print()
    print(f"✓ Saved SHAP summary: {output_file}")
    
    return summary


def main():
    """
    Main execution: Model explainability analysis for both optimized models.
    """
    print("\n" + "="*80)
    print("STEP 15: MODEL EXPLAINABILITY (XGBOOST & LIGHTGBM)".center(80))
    print("="*80)
    
    # Load models and data
    models, X_test, y_test, test_df, features = load_models_and_data()
    
    if not models:
        print("\n⚠️  No optimized models found. Please run step 12 first.")
        return
    
    # Analyze each model
    print_section("15.1: SHAP ANALYSIS FOR BOTH MODELS")
    
    all_summaries = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {model_name}".center(80))
        print(f"{'='*80}")
        
        # Compute SHAP values
        shap_values, explainer, X_sample = compute_shap_values(model, X_test, features, model_name)
        
        # SHAP summary plots
        if shap_values is not None:
            plot_shap_summary(shap_values, X_sample, features, model_name)
            
            # Explainability summary
            summary = create_explainability_summary(shap_values, features, model_name)
            all_summaries[model_name] = summary
    
    # Compare feature importance between models
    if len(all_summaries) == 2:
        print_section("15.2: FEATURE IMPORTANCE COMPARISON")
        
        xgb_summary = all_summaries.get('XGBoost')
        lgb_summary = all_summaries.get('LightGBM')
        
        if xgb_summary is not None and lgb_summary is not None:
            print("\nTop 10 Features Comparison:")
            print()
            print(f"{'Rank':<6} {'XGBoost Feature':<30} {'LightGBM Feature':<30}")
            print("-" * 70)
            
            for i in range(min(10, len(xgb_summary), len(lgb_summary))):
                xgb_feat = xgb_summary.iloc[i]['feature']
                lgb_feat = lgb_summary.iloc[i]['feature']
                match = "✓" if xgb_feat == lgb_feat else " "
                print(f"{i+1:<6} {xgb_feat:<30} {lgb_feat:<30} {match}")
            
            # Calculate agreement
            xgb_top10 = set(xgb_summary.head(10)['feature'])
            lgb_top10 = set(lgb_summary.head(10)['feature'])
            agreement = len(xgb_top10 & lgb_top10)
            
            print()
            print(f"Feature Agreement: {agreement}/10 features in common")
    
    # Example predictions (using first available model)
    print_section("15.3: EXAMPLE PREDICTIONS ANALYSIS")
    first_model = list(models.values())[0]
    analyze_example_predictions(first_model, X_test, y_test, test_df, features)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ STEP 15 COMPLETE".center(80))
    print("="*80)
    
    if all_summaries:
        print(f"\n✓ SHAP analysis completed for {len(models)} models")
        print("✓ Feature importance explained for each model")
        print("✓ Model comparison generated")
    else:
        print("\n⚠️  SHAP analysis skipped (install shap package)")
    
    print("✓ Example predictions analyzed")
    print("✓ Explainability reports generated")
    print(f"\n✓ All figures saved to: {FIGURES_DIR}")
    print("\n🎉 All pipeline steps complete!\n")


if __name__ == "__main__":
    main()
