"""
STEP 13: Model Evaluation

Comprehensive evaluation of both optimized models (XGBoost & LightGBM):
    13.1 ROC and Precision-Recall curves (both models)
    13.2 Feature importance analysis (both models)
    13.3 Performance by year (both models)
    13.4 Model comparison

Outputs:
    - Figures: report/figures/step13_*.png
    - Report: output/evaluation_report.csv
    - Console: Detailed evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


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
    
    # Load XGBoost model
    xgb_model = None
    xgb_threshold = 0.5
    try:
        with open(MODELS_DIR / 'xgboost_optimized.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            xgb_model = model_dict['model']
            xgb_threshold = model_dict['threshold']
        print(f"  ✓ XGBoost model loaded (threshold: {xgb_threshold:.2f})")
    except FileNotFoundError:
        print("  ⚠️  XGBoost model not found, skipping...")
    
    # Load LightGBM model
    lgb_model = None
    lgb_threshold = 0.5
    try:
        with open(MODELS_DIR / 'lightgbm_optimized.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            lgb_model = model_dict['model']
            lgb_threshold = model_dict['threshold']
        print(f"  ✓ LightGBM model loaded (threshold: {lgb_threshold:.2f})")
    except FileNotFoundError:
        print("  ⚠️  LightGBM model not found, skipping...")
    
    # Prepare test data
    X_test = test_df[features].copy()
    y_test = test_df['distress_flag'].copy()
    
    X_test = pd.DataFrame(
        scaler.transform(imputer.transform(X_test)),
        columns=features,
        index=X_test.index
    )
    
    print(f"  ✓ Test data: {X_test.shape}\n")
    
    return xgb_model, xgb_threshold, lgb_model, lgb_threshold, X_test, y_test, test_df, features


def plot_roc_pr_curves_comparison(xgb_model, lgb_model, X_test, y_test):
    """
    Plot ROC and Precision-Recall curves for both models.
    """
    print_section("13.1: ROC AND PRECISION-RECALL CURVES (BOTH MODELS)")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    results = {}
    
    # XGBoost
    if xgb_model is not None:
        y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        # ROC curve
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
        
        # Precision-Recall curve
        precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
        avg_precision_xgb = average_precision_score(y_test, y_proba_xgb)
        
        # Plot ROC
        ax1.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2.5, 
                label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
        
        # Plot PR
        ax2.plot(recall_xgb, precision_xgb, color='darkorange', lw=2.5, 
                label=f'XGBoost (AP = {avg_precision_xgb:.3f})')
        
        results['xgb'] = {'roc_auc': roc_auc_xgb, 'avg_precision': avg_precision_xgb}
        print(f"XGBoost - ROC AUC: {roc_auc_xgb:.4f}, Average Precision: {avg_precision_xgb:.4f}")
    
    # LightGBM
    if lgb_model is not None:
        y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
        
        # ROC curve
        fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb)
        roc_auc_lgb = auc(fpr_lgb, tpr_lgb)
        
        # Precision-Recall curve
        precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_proba_lgb)
        avg_precision_lgb = average_precision_score(y_test, y_proba_lgb)
        
        # Plot ROC
        ax1.plot(fpr_lgb, tpr_lgb, color='darkgreen', lw=2.5, 
                label=f'LightGBM (AUC = {roc_auc_lgb:.3f})')
        
        # Plot PR
        ax2.plot(recall_lgb, precision_lgb, color='darkgreen', lw=2.5, 
                label=f'LightGBM (AP = {avg_precision_lgb:.3f})')
        
        results['lgb'] = {'roc_auc': roc_auc_lgb, 'avg_precision': avg_precision_lgb}
        print(f"LightGBM - ROC AUC: {roc_auc_lgb:.4f}, Average Precision: {avg_precision_lgb:.4f}")
    
    # ROC Curve formatting
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random', alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve formatting
    baseline = y_test.mean()
    ax2.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--', 
             label=f'Baseline ({baseline:.3f})', alpha=0.5)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / 'step13_roc_pr_curves_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    plt.close()
    
    return results


def plot_feature_importance_comparison(xgb_model, lgb_model, features):
    """
    Plot feature importance for both models.
    """
    print_section("13.2: FEATURE IMPORTANCE ANALYSIS (BOTH MODELS)")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    feature_importances = {}
    
    # XGBoost
    if xgb_model is not None:
        importances_xgb = xgb_model.feature_importances_
        fi_xgb = pd.DataFrame({
            'feature': features,
            'importance': importances_xgb
        }).sort_values('importance', ascending=False)
        
        feature_importances['xgb'] = fi_xgb
        
        # Save
        fi_xgb.to_csv(OUTPUT_DIR / 'feature_importance_xgboost.csv', index=False)
        print(f"✓ Saved XGBoost feature importance: {OUTPUT_DIR / 'feature_importance_xgboost.csv'}")
        
        # Print top 10
        print("\nXGBoost - Top 10 Most Important Features:")
        for i, row in fi_xgb.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:8.0f}")
        
        # Plot top 20
        top_20_xgb = fi_xgb.head(20)
        ax1.barh(range(len(top_20_xgb)), top_20_xgb['importance'], color='darkorange', alpha=0.8)
        ax1.set_yticks(range(len(top_20_xgb)))
        ax1.set_yticklabels(top_20_xgb['feature'], fontsize=9)
        ax1.set_xlabel('Importance', fontsize=11)
        ax1.set_title('XGBoost - Top 20 Features', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
    
    # LightGBM
    if lgb_model is not None:
        importances_lgb = lgb_model.feature_importances_
        fi_lgb = pd.DataFrame({
            'feature': features,
            'importance': importances_lgb
        }).sort_values('importance', ascending=False)
        
        feature_importances['lgb'] = fi_lgb
        
        # Save
        fi_lgb.to_csv(OUTPUT_DIR / 'feature_importance_lightgbm.csv', index=False)
        print(f"✓ Saved LightGBM feature importance: {OUTPUT_DIR / 'feature_importance_lightgbm.csv'}")
        
        # Print top 10
        print("\nLightGBM - Top 10 Most Important Features:")
        for i, row in fi_lgb.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:8.0f}")
        
        # Plot top 20
        top_20_lgb = fi_lgb.head(20)
        ax2.barh(range(len(top_20_lgb)), top_20_lgb['importance'], color='darkgreen', alpha=0.8)
        ax2.set_yticks(range(len(top_20_lgb)))
        ax2.set_yticklabels(top_20_lgb['feature'], fontsize=9)
        ax2.set_xlabel('Importance', fontsize=11)
        ax2.set_title('LightGBM - Top 20 Features', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / 'step13_feature_importance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    plt.close()
    
    return feature_importances


def analyze_performance_by_year(model, threshold, X_test, y_test, test_df, model_name='Model'):
    """
    Analyze performance by year for a specific model.
    """
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Add to test_df
    test_df_eval = test_df.copy()
    test_df_eval['y_true'] = y_test.values
    test_df_eval['y_pred'] = y_pred
    test_df_eval['y_proba'] = y_proba
    test_df_eval['date'] = pd.to_datetime(test_df_eval['date'])
    test_df_eval['year'] = test_df_eval['date'].dt.year
    
    # Calculate metrics by year
    yearly_metrics = []
    
    for year in sorted(test_df_eval['year'].unique()):
        year_data = test_df_eval[test_df_eval['year'] == year]
        
        if len(year_data) > 0 and year_data['y_true'].sum() > 0:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'year': year,
                'n_obs': len(year_data),
                'n_distress': year_data['y_true'].sum(),
                'distress_rate': year_data['y_true'].mean(),
                'accuracy': accuracy_score(year_data['y_true'], year_data['y_pred']),
                'precision': precision_score(year_data['y_true'], year_data['y_pred'], zero_division=0),
                'recall': recall_score(year_data['y_true'], year_data['y_pred'], zero_division=0),
                'f1': f1_score(year_data['y_true'], year_data['y_pred'], zero_division=0),
                'auc': roc_auc_score(year_data['y_true'], year_data['y_proba'])
            }
            yearly_metrics.append(metrics)
    
    yearly_df = pd.DataFrame(yearly_metrics)
    
    # Print results
    print(f"\n{model_name} Performance by Year:")
    print()
    print(f"{'Year':>6} {'N':>6} {'Dist%':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 70)
    
    for _, row in yearly_df.iterrows():
        print(f"{row['year']:>6.0f} {row['n_obs']:>6.0f} {row['distress_rate']*100:>6.1f}% "
              f"{row['accuracy']:>7.3f} {row['precision']:>7.3f} {row['recall']:>7.3f} "
              f"{row['f1']:>7.3f} {row['auc']:>7.3f}")
    
    # Plot performance over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Set color based on model
    color = 'darkorange' if 'XGBoost' in model_name else 'darkgreen'
    
    # AUC over time
    axes[0, 0].plot(yearly_df['year'], yearly_df['auc'], marker='o', linewidth=2, markersize=8, color=color)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    axes[0, 0].set_xlabel('Year', fontsize=11)
    axes[0, 0].set_ylabel('AUC', fontsize=11)
    axes[0, 0].set_title(f'{model_name} - AUC Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # F1 Score over time
    axes[0, 1].plot(yearly_df['year'], yearly_df['f1'], marker='o', linewidth=2, markersize=8, color=color)
    axes[0, 1].set_xlabel('Year', fontsize=11)
    axes[0, 1].set_ylabel('F1 Score', fontsize=11)
    axes[0, 1].set_title(f'{model_name} - F1 Score Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision vs Recall
    axes[1, 0].plot(yearly_df['year'], yearly_df['precision'], marker='o', linewidth=2, markersize=8, 
                    color='orange', label='Precision')
    axes[1, 0].plot(yearly_df['year'], yearly_df['recall'], marker='s', linewidth=2, markersize=8, 
                    color='purple', label='Recall')
    axes[1, 0].set_xlabel('Year', fontsize=11)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title(f'{model_name} - Precision vs Recall Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Distress Rate
    axes[1, 1].bar(yearly_df['year'], yearly_df['distress_rate']*100, color='coral', alpha=0.7)
    axes[1, 1].set_xlabel('Year', fontsize=11)
    axes[1, 1].set_ylabel('Distress Rate (%)', fontsize=11)
    axes[1, 1].set_title('Actual Distress Rate by Year', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save with model-specific filename
    model_suffix = model_name.lower().replace(' ', '_')
    output_file = FIGURES_DIR / f'step13_performance_over_time_{model_suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    
    return yearly_df


def create_confusion_matrix_plot(model, threshold, X_test, y_test, model_name='Model'):
    """
    Create detailed confusion matrix visualization.
    """
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                square=True, linewidths=1, linecolor='black',
                xticklabels=['No Distress', 'Distress'],
                yticklabels=['No Distress', 'Distress'],
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Confusion Matrix (Threshold = {threshold:.2f})', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    # Save with model-specific filename
    model_suffix = model_name.lower().replace(' ', '_')
    output_file = FIGURES_DIR / f'step13_confusion_matrix_{model_suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    
    # Print detailed metrics
    print(f"\n{model_name} Confusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:5d} ({cm[0,0]/total*100:5.1f}%)")
    print(f"  False Positives: {cm[0,1]:5d} ({cm[0,1]/total*100:5.1f}%)")
    print(f"  False Negatives: {cm[1,0]:5d} ({cm[1,0]/total*100:5.1f}%)")
    print(f"  True Positives:  {cm[1,1]:5d} ({cm[1,1]/total*100:5.1f}%)")
    
    # Calculate rates
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
    
    print()
    print(f"Specificity (True Negative Rate): {specificity:.3f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.3f}")


def generate_evaluation_report(roc_auc, avg_precision, feature_importance, yearly_df):
    """
    Generate comprehensive evaluation report.
    """
    print_section("GENERATING EVALUATION REPORT")
    
    report = {
        'metric': [],
        'value': [],
        'description': []
    }
    
    # Overall metrics
    report['metric'].extend(['ROC AUC', 'Average Precision', 'Top Feature', 'Top Feature Importance'])
    report['value'].extend([
        f"{roc_auc:.4f}",
        f"{avg_precision:.4f}",
        feature_importance.iloc[0]['feature'],
        f"{feature_importance.iloc[0]['importance']:.0f}"
    ])
    report['description'].extend([
        'Area under ROC curve',
        'Area under PR curve',
        'Most important feature',
        'Importance score'
    ])
    
    # Temporal metrics
    report['metric'].extend(['Best Year (AUC)', 'Worst Year (AUC)', 'AUC Std Dev'])
    report['value'].extend([
        f"{yearly_df.loc[yearly_df['auc'].idxmax(), 'year']:.0f} ({yearly_df['auc'].max():.3f})",
        f"{yearly_df.loc[yearly_df['auc'].idxmin(), 'year']:.0f} ({yearly_df['auc'].min():.3f})",
        f"{yearly_df['auc'].std():.4f}"
    ])
    report['description'].extend([
        'Year with best AUC',
        'Year with worst AUC',
        'AUC stability across years'
    ])
    
    report_df = pd.DataFrame(report)
    
    # Save
    report_file = OUTPUT_DIR / 'evaluation_report.csv'
    report_df.to_csv(report_file, index=False)
    print(f"✓ Saved: {report_file}")
    
    # Print summary
    print("\nEvaluation Summary:")
    for _, row in report_df.iterrows():
        print(f"  {row['metric']:25s}: {row['value']:30s} ({row['description']})")


def main():
    """
    Main execution: Comprehensive model evaluation for both models.
    """
    print("\n" + "="*80)
    print("STEP 13: MODEL EVALUATION (XGBOOST & LIGHTGBM)".center(80))
    print("="*80)
    
    # Load models and data
    xgb_model, xgb_threshold, lgb_model, lgb_threshold, X_test, y_test, test_df, features = load_models_and_data()
    
    # ROC and PR curves comparison
    curve_results = plot_roc_pr_curves_comparison(xgb_model, lgb_model, X_test, y_test)
    
    # Feature importance comparison
    feature_importances = plot_feature_importance_comparison(xgb_model, lgb_model, features)
    
    # Performance by year for both models
    print_section("13.3: PERFORMANCE BY YEAR (BOTH MODELS)")
    
    yearly_results = {}
    if xgb_model is not None:
        yearly_xgb = analyze_performance_by_year(xgb_model, xgb_threshold, X_test, y_test, test_df, model_name='XGBoost')
        yearly_xgb.to_csv(OUTPUT_DIR / 'performance_by_year_xgboost.csv', index=False)
        yearly_results['xgb'] = yearly_xgb
    
    if lgb_model is not None:
        yearly_lgb = analyze_performance_by_year(lgb_model, lgb_threshold, X_test, y_test, test_df, model_name='LightGBM')
        yearly_lgb.to_csv(OUTPUT_DIR / 'performance_by_year_lightgbm.csv', index=False)
        yearly_results['lgb'] = yearly_lgb
    
    # Confusion matrices for both models
    print_section("13.4: CONFUSION MATRICES")
    
    if xgb_model is not None:
        create_confusion_matrix_plot(xgb_model, xgb_threshold, X_test, y_test, model_name='XGBoost')
    
    if lgb_model is not None:
        create_confusion_matrix_plot(lgb_model, lgb_threshold, X_test, y_test, model_name='LightGBM')
    
    # Model comparison summary
    print_section("13.5: MODEL COMPARISON SUMMARY")
    
    comparison_data = []
    if 'xgb' in curve_results:
        comparison_data.append({
            'model': 'XGBoost',
            'roc_auc': curve_results['xgb']['roc_auc'],
            'avg_precision': curve_results['xgb']['avg_precision'],
            'threshold': xgb_threshold,
            'top_feature': feature_importances['xgb'].iloc[0]['feature'] if 'xgb' in feature_importances else 'N/A'
        })
    
    if 'lgb' in curve_results:
        comparison_data.append({
            'model': 'LightGBM',
            'roc_auc': curve_results['lgb']['roc_auc'],
            'avg_precision': curve_results['lgb']['avg_precision'],
            'threshold': lgb_threshold,
            'top_feature': feature_importances['lgb'].iloc[0]['feature'] if 'lgb' in feature_importances else 'N/A'
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(OUTPUT_DIR / 'model_comparison_summary.csv', index=False)
        
        print("Model Comparison:")
        print()
        print(f"{'Model':<15} {'ROC AUC':>10} {'Avg Prec':>10} {'Threshold':>10} {'Top Feature':<25}")
        print("-" * 80)
        for _, row in comparison_df.iterrows():
            print(f"{row['model']:<15} {row['roc_auc']:>10.4f} {row['avg_precision']:>10.4f} "
                  f"{row['threshold']:>10.2f} {row['top_feature']:<25}")
        
        print(f"\n✓ Saved comparison: {OUTPUT_DIR / 'model_comparison_summary.csv'}")
    
    print("\n" + "="*80)
    print("✅ STEP 13 COMPLETE".center(80))
    print("="*80)
    print("\n✓ ROC and PR curves generated for both models")
    print("✓ Feature importance analyzed for both models")
    print("✓ Performance by year evaluated for both models")
    print("✓ Confusion matrices visualized for both models")
    print("✓ Model comparison summary saved")
    print(f"\n✓ All figures saved to: {FIGURES_DIR}")
    print("\nNext: Run step14_benchmark_comparison.py\n")


if __name__ == "__main__":
    main()
