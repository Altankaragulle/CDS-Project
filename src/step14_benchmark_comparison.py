"""
STEP 14: Benchmark Comparison

Compare our optimized models (XGBoost & LightGBM) against baseline approaches:
    14.1 Naive baselines (random, majority class, CDS lag only)
    14.2 Compare all models and visualize

Outputs:
    - Figure: report/figures/step14_model_comparison.png
    - Report: output/benchmark_comparison.csv
    - Console: Comparison results
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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.dummy import DummyClassifier

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


def load_data_and_models():
    """Load test data and all models."""
    print("Loading test data and models...")
    
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
    
    # Prepare test data
    X_test = test_df[features].copy()
    y_test = test_df['distress_flag'].copy()
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(imputer.transform(X_test)),
        columns=features,
        index=X_test.index
    )
    
    # Load trained models
    models = {}
    
    # Logistic Regression
    try:
        with open(MODELS_DIR / 'logistic_regression.pkl', 'rb') as f:
            models['Logistic Regression'] = pickle.load(f)
    except:
        pass
    
    # Random Forest
    try:
        with open(MODELS_DIR / 'random_forest.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
    except:
        pass
    
    # XGBoost
    try:
        with open(MODELS_DIR / 'xgboost.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
    except:
        pass
    
    # LightGBM
    try:
        with open(MODELS_DIR / 'lightgbm.pkl', 'rb') as f:
            models['LightGBM'] = pickle.load(f)
    except:
        pass
    
    # Optimized XGBoost
    try:
        with open(MODELS_DIR / 'xgboost_optimized.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            models['XGBoost Optimized'] = model_dict['model']
            print(f"  ✓ Loaded XGBoost Optimized (threshold: {model_dict['threshold']:.2f})")
    except:
        pass
    
    # Optimized LightGBM
    try:
        with open(MODELS_DIR / 'lightgbm_optimized.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            models['LightGBM Optimized'] = model_dict['model']
            print(f"  ✓ Loaded LightGBM Optimized (threshold: {model_dict['threshold']:.2f})")
    except:
        pass
    
    print(f"  ✓ Loaded {len(models)} trained models total")
    print(f"  ✓ Test data: {X_test_scaled.shape}\n")
    
    return X_test_scaled, y_test, models, test_df


def create_naive_baselines(X_test, y_test, test_df):
    """
    Create naive baseline models for comparison.
    
    Returns:
        Dictionary of baseline models and their predictions
    """
    print_section("14.1: CREATING NAIVE BASELINES")
    
    baselines = {}
    
    # 1. Random Classifier
    print("1. Random Classifier (coin flip)...")
    random_clf = DummyClassifier(strategy='uniform', random_state=42)
    random_clf.fit(X_test, y_test)
    baselines['Random'] = random_clf
    
    # 2. Majority Class Classifier
    print("2. Majority Class (always predict non-distress)...")
    majority_clf = DummyClassifier(strategy='most_frequent')
    majority_clf.fit(X_test, y_test)
    baselines['Majority Class'] = majority_clf
    
    # 3. Stratified (proportional to class distribution)
    print("3. Stratified (predict based on class distribution)...")
    stratified_clf = DummyClassifier(strategy='stratified', random_state=42)
    stratified_clf.fit(X_test, y_test)
    baselines['Stratified'] = stratified_clf
    
    # 4. CDS-Only Baseline (TA's suggested baseline!)
    print("4. CDS-Only (high CDS spread → predict distress)...")
    baselines['CDS-Only'] = create_cds_only_baseline(test_df, y_test)
    
    print(f"\n✓ Created {len(baselines)} baseline models")
    
    return baselines


def create_cds_only_baseline(test_df, y_test):
    """
    CDS-only baseline: predict distress based on current CDS spread.
    
    This is the key baseline suggested by the TA:
    "A baseline naïve model would be to just look at the CDS spread, 
    with high spread → high probability of distress"
    
    Strategy: Use current CDS spread directly as probability score
    (higher CDS → higher probability of distress)
    
    Returns:
        A simple classifier that uses CDS spread as predictor
    """
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    class CDSOnlyClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.test_df_ = None
            self.median_cds_ = None
            self.std_cds_ = None
        
        def fit(self, X, y):
            return self
        
        def set_data(self, test_df, y):
            """Set the test dataframe and compute CDS statistics."""
            self.test_df_ = test_df
            
            # Get current CDS spread
            current_cds = test_df['cds_spread'].values
            valid_cds = current_cds[~np.isnan(current_cds)]
            
            if len(valid_cds) > 0:
                self.median_cds_ = np.median(valid_cds)
                self.std_cds_ = np.std(valid_cds)
            else:
                self.median_cds_ = 0
                self.std_cds_ = 1
            
            return self
        
        def predict(self, X):
            """Predict based on CDS spread (above median → distress)."""
            if self.test_df_ is None:
                return np.zeros(len(X))
            
            current_cds = self.test_df_['cds_spread'].values[:len(X)]
            
            # Simple rule: CDS above median → predict distress
            predictions = (current_cds > self.median_cds_).astype(int)
            
            # Handle NaN - predict 0 (no distress)
            predictions[np.isnan(current_cds)] = 0
            
            return predictions
        
        def predict_proba(self, X):
            """
            Return probabilities using CDS spread directly.
            
            Strategy: Convert CDS to probability using sigmoid-like transformation
            - Higher CDS → higher probability of distress
            - Use median as inflection point
            """
            if self.test_df_ is None:
                return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
            
            current_cds = self.test_df_['cds_spread'].values[:len(X)]
            
            # Standardize CDS around median
            # (CDS - median) / std gives z-score
            z_score = (current_cds - self.median_cds_) / (self.std_cds_ + 1e-6)
            
            # Convert to probability using sigmoid
            # sigmoid(z) = 1 / (1 + exp(-z))
            # This gives smooth probability: 
            # - CDS at median → prob = 0.5
            # - CDS above median → prob > 0.5
            # - CDS below median → prob < 0.5
            proba_distress = 1 / (1 + np.exp(-z_score))
            
            # Handle NaN
            proba_distress[np.isnan(current_cds)] = 0.5  # Neutral for missing
            
            proba_no_distress = 1 - proba_distress
            
            return np.column_stack([proba_no_distress, proba_distress])
    
    # Create and fit the classifier
    clf = CDSOnlyClassifier()
    clf.set_data(test_df, y_test)
    
    print(f"   → Using CDS spread directly as probability")
    print(f"   → Median CDS: {clf.median_cds_:.2f} bps (decision boundary)")
    
    return clf


def evaluate_all_models(models, baselines, X_test, y_test):
    """
    Evaluate all models (trained + baselines).
    
    Returns:
        DataFrame with all results
    """
    print_section("14.2: EVALUATING ALL MODELS")
    
    results = []
    
    # Evaluate baselines
    print("Evaluating baseline models...")
    for name, model in baselines.items():
        y_pred = model.predict(X_test)
        
        # For baselines, use predict_proba if available, else use predictions
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_proba = y_pred.astype(float)
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5  # Random performance
        
        results.append({
            'Model': name,
            'Type': 'Baseline',
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUC': auc
        })
        
        print(f"  ✓ {name:20s}: AUC = {auc:.4f}, F1 = {results[-1]['F1']:.4f}")
    
    # Evaluate trained models
    print("\nEvaluating trained models...")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results.append({
            'Model': name,
            'Type': 'ML Model',
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, y_proba)
        })
        
        print(f"  ✓ {name:20s}: AUC = {results[-1]['AUC']:.4f}, F1 = {results[-1]['F1']:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AUC', ascending=False)
    
    print()
    print("✓ All models evaluated")
    
    return results_df


def visualize_comparison(results_df):
    """
    Create visualization comparing all models.
    """
    print_section("VISUALIZATION")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color palette
    colors = ['#d62728' if t == 'Baseline' else '#2ca02c' for t in results_df['Type']]
    
    # 1. AUC Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.barh(range(len(results_df)), results_df['AUC'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['Model'])
    ax1.set_xlabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax1.set_title('Model Comparison: AUC-ROC', fontsize=12, fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, results_df['AUC'])):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=9)
    
    # 2. F1 Score Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.barh(range(len(results_df)), results_df['F1'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(results_df)))
    ax2.set_yticklabels(results_df['Model'])
    ax2.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
    ax2.set_title('Model Comparison: F1 Score', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, results_df['F1'])):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=9)
    
    # 3. Precision vs Recall
    ax3 = axes[1, 0]
    for model_type in ['Baseline', 'ML Model']:
        mask = results_df['Type'] == model_type
        ax3.scatter(results_df[mask]['Recall'], results_df[mask]['Precision'], 
                   s=150, alpha=0.7, label=model_type)
    
    # Add labels for each point
    for _, row in results_df.iterrows():
        ax3.annotate(row['Model'], (row['Recall'], row['Precision']), 
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax3.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # 4. Overall Performance Radar (top 5 models)
    ax4 = axes[1, 1]
    
    # Select top 5 models by AUC
    top_5 = results_df.head(5)
    
    metrics = ['AUC', 'F1', 'Precision', 'Recall', 'Accuracy']
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, (_, row) in enumerate(top_5.iterrows()):
        values = [row[m] for m in metrics]
        ax4.bar(x + i*width, values, width, label=row['Model'], alpha=0.7)
    
    ax4.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax4.set_title('Top 5 Models: Multi-Metric Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width * 2)
    ax4.set_xticklabels(metrics)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / 'step14_model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def print_comparison_table(results_df):
    """
    Print detailed comparison table.
    """
    print_section("DETAILED COMPARISON")
    
    print("Model Performance Comparison:")
    print()
    print(f"{'Rank':<6} {'Model':<25} {'Type':<12} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    print("-" * 95)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i:<6} {row['Model']:<25} {row['Type']:<12} "
              f"{row['AUC']:>8.4f} {row['F1']:>8.4f} {row['Precision']:>8.4f} "
              f"{row['Recall']:>8.4f} {row['Accuracy']:>8.4f}")
    
    # Calculate improvements
    print()
    print("="*95)
    print("IMPROVEMENT OVER BASELINES:")
    print("="*95)
    
    best_model = results_df.iloc[0]
    best_baseline = results_df[results_df['Type'] == 'Baseline'].iloc[0]
    
    print(f"\nBest Model: {best_model['Model']}")
    print(f"Best Baseline: {best_baseline['Model']}")
    print()
    
    for metric in ['AUC', 'F1', 'Precision', 'Recall', 'Accuracy']:
        improvement = best_model[metric] - best_baseline[metric]
        pct_improvement = (improvement / best_baseline[metric] * 100) if best_baseline[metric] > 0 else 0
        print(f"  {metric:12s}: {best_model[metric]:.4f} vs {best_baseline[metric]:.4f} "
              f"(+{improvement:.4f}, +{pct_improvement:.1f}%)")


def save_comparison_results(results_df):
    """
    Save comparison results.
    """
    print_section("SAVING RESULTS")
    
    # Save full results
    output_file = OUTPUT_DIR / 'benchmark_comparison.csv'
    results_df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    
    # Create summary
    summary = {
        'best_model': results_df.iloc[0]['Model'],
        'best_auc': results_df.iloc[0]['AUC'],
        'best_f1': results_df.iloc[0]['F1'],
        'best_baseline': results_df[results_df['Type'] == 'Baseline'].iloc[0]['Model'],
        'baseline_auc': results_df[results_df['Type'] == 'Baseline'].iloc[0]['AUC'],
        'improvement_auc': results_df.iloc[0]['AUC'] - results_df[results_df['Type'] == 'Baseline'].iloc[0]['AUC']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = OUTPUT_DIR / 'benchmark_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved: {summary_file}")


def main():
    """
    Main execution: Benchmark comparison.
    """
    print("\n" + "="*80)
    print("STEP 14: BENCHMARK COMPARISON".center(80))
    print("="*80)
    
    # Load data and models
    X_test, y_test, models, test_df = load_data_and_models()
    
    # Create baselines (including CDS-only baseline)
    baselines = create_naive_baselines(X_test, y_test, test_df)
    
    # Evaluate all models
    results_df = evaluate_all_models(models, baselines, X_test, y_test)
    
    # Visualize
    visualize_comparison(results_df)
    
    # Print comparison
    print_comparison_table(results_df)
    
    # Save results
    save_comparison_results(results_df)
    
    print("\n" + "="*80)
    print(" STEP 14 COMPLETE".center(80))
    print("="*80)
    print("\n✓ Baseline models created and evaluated")
    print("✓ All models compared")
    print("✓ Comparison visualization generated")
    print("✓ Results saved")
    print("\nNext: Run step15_explainability.py\n")
    
    return results_df


if __name__ == "__main__":
    results_df = main()
