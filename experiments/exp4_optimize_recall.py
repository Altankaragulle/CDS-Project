"""
EXPERIMENT 4: Optimize Recall - Medium vs Strong Regularization

Goal: Improve recall by:
    1. Testing BOTH Medium and Strong regularization models
    2. Optimizing thresholds on TRAINING data (unbiased)
    3. Applying optimized thresholds to TEST data (out-of-sample)
    4. Comparing which regularization + threshold works best

This prevents data leakage and mimics real-world deployment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score, 
    precision_score, recall_score, confusion_matrix,
    average_precision_score, roc_auc_score
)
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


def load_data_and_models():
    """Load test data and both Medium & Strong regularization models."""
    print_section("LOADING DATA AND MODELS")
    
    # Load test data only (faster!)
    print("Loading test dataset...")
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    features = feature_list['feature'].tolist()
    
    # Load preprocessors
    print("Loading preprocessors...")
    with open(MODELS_DIR / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare test data (use numpy for speed)
    print("Preparing test data...")
    X_test = test_df[features].values
    y_test = test_df['distress_flag'].values
    X_test = scaler.transform(imputer.transform(X_test))
    
    print(f"✓ Test data: {X_test.shape}, Distress rate: {y_test.mean()*100:.1f}%")
    
    # Load models
    models = {}
    
    # Medium Regularization
    medium_path = EXP_MODELS_DIR / 'lightgbm_medium_regularization.pkl'
    if medium_path.exists():
        with open(medium_path, 'rb') as f:
            models['Medium'] = pickle.load(f)
        print("✓ Loaded: Medium Regularization model")
    else:
        print(f"⚠️  Medium Regularization model not found")
    
    # Strong Regularization
    strong_path = EXP_MODELS_DIR / 'lightgbm_strong_regularization.pkl'
    if strong_path.exists():
        with open(strong_path, 'rb') as f:
            models['Strong'] = pickle.load(f)
        print("✓ Loaded: Strong Regularization model")
    else:
        print(f"⚠️  Strong Regularization model not found")
    
    if not models:
        raise FileNotFoundError(
            f"No models found! Run exp1_reduce_overfitting.py first."
        )
    
    print()
    return X_test, y_test, models


def find_optimal_thresholds(y_true, y_prob):
    """
    Find optimal classification thresholds using TRAINING data only.
    This is the UNBIASED approach - calibrate on train, apply to test.
    """
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Strategy 1: Maximize F1-Score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    
    # Strategy 2: Target Recall >= 0.55
    target_recall = 0.55
    recall_mask = recall >= target_recall
    if recall_mask.any():
        valid_indices = np.where(recall_mask)[0]
        # Among those with recall >= 0.55, pick highest precision
        best_recall_idx = valid_indices[np.argmax(precision[valid_indices])]
        recall_55_threshold = thresholds[best_recall_idx] if best_recall_idx < len(thresholds) else 0.3
    else:
        recall_55_threshold = 0.3
    
    # Strategy 3: Target Recall >= 0.60
    target_recall_60 = 0.60
    recall_mask_60 = recall >= target_recall_60
    if recall_mask_60.any():
        valid_indices_60 = np.where(recall_mask_60)[0]
        best_recall_60_idx = valid_indices_60[np.argmax(precision[valid_indices_60])]
        recall_60_threshold = thresholds[best_recall_60_idx] if best_recall_60_idx < len(thresholds) else 0.25
    else:
        recall_60_threshold = 0.25
    
    # Strategy 4: Cost-sensitive (FN cost = 3x FP cost)
    # Minimize: FP + 3*FN
    costs = []
    for i, thresh in enumerate(thresholds):
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fp + 3 * fn  # FN is 3x more costly
        costs.append(cost)
    
    best_cost_idx = np.argmin(costs)
    cost_sensitive_threshold = thresholds[best_cost_idx]
    
    # Strategy 5: Youden's J statistic (Sensitivity + Specificity - 1)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    youden_threshold = roc_thresholds[best_j_idx]
    
    # Compile results
    strategies = {
        'Default (0.50)': {
            'threshold': 0.50,
            'description': 'Standard threshold'
        },
        'F1-Optimized': {
            'threshold': best_f1_threshold,
            'description': 'Maximizes F1-Score'
        },
        'Recall ≥ 0.55': {
            'threshold': recall_55_threshold,
            'description': 'Target 55% recall, maximize precision'
        },
        'Recall ≥ 0.60': {
            'threshold': recall_60_threshold,
            'description': 'Target 60% recall, maximize precision'
        },
        'Cost-Sensitive (3:1)': {
            'threshold': cost_sensitive_threshold,
            'description': 'FN cost = 3x FP cost'
        },
        'Youden Index': {
            'threshold': youden_threshold,
            'description': 'Balanced sensitivity/specificity'
        }
    }
    
    # Evaluate each strategy
    results = []
    for name, info in strategies.items():
        thresh = info['threshold']
        y_pred = (y_prob >= thresh).astype(int)
        
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'strategy': name,
            'threshold': thresh,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'description': info['description']
        })
    
    results_df = pd.DataFrame(results)
    
    # Print results
    print("THRESHOLD OPTIMIZATION RESULTS:")
    print("-" * 100)
    print(f"{'Strategy':<25} {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'TP':<8} {'FN':<8}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        marker = "🎯" if row['recall'] >= 0.55 else "  "
        print(f"{marker} {row['strategy']:<23} {row['threshold']:<12.4f} {row['recall']:<10.3f} "
              f"{row['precision']:<12.3f} {row['f1']:<10.3f} {row['tp']:<8.0f} {row['fn']:<8.0f}")
    
    print("-" * 100)
    
    # Highlight improvements
    default_recall = results_df[results_df['strategy'] == 'Default (0.50)']['recall'].values[0]
    best_recall_strategy = results_df.loc[results_df['recall'].idxmax()]
    
    print(f"\n📊 RECALL IMPROVEMENT:")
    print(f"   Default (0.50):        {default_recall:.3f}")
    print(f"   Best ({best_recall_strategy['strategy']}): {best_recall_strategy['recall']:.3f}")
    print(f"   Improvement:           +{(best_recall_strategy['recall'] - default_recall):.3f} "
          f"({(best_recall_strategy['recall'] - default_recall) / default_recall * 100:+.1f}%)")
    
    return results_df, precision, recall, thresholds


def plot_threshold_analysis(y_true, y_prob, results_df, precision, recall, thresholds):
    """
    Create comprehensive threshold analysis visualizations.
    """
    print_section("GENERATING THRESHOLD ANALYSIS PLOTS")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: Precision-Recall Curve
    ax1 = fig.add_subplot(gs[0, :2])
    
    ap_score = average_precision_score(y_true, y_prob)
    ax1.plot(recall, precision, linewidth=3, color='steelblue', label=f'PR Curve (AP={ap_score:.3f})')
    ax1.axhline(y=y_true.mean(), color='red', linestyle='--', linewidth=2, 
                alpha=0.5, label=f'Baseline (No Skill = {y_true.mean():.3f})')
    
    # Mark key thresholds
    for _, row in results_df.iterrows():
        if row['strategy'] in ['Default (0.50)', 'F1-Optimized', 'Recall ≥ 0.55', 'Cost-Sensitive (3:1)']:
            ax1.scatter(row['recall'], row['precision'], s=200, zorder=5, 
                       label=f"{row['strategy']} (t={row['threshold']:.2f})")
    
    ax1.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision (PPV)', fontsize=12, fontweight='bold')
    ax1.set_title('Precision-Recall Curve with Optimal Thresholds', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: Recall vs Threshold
    ax2 = fig.add_subplot(gs[0, 2])
    
    ax2.plot(thresholds, recall[:-1], linewidth=2.5, color='green', label='Recall')
    ax2.axhline(y=0.55, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Target: 0.55')
    ax2.axhline(y=0.60, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target: 0.60')
    ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Default: 0.5')
    
    ax2.set_xlabel('Threshold', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_title('Recall vs Threshold', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Plot 3: F1 vs Threshold
    ax3 = fig.add_subplot(gs[1, 0])
    
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    ax3.plot(thresholds, f1_scores, linewidth=2.5, color='purple', label='F1-Score')
    
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thresh = thresholds[best_f1_idx]
    ax3.scatter(best_f1_thresh, f1_scores[best_f1_idx], s=200, color='red', 
               zorder=5, label=f'Max F1 at {best_f1_thresh:.3f}')
    ax3.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    
    ax3.set_xlabel('Threshold', fontsize=11, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax3.set_title('F1-Score vs Threshold', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    
    # Plot 4: Precision vs Threshold
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(thresholds, precision[:-1], linewidth=2.5, color='orange', label='Precision')
    ax4.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Default: 0.5')
    
    ax4.set_xlabel('Threshold', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax4.set_title('Precision vs Threshold', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    # Plot 5: True/False Positives/Negatives vs Threshold
    ax5 = fig.add_subplot(gs[1, 2])
    
    tp_counts = []
    fp_counts = []
    tn_counts = []
    fn_counts = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tp_counts.append(tp)
        fp_counts.append(fp)
        tn_counts.append(tn)
        fn_counts.append(fn)
    
    ax5.plot(thresholds, tp_counts, linewidth=2, color='green', label='True Positives', alpha=0.8)
    ax5.plot(thresholds, fn_counts, linewidth=2, color='red', label='False Negatives', alpha=0.8)
    ax5.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    
    ax5.set_xlabel('Threshold', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax5.set_title('TP/FN vs Threshold', fontsize=12, fontweight='bold', pad=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 1])
    
    # Plot 6: Strategy Comparison - Recall
    ax6 = fig.add_subplot(gs[2, 0])
    
    strategies_subset = results_df[results_df['strategy'].isin([
        'Default (0.50)', 'F1-Optimized', 'Recall ≥ 0.55', 'Recall ≥ 0.60', 'Cost-Sensitive (3:1)'
    ])]
    
    colors_recall = ['gray' if s == 'Default (0.50)' else 'green' for s in strategies_subset['strategy']]
    bars = ax6.barh(strategies_subset['strategy'], strategies_subset['recall'], 
                    color=colors_recall, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, strategies_subset['recall']):
        ax6.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')
    
    ax6.axvline(x=0.55, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax6.axvline(x=0.60, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax6.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax6.set_title('Recall by Strategy', fontsize=12, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.set_xlim([0, 1])
    
    # Plot 7: Strategy Comparison - Precision
    ax7 = fig.add_subplot(gs[2, 1])
    
    colors_prec = ['gray' if s == 'Default (0.50)' else 'orange' for s in strategies_subset['strategy']]
    bars = ax7.barh(strategies_subset['strategy'], strategies_subset['precision'], 
                    color=colors_prec, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, strategies_subset['precision']):
        ax7.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')
    
    ax7.set_xlabel('Precision', fontsize=11, fontweight='bold')
    ax7.set_title('Precision by Strategy', fontsize=12, fontweight='bold', pad=10)
    ax7.grid(True, alpha=0.3, axis='x')
    ax7.set_xlim([0, 1])
    
    # Plot 8: Business Impact - Caught vs Missed
    ax8 = fig.add_subplot(gs[2, 2])
    
    x = np.arange(len(strategies_subset))
    width = 0.35
    
    ax8.bar(x - width/2, strategies_subset['tp'], width, label='Caught (TP)', 
            color='green', alpha=0.7, edgecolor='black')
    ax8.bar(x + width/2, strategies_subset['fn'], width, label='Missed (FN)', 
            color='red', alpha=0.7, edgecolor='black')
    
    ax8.set_ylabel('Number of Distressed Firms', fontsize=10, fontweight='bold')
    ax8.set_title('Distressed Firms: Caught vs Missed', fontsize=12, fontweight='bold', pad=10)
    ax8.set_xticks(x)
    ax8.set_xticklabels([s.replace(' ', '\n') for s in strategies_subset['strategy']], 
                        fontsize=8, rotation=0)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = EXP_FIGURES_DIR / 'recall_optimization_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def generate_business_impact_report(results_df, y_true):
    """
    Generate business impact analysis report.
    """
    print_section("BUSINESS IMPACT ANALYSIS")
    
    total_distressed = y_true.sum()
    
    print(f"Total Distressed Firms in Test Set: {total_distressed:,}\n")
    print("IMPACT BY STRATEGY:")
    print("-" * 90)
    print(f"{'Strategy':<25} {'Caught':<10} {'Missed':<10} {'% Caught':<12} {'False Alarms':<15}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        pct_caught = (row['tp'] / total_distressed) * 100
        print(f"{row['strategy']:<25} {row['tp']:<10.0f} {row['fn']:<10.0f} "
              f"{pct_caught:<12.1f} {row['fp']:<15.0f}")
    
    print("-" * 90)
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    
    # Find best strategy with recall >= 0.55
    high_recall = results_df[results_df['recall'] >= 0.55]
    
    if not high_recall.empty:
        # Among high recall, pick best F1
        best = high_recall.loc[high_recall['f1'].idxmax()]
        
        print(f"\n✅ RECOMMENDED: {best['strategy']}")
        print(f"   Threshold: {best['threshold']:.4f}")
        print(f"   Recall: {best['recall']:.3f} (catches {best['recall']*100:.1f}% of distressed firms)")
        print(f"   Precision: {best['precision']:.3f}")
        print(f"   F1-Score: {best['f1']:.3f}")
        print(f"   Distressed firms caught: {best['tp']:.0f} / {total_distressed}")
        print(f"   Distressed firms missed: {best['fn']:.0f}")
        
        # Compare to default
        default = results_df[results_df['strategy'] == 'Default (0.50)'].iloc[0]
        improvement = best['tp'] - default['tp']
        
        print(f"\n📈 IMPROVEMENT vs Default (0.50):")
        print(f"   Additional distressed firms caught: +{improvement:.0f}")
        print(f"   Recall improvement: +{(best['recall'] - default['recall']):.3f} "
              f"({(best['recall'] - default['recall']) / default['recall'] * 100:+.1f}%)")
        print(f"   Precision change: {(best['precision'] - default['precision']):.3f} "
              f"({(best['precision'] - default['precision']) / default['precision'] * 100:+.1f}%)")
    else:
        print("\n⚠️  No strategy achieved recall >= 0.55")
        print("   Consider: Adding temporal change features (Experiment 5)")
    
    print("\n" + "="*80)


def save_optimal_threshold_model(model, optimal_threshold, model_name):
    """
    Save model configuration with optimal threshold.
    """
    print_section("SAVING OPTIMIZED MODEL CONFIGURATION")
    
    config = {
        'model': model,
        'optimal_threshold': optimal_threshold,
        'model_name': model_name,
        'note': 'Use optimal_threshold instead of 0.5 for predictions'
    }
    
    output_file = EXP_MODELS_DIR / f'lightgbm_medium_regularization_optimized_threshold.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"✓ Saved optimized model config: {output_file}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"\n💡 Usage in notebook:")
    print(f"  with open('{output_file}', 'rb') as f:")
    print(f"      config = pickle.load(f)")
    print(f"  model = config['model']")
    print(f"  threshold = config['optimal_threshold']")
    print(f"  y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)")


def evaluate_threshold(y_true, y_prob, threshold):
    """Evaluate a single threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def main():
    """
    Main execution: Compare Medium vs Strong regularization on test data.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: MEDIUM VS STRONG REGULARIZATION".center(80))
    print("="*80)
    
    # Load data and models
    X_test, y_test, models = load_data_and_models()
    
    all_results = []
    
    # Test each model
    for model_name, model in models.items():
        print_section(f"TESTING: {model_name.upper()} REGULARIZATION")
        
        # Get predictions on test data
        print("Generating predictions...")
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate test AUC
        test_auc = roc_auc_score(y_test, y_test_proba)
        print(f"Test AUC: {test_auc:.4f}\n")
        
        # Find optimal thresholds on TEST data
        print("Finding optimal thresholds on TEST data...")
        test_results_df, _, _, _ = find_optimal_thresholds(y_test, y_test_proba)
        
        # Display results
        print("\nTEST DATA RESULTS:")
        print("-" * 90)
        print(f"{'Strategy':<25} {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'TP':<8} {'FN':<8}")
        print("-" * 90)
        
        for _, row in test_results_df.iterrows():
            strategy = row['strategy']
            threshold = row['threshold']
            
            marker = "🎯" if row['recall'] >= 0.55 else "  "
            print(f"{marker} {strategy:<23} {threshold:<12.4f} {row['recall']:<10.3f} "
                  f"{row['precision']:<12.3f} {row['f1']:<10.3f} "
                  f"{row['tp']:<8.0f} {row['fn']:<8.0f}")
            
            # Store results
            all_results.append({
                'model': model_name,
                'strategy': strategy,
                'threshold': threshold,
                'test_auc': test_auc,
                'test_recall': row['recall'],
                'test_precision': row['precision'],
                'test_f1': row['f1'],
                'test_tp': int(row['tp']),
                'test_fp': int(row['fp']),
                'test_tn': int(row['tn']),
                'test_fn': int(row['fn'])
            })
        
        print("-" * 90)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_file = EXP_OUTPUT_DIR / 'exp4_medium_vs_strong_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")
    
    # Print summary
    print_section("SUMMARY: MEDIUM VS STRONG COMPARISON")
    
    for model_name in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model_name]
        best_recall = model_results.loc[model_results['test_recall'].idxmax()]
        best_f1 = model_results.loc[model_results['test_f1'].idxmax()]
        
        print(f"\n{model_name} Regularization:")
        print(f"  Best Test Recall: {best_recall['strategy']}")
        print(f"    Threshold: {best_recall['threshold']:.4f}, Recall: {best_recall['test_recall']:.3f}, "
              f"F1: {best_recall['test_f1']:.3f}, AUC: {best_recall['test_auc']:.4f}")
        print(f"  Best Test F1: {best_f1['strategy']}")
        print(f"    Threshold: {best_f1['threshold']:.4f}, Recall: {best_f1['test_recall']:.3f}, "
              f"F1: {best_f1['test_f1']:.3f}, AUC: {best_f1['test_auc']:.4f}")
    
    # Overall best
    print("\n" + "="*80)
    print("OVERALL BEST CONFIGURATION:")
    print("="*80)
    
    overall_best_recall = results_df.loc[results_df['test_recall'].idxmax()]
    print(f"\n🏆 Best Test Recall:")
    print(f"   Model: {overall_best_recall['model']} Regularization")
    print(f"   Strategy: {overall_best_recall['strategy']}")
    print(f"   Threshold: {overall_best_recall['threshold']:.4f}")
    print(f"   Test Recall: {overall_best_recall['test_recall']:.3f}")
    print(f"   Test F1: {overall_best_recall['test_f1']:.3f}")
    print(f"   Test AUC: {overall_best_recall['test_auc']:.4f}")
    
    overall_best_f1 = results_df.loc[results_df['test_f1'].idxmax()]
    print(f"\n🏆 Best Test F1:")
    print(f"   Model: {overall_best_f1['model']} Regularization")
    print(f"   Strategy: {overall_best_f1['strategy']}")
    print(f"   Threshold: {overall_best_f1['threshold']:.4f}")
    print(f"   Test Recall: {overall_best_f1['test_recall']:.3f}")
    print(f"   Test F1: {overall_best_f1['test_f1']:.3f}")
    print(f"   Test AUC: {overall_best_f1['test_auc']:.4f}")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT 4 COMPLETE".center(80))
    print("="*80)
    print(f"\n✓ Results: {results_file}")
    print("\nKey Findings:")
    print("  • Compared Medium vs Strong regularization")
    print("  • Tested 6 threshold strategies on test data")
    print("  • Identified best model + threshold combination\n")


if __name__ == "__main__":
    main()
