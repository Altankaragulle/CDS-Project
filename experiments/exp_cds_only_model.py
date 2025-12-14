"""
EXPERIMENT: CDS-Only Model - Baseline Comparison

Goal: Train LightGBM using ONLY CDS spread features to establish a baseline
      and compare against the full model with fundamentals + market data.

This helps answer: "How much value do fundamentals and market data add
                    beyond just using CDS spreads?"

Expected: CDS-only model should have lower performance than full model,
          demonstrating the value of comprehensive feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
EXP_OUTPUT_DIR = OUTPUT_DIR / 'experiments'
EXP_MODELS_DIR = EXP_OUTPUT_DIR / 'models'
EXP_FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures' / 'experiments'

# Create directories
EXP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def identify_cds_features(all_features):
    """
    Identify CDS-related features from the full feature list.
    
    CDS features include:
        - cds_spread_lag1, cds_spread_lag4
        - cds_spread_change_1q, cds_spread_change_1y (if temporal features exist)
        - Any other features with 'cds' in the name
    """
    cds_features = []
    
    for feature in all_features:
        if 'cds' in feature.lower():
            cds_features.append(feature)
    
    return cds_features


def load_data_and_identify_features():
    """Load train/test data and identify CDS features."""
    print_section("STEP 1: LOAD DATA & IDENTIFY CDS FEATURES")
    
    # Load data
    print("Loading train/test data...")
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    
    print(f"  ✓ Train: {train_df.shape}")
    print(f"  ✓ Test: {test_df.shape}")
    
    # Load full feature list
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    all_features = feature_list['feature'].tolist()
    
    print(f"\n  ✓ Total features in full model: {len(all_features)}")
    
    # Identify CDS features
    cds_features = identify_cds_features(all_features)
    
    print(f"\n  ✓ CDS features identified: {len(cds_features)}")
    print("\nCDS Features:")
    for feat in cds_features:
        print(f"    • {feat}")
    
    # Check which features exist in the data
    available_cds_features = [f for f in cds_features if f in train_df.columns]
    
    if len(available_cds_features) < len(cds_features):
        print(f"\n  ⚠️  Note: {len(cds_features) - len(available_cds_features)} CDS features not found in data")
        print("     (This is normal if temporal features haven't been created yet)")
    
    print(f"\n  ✓ Available CDS features: {len(available_cds_features)}")
    
    return train_df, test_df, available_cds_features, all_features


def train_cds_only_model(train_df, test_df, cds_features):
    """
    Train LightGBM model using ONLY CDS features.
    """
    print_section("STEP 2: TRAIN CDS-ONLY MODEL")
    
    # Prepare data
    X_train = train_df[cds_features].copy()
    y_train = train_df['distress_flag'].copy()
    X_test = test_df[cds_features].copy()
    y_test = test_df['distress_flag'].copy()
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Features: {len(cds_features)} (CDS-only)")
    print(f"Distress rate - Train: {y_train.mean()*100:.1f}%, Test: {y_test.mean()*100:.1f}%")
    
    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_processed = scaler.transform(imputer.transform(X_test))
    
    X_train_processed = pd.DataFrame(X_train_processed, columns=cds_features, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=cds_features, index=X_test.index)
    
    # Model configuration (Medium Regularization - same as full model)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print(f"\nTraining LightGBM (CDS-Only, Medium Regularization)...")
    print(f"  • Class imbalance handling: scale_pos_weight={scale_pos_weight:.2f}")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=80,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=0.3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_processed, y_train)
    
    print("✓ Training complete")
    
    # Predictions
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
    
    print(f"\nPerformance (Default Threshold 0.50):")
    print(f"  Train AUC: {results['train_auc']:.4f} | Test AUC: {results['test_auc']:.4f}")
    print(f"  Train Recall: {results['train_recall']:.4f} | Test Recall: {results['test_recall']:.4f}")
    print(f"  Train Precision: {results['train_precision']:.4f} | Test Precision: {results['test_precision']:.4f}")
    print(f"  Train F1: {results['train_f1']:.4f} | Test F1: {results['test_f1']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    total_distressed = y_test.sum()
    
    print(f"\nBusiness Impact:")
    print(f"  Distressed firms caught: {tp} / {total_distressed} ({tp/total_distressed*100:.1f}%)")
    print(f"  Distressed firms missed: {fn} ({fn/total_distressed*100:.1f}%)")
    
    # Save model
    model_file = EXP_MODELS_DIR / 'lightgbm_cds_only_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'imputer': imputer,
            'scaler': scaler,
            'features': cds_features,
            'performance': results
        }, f)
    print(f"\n✓ Model saved: {model_file}")
    
    return model, results, X_test_processed, y_test, y_test_proba, y_test_pred


def compare_to_full_model(cds_results):
    """
    Compare CDS-only model to full model.
    """
    print_section("STEP 3: COMPARISON TO FULL MODEL")
    
    # Full model results (from your main pipeline)
    # These are baseline values - will be updated when you run the full model
    full_model_results = {
        'test_auc': 0.6431,      # From your Medium Regularization model
        'test_recall': 0.4641,    # Default threshold
        'test_precision': 0.3422,
        'test_f1': 0.3940
    }
    
    print("PERFORMANCE COMPARISON:")
    print("="*90)
    print(f"{'Metric':<20} {'CDS-Only':<15} {'Full Model':<15} {'Difference':<15} {'% Change':<15}")
    print("="*90)
    
    metrics = ['test_auc', 'test_recall', 'test_precision', 'test_f1']
    metric_names = ['AUC', 'Recall', 'Precision', 'F1-Score']
    
    for metric, name in zip(metrics, metric_names):
        cds_val = cds_results[metric]
        full_val = full_model_results[metric]
        diff = full_val - cds_val
        pct_change = (diff / cds_val) * 100 if cds_val != 0 else 0
        
        marker = "✅" if diff > 0 else "⚠️" if diff == 0 else "❌"
        print(f"{marker} {name:<18} {cds_val:<15.4f} {full_val:<15.4f} {diff:>+7.4f} {pct_change:>+7.1f}%")
    
    print("="*90)
    
    # Summary
    auc_improvement = full_model_results['test_auc'] - cds_results['test_auc']
    recall_improvement = full_model_results['test_recall'] - cds_results['test_recall']
    
    print(f"\n📊 KEY INSIGHTS:")
    print(f"   AUC Improvement: {auc_improvement:+.4f} ({auc_improvement/cds_results['test_auc']*100:+.1f}%)")
    print(f"   Recall Improvement: {recall_improvement:+.4f} ({recall_improvement/cds_results['test_recall']*100:+.1f}%)")
    
    if auc_improvement > 0.05:
        print(f"\n   ✅ SIGNIFICANT: Full model shows substantial improvement over CDS-only")
        print(f"      → Fundamentals and market data add significant predictive value")
    elif auc_improvement > 0:
        print(f"\n   ⚠️  MODERATE: Full model shows some improvement over CDS-only")
        print(f"      → Additional features provide incremental value")
    else:
        print(f"\n   ❌ SURPRISING: CDS-only model performs as well or better")
        print(f"      → May indicate CDS spreads already capture most distress signals")
    
    return full_model_results


def create_visualizations(model, cds_features, y_test, y_test_proba, y_test_pred, cds_results, full_model_results):
    """
    Create comprehensive visualizations comparing CDS-only vs Full model.
    """
    print_section("STEP 4: GENERATE VISUALIZATIONS")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: ROC Curve Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    ax1.plot(fpr, tpr, linewidth=3, color='steelblue', 
            label=f'CDS-Only (AUC={cds_results["test_auc"]:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.3, label='Random')
    
    ax1.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
    ax1.set_title('ROC Curve: CDS-Only Model', fontweight='bold', fontsize=12, pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    ax2.plot(recall, precision, linewidth=3, color='darkorange', label='CDS-Only')
    ax2.axhline(y=y_test.mean(), color='red', linestyle='--', linewidth=2, 
                alpha=0.5, label=f'Baseline ({y_test.mean():.3f})')
    
    ax2.set_xlabel('Recall', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Precision', fontweight='bold', fontsize=11)
    ax2.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12, pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Plot 3: Feature Importance
    ax3 = fig.add_subplot(gs[0, 2])
    
    importance_df = pd.DataFrame({
        'feature': cds_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    ax3.barh(importance_df['feature'], importance_df['importance'], 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Importance', fontweight='bold', fontsize=11)
    ax3.set_title('CDS Feature Importance', fontweight='bold', fontsize=12, pad=10)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4,
               xticklabels=['Pred: No', 'Pred: Yes'],
               yticklabels=['True: No', 'True: Yes'])
    ax4.set_title('Confusion Matrix\n(CDS-Only Model)', fontweight='bold', fontsize=12, pad=10)
    
    # Plot 5: Model Comparison - AUC
    ax5 = fig.add_subplot(gs[1, 1])
    
    models = ['CDS-Only', 'Full Model']
    aucs = [cds_results['test_auc'], full_model_results['test_auc']]
    colors = ['steelblue', 'darkgreen']
    
    bars = ax5.bar(models, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('AUC', fontweight='bold', fontsize=11)
    ax5.set_title('AUC Comparison', fontweight='bold', fontsize=12, pad=10)
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    for bar, val in zip(bars, aucs):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 6: Model Comparison - Recall
    ax6 = fig.add_subplot(gs[1, 2])
    
    recalls = [cds_results['test_recall'], full_model_results['test_recall']]
    
    bars = ax6.bar(models, recalls, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Recall', fontweight='bold', fontsize=11)
    ax6.set_title('Recall Comparison', fontweight='bold', fontsize=12, pad=10)
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=0.55, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target')
    ax6.legend(fontsize=9)
    
    for bar, val in zip(bars, recalls):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 7: Metrics Comparison
    ax7 = fig.add_subplot(gs[2, :])
    
    metrics = ['AUC', 'Recall', 'Precision', 'F1-Score']
    cds_vals = [cds_results['test_auc'], cds_results['test_recall'], 
                cds_results['test_precision'], cds_results['test_f1']]
    full_vals = [full_model_results['test_auc'], full_model_results['test_recall'],
                 full_model_results['test_precision'], full_model_results['test_f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax7.bar(x - width/2, cds_vals, width, label='CDS-Only', 
           color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax7.bar(x + width/2, full_vals, width, label='Full Model', 
           color='darkgreen', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax7.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax7.set_title('Complete Metrics Comparison: CDS-Only vs Full Model', 
                 fontweight='bold', fontsize=14, pad=15)
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics)
    ax7.legend(fontsize=11, loc='upper right')
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_ylim([0, 1])
    
    # Add value labels
    for i, (cds_v, full_v) in enumerate(zip(cds_vals, full_vals)):
        ax7.text(i - width/2, cds_v + 0.02, f'{cds_v:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax7.text(i + width/2, full_v + 0.02, f'{full_v:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle('CDS-Only Model Analysis & Comparison', 
                fontweight='bold', fontsize=16, y=0.995)
    
    output_file = EXP_FIGURES_DIR / 'cds_only_model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_file}")
    
    plt.close()


def main():
    """
    Main execution: CDS-only model experiment.
    """
    print("\n" + "="*80)
    print("EXPERIMENT: CDS-ONLY MODEL - BASELINE COMPARISON".center(80))
    print("="*80)
    
    # Load data and identify CDS features
    train_df, test_df, cds_features, all_features = load_data_and_identify_features()
    
    if len(cds_features) == 0:
        print("\n❌ ERROR: No CDS features found in the data!")
        print("   Make sure the data contains CDS-related features.")
        return
    
    # Train CDS-only model
    model, cds_results, X_test, y_test, y_test_proba, y_test_pred = train_cds_only_model(
        train_df, test_df, cds_features
    )
    
    # Compare to full model
    full_model_results = compare_to_full_model(cds_results)
    
    # Create visualizations
    create_visualizations(model, cds_features, y_test, y_test_proba, y_test_pred,
                         cds_results, full_model_results)
    
    # Save results
    results_df = pd.DataFrame([
        {'model': 'CDS-Only', 'features': len(cds_features), **cds_results},
        {'model': 'Full Model', 'features': len(all_features), **full_model_results}
    ])
    
    results_file = EXP_OUTPUT_DIR / 'cds_only_comparison_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")
    
    # Final summary
    print_section("✅ CDS-ONLY MODEL EXPERIMENT COMPLETE")
    
    print("🎯 SUMMARY:")
    print(f"  CDS-Only Model:")
    print(f"    • Features: {len(cds_features)} (CDS spreads only)")
    print(f"    • Test AUC: {cds_results['test_auc']:.4f}")
    print(f"    • Test Recall: {cds_results['test_recall']:.4f}")
    print(f"    • Test F1: {cds_results['test_f1']:.4f}")
    
    print(f"\n  Full Model:")
    print(f"    • Features: {len(all_features)} (fundamentals + market + CDS)")
    print(f"    • Test AUC: {full_model_results['test_auc']:.4f}")
    print(f"    • Test Recall: {full_model_results['test_recall']:.4f}")
    print(f"    • Test F1: {full_model_results['test_f1']:.4f}")
    
    auc_diff = full_model_results['test_auc'] - cds_results['test_auc']
    recall_diff = full_model_results['test_recall'] - cds_results['test_recall']
    
    print(f"\n  📊 Value Added by Full Model:")
    print(f"    • AUC: {auc_diff:+.4f} ({auc_diff/cds_results['test_auc']*100:+.1f}%)")
    print(f"    • Recall: {recall_diff:+.4f} ({recall_diff/cds_results['test_recall']*100:+.1f}%)")
    
    print(f"\n💡 INTERPRETATION:")
    if auc_diff > 0.05:
        print(f"   ✅ Fundamentals and market data significantly improve prediction")
        print(f"      → Full feature engineering is valuable")
    elif auc_diff > 0:
        print(f"   ⚠️  Fundamentals and market data provide moderate improvement")
        print(f"      → CDS spreads capture most of the distress signal")
    else:
        print(f"   ❌ CDS spreads alone are as predictive as full model")
        print(f"      → Consider simplifying to CDS-only for efficiency")
    
    print(f"\n✓ Model saved: {EXP_MODELS_DIR / 'lightgbm_cds_only_model.pkl'}")
    print(f"✓ Figures saved: {EXP_FIGURES_DIR}")
    print("\nThis baseline helps quantify the value of comprehensive feature engineering!\n")


if __name__ == "__main__":
    main()
