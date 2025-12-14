"""
EXPERIMENT 1: Reduce Overfitting with Stronger Regularization

Goal: Train LightGBM models with different regularization levels to reduce
      the train/test gap while maintaining test performance.

Strategy:
    1. Load same train/test data as base pipeline
    2. Train multiple LightGBM variants with increasing regularization
    3. Compare train vs test performance
    4. Identify best trade-off

Outputs:
    - Models: output/experiments/models/*.pkl
    - Results: output/experiments/overfitting_experiment.csv
    - Figures: report/figures/experiments/overfitting_comparison.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
EXP_OUTPUT_DIR = OUTPUT_DIR / 'experiments'
EXP_MODELS_DIR = EXP_OUTPUT_DIR / 'models'
EXP_FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures' / 'experiments'

# Create directories
EXP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_data():
    """
    Load preprocessed train/test data (same as base pipeline).
    """
    print("Loading train/test data...")
    
    # Load datasets
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
    print(f"  ✓ Test: {X_test.shape}")
    print(f"  ✓ Features: {len(features)}")
    print(f"  ✓ Train distress rate: {y_train.mean()*100:.1f}%")
    print(f"  ✓ Test distress rate: {y_test.mean()*100:.1f}%\n")
    
    return X_train, X_test, y_train, y_test, features


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluate model on train and test sets.
    """
    # Train predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = model.predict(X_train)
    
    # Test predictions
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)
    
    # Metrics
    results = {
        'model': model_name,
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred)
    }
    
    # Calculate overfitting gap
    results['auc_gap'] = results['train_auc'] - results['test_auc']
    results['f1_gap'] = results['train_f1'] - results['test_f1']
    
    return results


def train_lightgbm_variants(X_train, y_train, X_test, y_test):
    """
    Train multiple LightGBM variants with different regularization levels.
    """
    print_section("TRAINING LIGHTGBM VARIANTS")
    
    try:
        import lightgbm as lgb
    except ImportError:
        print("❌ LightGBM not installed. Install with: pip install lightgbm")
        return []
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}")
    print()
    
    # Define model configurations
    configs = [
        {
            'name': 'Original (Baseline)',
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'verbose': -1
            }
        },
        {
            'name': 'Light Regularization',
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.08,
                'num_leaves': 25,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_samples': 30,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'verbose': -1
            }
        },
        {
            'name': 'Medium Regularization',
            'params': {
                'n_estimators': 80,
                'max_depth': 4,
                'learning_rate': 0.05,
                'num_leaves': 20,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_samples': 50,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'verbose': -1
            }
        },
        {
            'name': 'Strong Regularization',
            'params': {
                'n_estimators': 60,
                'max_depth': 3,
                'learning_rate': 0.03,
                'num_leaves': 15,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'min_child_samples': 100,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'verbose': -1
            }
        },
        {
            'name': 'Very Strong Regularization',
            'params': {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.02,
                'num_leaves': 10,
                'subsample': 0.5,
                'colsample_bytree': 0.5,
                'min_child_samples': 150,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'verbose': -1
            }
        }
    ]
    
    results = []
    models = []
    
    for i, config in enumerate(configs, 1):
        print(f"{i}. Training: {config['name']}")
        print(f"   Key params: max_depth={config['params']['max_depth']}, "
              f"n_estimators={config['params']['n_estimators']}, "
              f"learning_rate={config['params']['learning_rate']}")
        
        # Train model
        model = lgb.LGBMClassifier(**config['params'])
        model.fit(X_train, y_train)
        
        # Evaluate
        result = evaluate_model(model, X_train, y_train, X_test, y_test, config['name'])
        results.append(result)
        models.append((config['name'], model))
        
        # Print results
        print(f"   Train AUC: {result['train_auc']:.4f} | Test AUC: {result['test_auc']:.4f} | Gap: {result['auc_gap']:.4f}")
        print(f"   Train F1:  {result['train_f1']:.4f} | Test F1:  {result['test_f1']:.4f} | Gap: {result['f1_gap']:.4f}")
        print()
        
        # Save model
        model_file = EXP_MODELS_DIR / f"lightgbm_{config['name'].lower().replace(' ', '_')}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    
    return results, models


def plot_results(results_df):
    """
    Create comprehensive visualization of results.
    """
    print_section("GENERATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Train vs Test AUC
    ax1 = fig.add_subplot(gs[0, 0])
    
    x = np.arange(len(results_df))
    width = 0.35
    
    ax1.bar(x - width/2, results_df['train_auc'], width, label='Train AUC', 
            color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, results_df['test_auc'], width, label='Test AUC', 
            color='darkorange', alpha=0.8)
    
    ax1.set_xlabel('Model Configuration', fontweight='bold')
    ax1.set_ylabel('AUC', fontweight='bold')
    ax1.set_title('Train vs Test AUC', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['model'], rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # Plot 2: Overfitting Gap (AUC)
    ax2 = fig.add_subplot(gs[0, 1])
    
    colors = ['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green' 
              for gap in results_df['auc_gap']]
    
    bars = ax2.bar(x, results_df['auc_gap'], color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, gap in zip(bars, results_df['auc_gap']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Model Configuration', fontweight='bold')
    ax2.set_ylabel('AUC Gap (Train - Test)', fontweight='bold')
    ax2.set_title('Overfitting Gap Analysis', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['model'], rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='10% threshold')
    ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, linewidth=1, label='20% threshold')
    ax2.legend(fontsize=8)
    
    # Plot 3: Test AUC vs Overfitting Gap (Scatter)
    ax3 = fig.add_subplot(gs[1, 0])
    
    scatter = ax3.scatter(results_df['auc_gap'], results_df['test_auc'], 
                         s=200, c=results_df['test_auc'], cmap='RdYlGn',
                         alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels
    for idx, row in results_df.iterrows():
        ax3.annotate(row['model'], 
                    (row['auc_gap'], row['test_auc']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax3.set_xlabel('Overfitting Gap (Train AUC - Test AUC)', fontweight='bold')
    ax3.set_ylabel('Test AUC', fontweight='bold')
    ax3.set_title('Trade-off: Test Performance vs Overfitting', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Test AUC')
    
    # Optimal region (high test AUC, low gap)
    ax3.axvline(x=0.1, color='green', linestyle='--', alpha=0.3, linewidth=2)
    ax3.axhline(y=0.63, color='green', linestyle='--', alpha=0.3, linewidth=2)
    ax3.text(0.05, 0.635, 'Optimal\nRegion', fontsize=9, color='green', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Plot 4: F1 Score Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.bar(x - width/2, results_df['train_f1'], width, label='Train F1', 
            color='steelblue', alpha=0.8)
    ax4.bar(x + width/2, results_df['test_f1'], width, label='Test F1', 
            color='darkorange', alpha=0.8)
    
    ax4.set_xlabel('Model Configuration', fontweight='bold')
    ax4.set_ylabel('F1 Score', fontweight='bold')
    ax4.set_title('Train vs Test F1 Score', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(results_df['model'], rotation=45, ha='right', fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_file = EXP_FIGURES_DIR / 'overfitting_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def generate_report(results_df):
    """
    Generate summary report and recommendations.
    """
    print_section("EXPERIMENT SUMMARY")
    
    # Save results
    results_file = EXP_OUTPUT_DIR / 'overfitting_experiment.csv'
    results_df.to_csv(results_file, index=False)
    print(f"✓ Saved results: {results_file}\n")
    
    # Print comparison table
    print("RESULTS COMPARISON:")
    print("-" * 100)
    print(f"{'Model':<30} {'Train AUC':<12} {'Test AUC':<12} {'AUC Gap':<12} {'Test F1':<12}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:<30} {row['train_auc']:<12.4f} {row['test_auc']:<12.4f} "
              f"{row['auc_gap']:<12.4f} {row['test_f1']:<12.4f}")
    
    print("-" * 100)
    
    # Find best models
    print("\nKEY FINDINGS:\n")
    
    # Best test AUC
    best_test_auc = results_df.loc[results_df['test_auc'].idxmax()]
    print(f"1. Best Test AUC: {best_test_auc['model']}")
    print(f"   Test AUC: {best_test_auc['test_auc']:.4f}")
    print(f"   Overfitting Gap: {best_test_auc['auc_gap']:.4f}")
    
    # Lowest overfitting
    lowest_gap = results_df.loc[results_df['auc_gap'].idxmin()]
    print(f"\n2. Lowest Overfitting: {lowest_gap['model']}")
    print(f"   Test AUC: {lowest_gap['test_auc']:.4f}")
    print(f"   Overfitting Gap: {lowest_gap['auc_gap']:.4f}")
    
    # Best trade-off (high test AUC, low gap)
    results_df['score'] = results_df['test_auc'] - 0.5 * results_df['auc_gap']
    best_tradeoff = results_df.loc[results_df['score'].idxmax()]
    print(f"\n3. Best Trade-off: {best_tradeoff['model']}")
    print(f"   Test AUC: {best_tradeoff['test_auc']:.4f}")
    print(f"   Overfitting Gap: {best_tradeoff['auc_gap']:.4f}")
    print(f"   Trade-off Score: {best_tradeoff['score']:.4f}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if best_tradeoff['auc_gap'] < 0.15:
        print("\n✅ SUCCESS: Found configuration with low overfitting (<15% gap)")
        print(f"   → Use: {best_tradeoff['model']}")
        print(f"   → Test AUC: {best_tradeoff['test_auc']:.4f}")
        print(f"   → Overfitting reduced by: {(results_df.iloc[0]['auc_gap'] - best_tradeoff['auc_gap'])*100:.1f}%")
    elif best_tradeoff['auc_gap'] < 0.20:
        print("\n⚠️  MODERATE: Best configuration still has 15-20% gap")
        print(f"   → Use: {best_tradeoff['model']}")
        print(f"   → Consider: This is acceptable for complex models")
    else:
        print("\n⚠️  HIGH OVERFITTING: All configurations show >20% gap")
        print(f"   → Best option: {best_tradeoff['model']}")
        print(f"   → Consider: Simpler model (Logistic Regression) or more data")
    
    print("\n" + "="*80)


def main():
    """
    Main execution: Overfitting reduction experiment.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: REDUCE OVERFITTING".center(80))
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test, features = load_data()
    
    # Train variants
    results, models = train_lightgbm_variants(X_train, y_train, X_test, y_test)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    plot_results(results_df)
    
    # Generate report
    generate_report(results_df)
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE".center(80))
    print("="*80)
    print(f"\n✓ Models saved to: {EXP_MODELS_DIR}")
    print(f"✓ Results saved to: {EXP_OUTPUT_DIR}")
    print(f"✓ Figures saved to: {EXP_FIGURES_DIR}")
    print("\nNext: Run exp2_model_comparison.py to compare with original model\n")


if __name__ == "__main__":
    main()
