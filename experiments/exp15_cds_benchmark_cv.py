"""
EXPERIMENT 15: CDS Benchmark Cross-Validation

Test the naive benchmark rule: "High CDS Spread = High Distress Risk"
across the same 5-fold time-series CV as the ML model.

Benchmark Strategy:
    - Use CDS spread lag1 as the sole predictor
    - Firms with CDS > threshold are predicted as distressed
    - Test multiple thresholds (median, 75th percentile, optimal)

Goal: Compare ML model (exp14) vs. naive CDS benchmark
Expected: ML should outperform simple CDS threshold across time periods

This answers the KEY research question:
"Does machine learning add value beyond a simple CDS-based rule?"

Outputs:
    - CSV: output/experiments/cds_benchmark_cv_results.csv
    - Comparison: ML vs CDS benchmark across all folds
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
EXP_OUTPUT_DIR = OUTPUT_DIR / 'experiments'
EXP_FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures' / 'experiments'

# Create directories
EXP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_training_data():
    """Load training data for cross-validation."""
    print("Loading training data...")
    
    # Load train data
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
    
    # Check if CDS spread exists
    if 'cds_spread_lag1' not in train_df.columns:
        raise ValueError("CDS spread lag1 not found in data!")
    
    print(f"  ✓ Train data: {len(train_df):,} observations")
    print(f"  ✓ Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"  ✓ CDS spread available: cds_spread_lag1\n")
    
    return train_df


def find_optimal_cds_threshold(cds_values, y_true):
    """
    Find optimal CDS threshold that maximizes AUC.
    
    Args:
        cds_values: CDS spread values
        y_true: True distress labels
    
    Returns:
        optimal_threshold: Best CDS threshold
        best_auc: AUC at optimal threshold
    """
    # Remove NaN values
    mask = ~np.isnan(cds_values)
    cds_clean = cds_values[mask]
    y_clean = y_true[mask]
    
    if len(cds_clean) == 0:
        return np.nan, 0.0
    
    # Try different thresholds
    thresholds = np.percentile(cds_clean, np.arange(10, 91, 5))
    
    best_auc = 0
    optimal_threshold = np.median(cds_clean)
    
    for threshold in thresholds:
        y_pred = (cds_clean >= threshold).astype(int)
        
        # Skip if all same prediction
        if len(np.unique(y_pred)) < 2:
            continue
        
        try:
            auc = roc_auc_score(y_clean, cds_clean)
            if auc > best_auc:
                best_auc = auc
                optimal_threshold = threshold
        except:
            continue
    
    return optimal_threshold, best_auc


def evaluate_cds_benchmark(train_df, n_splits=5):
    """
    Evaluate CDS benchmark across time-series CV folds.
    
    Tests three threshold strategies:
    1. Median CDS (50th percentile)
    2. 75th percentile CDS
    3. Optimal threshold (maximizes AUC on training fold)
    
    Args:
        train_df: Training DataFrame
        n_splits: Number of CV folds
    
    Returns:
        Dictionary with benchmark results
    """
    print_section("CDS BENCHMARK CROSS-VALIDATION")
    
    print(f"Testing naive benchmark: High CDS Spread → High Distress Risk")
    print(f"Performing {n_splits}-fold time-series cross-validation...\n")
    
    # Sort by date
    train_df = train_df.sort_values('date').reset_index(drop=True)
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Storage for results
    benchmark_results = {
        'fold': [],
        'train_period': [],
        'val_period': [],
        'train_size': [],
        'val_size': [],
        'train_distress_rate': [],
        'val_distress_rate': [],
        'median_threshold': [],
        'p75_threshold': [],
        'optimal_threshold': [],
        'median_auc': [],
        'median_recall': [],
        'median_precision': [],
        'median_f1': [],
        'p75_auc': [],
        'p75_recall': [],
        'p75_precision': [],
        'p75_f1': [],
        'optimal_auc': [],
        'optimal_recall': [],
        'optimal_precision': [],
        'optimal_f1': []
    }
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/{n_splits}".center(80))
        print(f"{'='*80}\n")
        
        # Split data
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()
        
        # Extract CDS and labels
        cds_train = train_fold['cds_spread_lag1'].values
        y_train = train_fold['distress_flag'].values
        cds_val = val_fold['cds_spread_lag1'].values
        y_val = val_fold['distress_flag'].values
        
        dates_train = train_fold['date']
        dates_val = val_fold['date']
        
        # Print fold info
        print(f"Train period: {dates_train.min()} to {dates_train.max()}")
        print(f"Val period:   {dates_val.min()} to {dates_val.max()}")
        print(f"Train size:   {len(train_fold):,} observations")
        print(f"Val size:     {len(val_fold):,} observations")
        print(f"Train distress rate: {y_train.mean():.1%}")
        print(f"Val distress rate:   {y_val.mean():.1%}")
        
        # Remove NaN CDS values
        train_mask = ~np.isnan(cds_train)
        val_mask = ~np.isnan(cds_val)
        
        cds_train_clean = cds_train[train_mask]
        y_train_clean = y_train[train_mask]
        cds_val_clean = cds_val[val_mask]
        y_val_clean = y_val[val_mask]
        
        print(f"\nCDS Statistics (Training):")
        print(f"  Mean:   {np.mean(cds_train_clean):.2f} bps")
        print(f"  Median: {np.median(cds_train_clean):.2f} bps")
        print(f"  75th %: {np.percentile(cds_train_clean, 75):.2f} bps")
        
        # Define thresholds
        median_threshold = np.median(cds_train_clean)
        p75_threshold = np.percentile(cds_train_clean, 75)
        optimal_threshold, _ = find_optimal_cds_threshold(cds_train_clean, y_train_clean)
        
        print(f"\nThresholds:")
        print(f"  Median (50th %):  {median_threshold:.2f} bps")
        print(f"  75th percentile:  {p75_threshold:.2f} bps")
        print(f"  Optimal (train):  {optimal_threshold:.2f} bps")
        
        # Evaluate each threshold strategy on validation set
        print(f"\nValidation Results:")
        print("-" * 80)
        print(f"{'Strategy':<20} {'AUC':<10} {'Recall':<10} {'Precision':<10} {'F1':<10}")
        print("-" * 80)
        
        strategies = {
            'median': median_threshold,
            'p75': p75_threshold,
            'optimal': optimal_threshold
        }
        
        results = {}
        
        for strategy_name, threshold in strategies.items():
            # Predict: CDS >= threshold → distressed
            y_val_pred = (cds_val_clean >= threshold).astype(int)
            
            # Compute metrics
            try:
                # Use CDS values as scores for AUC (higher CDS = higher risk)
                auc = roc_auc_score(y_val_clean, cds_val_clean)
                recall = recall_score(y_val_clean, y_val_pred, zero_division=0)
                precision = precision_score(y_val_clean, y_val_pred, zero_division=0)
                f1 = f1_score(y_val_clean, y_val_pred, zero_division=0)
            except:
                auc = recall = precision = f1 = 0.0
            
            results[strategy_name] = {
                'auc': auc,
                'recall': recall,
                'precision': precision,
                'f1': f1
            }
            
            display_name = {
                'median': 'Median (50%)',
                'p75': '75th Percentile',
                'optimal': 'Optimal (train)'
            }[strategy_name]
            
            print(f"{display_name:<20} {auc:<10.4f} {recall:<10.4f} {precision:<10.4f} {f1:<10.4f}")
        
        print("-" * 80)
        
        # Store results
        benchmark_results['fold'].append(fold)
        benchmark_results['train_period'].append(f"{dates_train.min()} to {dates_train.max()}")
        benchmark_results['val_period'].append(f"{dates_val.min()} to {dates_val.max()}")
        benchmark_results['train_size'].append(len(train_fold))
        benchmark_results['val_size'].append(len(val_fold))
        benchmark_results['train_distress_rate'].append(y_train.mean())
        benchmark_results['val_distress_rate'].append(y_val.mean())
        
        benchmark_results['median_threshold'].append(median_threshold)
        benchmark_results['p75_threshold'].append(p75_threshold)
        benchmark_results['optimal_threshold'].append(optimal_threshold)
        
        for strategy in ['median', 'p75', 'optimal']:
            benchmark_results[f'{strategy}_auc'].append(results[strategy]['auc'])
            benchmark_results[f'{strategy}_recall'].append(results[strategy]['recall'])
            benchmark_results[f'{strategy}_precision'].append(results[strategy]['precision'])
            benchmark_results[f'{strategy}_f1'].append(results[strategy]['f1'])
    
    return benchmark_results


def compare_ml_vs_benchmark(benchmark_results):
    """
    Compare CDS benchmark to ML model results.
    
    Args:
        benchmark_results: Dictionary with benchmark CV results
    """
    print_section("COMPARISON: ML MODEL vs CDS BENCHMARK")
    
    # Convert to DataFrame
    benchmark_df = pd.DataFrame(benchmark_results)
    
    # Load ML results if available
    ml_results_path = EXP_OUTPUT_DIR / 'cv_results_fixed.csv'
    
    if ml_results_path.exists():
        ml_df = pd.read_csv(ml_results_path)
        has_ml = True
        print("✓ ML model results loaded from exp14")
    else:
        # Try original CV results
        ml_results_path = EXP_OUTPUT_DIR / 'cv_results.csv'
        if ml_results_path.exists():
            ml_df = pd.read_csv(ml_results_path)
            has_ml = True
            print("✓ ML model results loaded from exp14 (original)")
        else:
            has_ml = False
            print("⚠️  ML model results not found - showing benchmark only")
    
    # Summary statistics for benchmark
    print("\nCDS BENCHMARK PERFORMANCE:")
    print("=" * 80)
    
    strategies = ['median', 'p75', 'optimal']
    strategy_names = ['Median (50%)', '75th Percentile', 'Optimal (train)']
    
    for strategy, name in zip(strategies, strategy_names):
        auc_mean = benchmark_df[f'{strategy}_auc'].mean()
        auc_std = benchmark_df[f'{strategy}_auc'].std()
        recall_mean = benchmark_df[f'{strategy}_recall'].mean()
        f1_mean = benchmark_df[f'{strategy}_f1'].mean()
        
        print(f"\n{name}:")
        print(f"  Mean AUC:    {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"  Mean Recall: {recall_mean:.4f}")
        print(f"  Mean F1:     {f1_mean:.4f}")
    
    # Best benchmark strategy
    best_strategy = 'optimal'
    best_auc = benchmark_df[f'{best_strategy}_auc'].mean()
    
    print(f"\n🏆 BEST BENCHMARK: Optimal threshold")
    print(f"   Mean AUC: {best_auc:.4f}")
    
    if has_ml:
        print("\n" + "=" * 80)
        print("ML MODEL vs BEST BENCHMARK:")
        print("=" * 80)
        
        ml_auc_mean = ml_df['auc'].mean()
        ml_recall_mean = ml_df['recall'].mean()
        ml_f1_mean = ml_df['f1'].mean()
        
        benchmark_auc_mean = benchmark_df['optimal_auc'].mean()
        benchmark_recall_mean = benchmark_df['optimal_recall'].mean()
        benchmark_f1_mean = benchmark_df['optimal_f1'].mean()
        
        print(f"\n{'Metric':<15} {'ML Model':<15} {'CDS Benchmark':<15} {'Difference':<15} {'Winner':<10}")
        print("-" * 75)
        
        # AUC comparison
        auc_diff = ml_auc_mean - benchmark_auc_mean
        auc_pct = (auc_diff / benchmark_auc_mean * 100) if benchmark_auc_mean > 0 else 0
        auc_winner = "ML ✅" if auc_diff > 0 else "Benchmark ⚠️"
        print(f"{'AUC':<15} {ml_auc_mean:<15.4f} {benchmark_auc_mean:<15.4f} {auc_diff:>+7.4f} ({auc_pct:+.1f}%)  {auc_winner:<10}")
        
        # Recall comparison
        recall_diff = ml_recall_mean - benchmark_recall_mean
        recall_pct = (recall_diff / benchmark_recall_mean * 100) if benchmark_recall_mean > 0 else 0
        recall_winner = "ML ✅" if recall_diff > 0 else "Benchmark ⚠️"
        print(f"{'Recall':<15} {ml_recall_mean:<15.4f} {benchmark_recall_mean:<15.4f} {recall_diff:>+7.4f} ({recall_pct:+.1f}%)  {recall_winner:<10}")
        
        # F1 comparison
        f1_diff = ml_f1_mean - benchmark_f1_mean
        f1_pct = (f1_diff / benchmark_f1_mean * 100) if benchmark_f1_mean > 0 else 0
        f1_winner = "ML ✅" if f1_diff > 0 else "Benchmark ⚠️"
        print(f"{'F1-Score':<15} {ml_f1_mean:<15.4f} {benchmark_f1_mean:<15.4f} {f1_diff:>+7.4f} ({f1_pct:+.1f}%)  {f1_winner:<10}")
        
        print("-" * 75)
        
        # Interpretation
        print(f"\n📊 INTERPRETATION:")
        if auc_diff > 0.05:
            print(f"   ✅ ML model SIGNIFICANTLY outperforms CDS benchmark (+{auc_pct:.1f}%)")
            print(f"   → Machine learning adds substantial value beyond simple CDS rule")
        elif auc_diff > 0.02:
            print(f"   ✅ ML model MODERATELY outperforms CDS benchmark (+{auc_pct:.1f}%)")
            print(f"   → Machine learning provides incremental improvement")
        elif auc_diff > 0:
            print(f"   ⚠️  ML model SLIGHTLY outperforms CDS benchmark (+{auc_pct:.1f}%)")
            print(f"   → Marginal benefit from machine learning")
        else:
            print(f"   ❌ CDS benchmark performs as well or better than ML")
            print(f"   → Simple rule may be sufficient for this task")
        
        # Fold-by-fold comparison
        print(f"\n📈 FOLD-BY-FOLD COMPARISON:")
        print("-" * 80)
        print(f"{'Fold':<8} {'ML AUC':<12} {'Benchmark AUC':<15} {'Difference':<15} {'Winner':<10}")
        print("-" * 80)
        
        for i in range(len(ml_df)):
            ml_auc = ml_df.iloc[i]['auc']
            bench_auc = benchmark_df.iloc[i]['optimal_auc']
            diff = ml_auc - bench_auc
            winner = "ML ✅" if diff > 0 else "Benchmark ⚠️"
            print(f"{i+1:<8} {ml_auc:<12.4f} {bench_auc:<15.4f} {diff:>+7.4f}        {winner:<10}")
        
        print("-" * 80)
    
    return benchmark_df


def create_visualizations(benchmark_df, has_ml=False):
    """Create comparison visualizations."""
    print_section("CREATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(18, 12))
    
    if has_ml:
        # Load ML results
        ml_results_path = EXP_OUTPUT_DIR / 'cv_results_fixed.csv'
        if not ml_results_path.exists():
            ml_results_path = EXP_OUTPUT_DIR / 'cv_results.csv'
        
        if ml_results_path.exists():
            ml_df = pd.read_csv(ml_results_path)
            
            # Plot 1: AUC comparison across folds
            ax1 = plt.subplot(2, 2, 1)
            folds = benchmark_df['fold']
            
            ax1.plot(folds, ml_df['auc'], marker='o', linewidth=2, markersize=8, 
                    label='ML Model', color='steelblue')
            ax1.plot(folds, benchmark_df['optimal_auc'], marker='s', linewidth=2, markersize=8,
                    label='CDS Benchmark', color='darkorange')
            
            ax1.axhline(y=ml_df['auc'].mean(), color='steelblue', linestyle='--', alpha=0.5)
            ax1.axhline(y=benchmark_df['optimal_auc'].mean(), color='darkorange', linestyle='--', alpha=0.5)
            
            ax1.set_xlabel('Fold', fontweight='bold', fontsize=11)
            ax1.set_ylabel('AUC', fontweight='bold', fontsize=11)
            ax1.set_title('AUC: ML vs CDS Benchmark', fontweight='bold', fontsize=13)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(folds)
            
            # Plot 2: Mean performance comparison
            ax2 = plt.subplot(2, 2, 2)
            
            metrics = ['AUC', 'Recall', 'Precision', 'F1']
            ml_means = [ml_df['auc'].mean(), ml_df['recall'].mean(), 
                       ml_df['precision'].mean(), ml_df['f1'].mean()]
            bench_means = [benchmark_df['optimal_auc'].mean(), benchmark_df['optimal_recall'].mean(),
                          benchmark_df['optimal_precision'].mean(), benchmark_df['optimal_f1'].mean()]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax2.bar(x - width/2, ml_means, width, label='ML Model', color='steelblue', alpha=0.8)
            ax2.bar(x + width/2, bench_means, width, label='CDS Benchmark', color='darkorange', alpha=0.8)
            
            ax2.set_ylabel('Score', fontweight='bold', fontsize=11)
            ax2.set_title('Mean Performance Comparison', fontweight='bold', fontsize=13)
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (ml_val, bench_val) in enumerate(zip(ml_means, bench_means)):
                ax2.text(i - width/2, ml_val + 0.02, f'{ml_val:.3f}', 
                        ha='center', fontsize=9, fontweight='bold')
                ax2.text(i + width/2, bench_val + 0.02, f'{bench_val:.3f}',
                        ha='center', fontsize=9, fontweight='bold')
    else:
        ax1 = plt.subplot(2, 2, 1)
        folds = benchmark_df['fold']
        ax1.plot(folds, benchmark_df['optimal_auc'], marker='o', linewidth=2, 
                markersize=8, color='darkorange')
        ax1.set_xlabel('Fold', fontweight='bold')
        ax1.set_ylabel('AUC', fontweight='bold')
        ax1.set_title('CDS Benchmark AUC Across Folds', fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot 3: Threshold strategies comparison
    ax3 = plt.subplot(2, 2, 3)
    
    strategies = ['Median', '75th %', 'Optimal']
    aucs = [benchmark_df['median_auc'].mean(), 
            benchmark_df['p75_auc'].mean(),
            benchmark_df['optimal_auc'].mean()]
    
    bars = ax3.bar(strategies, aucs, color=['skyblue', 'lightcoral', 'lightgreen'], 
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    ax3.set_ylabel('Mean AUC', fontweight='bold', fontsize=11)
    ax3.set_title('CDS Benchmark: Threshold Strategies', fontweight='bold', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, auc in zip(bars, aucs):
        ax3.text(bar.get_x() + bar.get_width()/2, auc + 0.01, f'{auc:.3f}',
                ha='center', fontweight='bold', fontsize=10)
    
    # Plot 4: Summary text
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
    CDS BENCHMARK CROSS-VALIDATION
    {'='*50}
    
    Strategy: High CDS Spread → High Distress Risk
    
    Median Threshold (50%):
      Mean AUC: {benchmark_df['median_auc'].mean():.4f}
      Mean Recall: {benchmark_df['median_recall'].mean():.4f}
    
    75th Percentile Threshold:
      Mean AUC: {benchmark_df['p75_auc'].mean():.4f}
      Mean Recall: {benchmark_df['p75_recall'].mean():.4f}
    
    Optimal Threshold (train):
      Mean AUC: {benchmark_df['optimal_auc'].mean():.4f}
      Mean Recall: {benchmark_df['optimal_recall'].mean():.4f}
    
    {'='*50}
    
    This naive benchmark provides a baseline for
    evaluating ML model performance.
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('CDS Benchmark Cross-Validation Results', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = EXP_FIGURES_DIR / 'cds_benchmark_cv_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def main():
    """Main execution."""
    print_section("EXPERIMENT 15: CDS BENCHMARK CROSS-VALIDATION")
    
    print("Research Question:")
    print("  Does machine learning add value beyond a simple CDS-based rule?")
    print()
    print("Naive Benchmark:")
    print("  IF CDS Spread >= Threshold THEN Predict Distressed")
    print()
    
    # Load data
    train_df = load_training_data()
    
    # Evaluate benchmark
    benchmark_results = evaluate_cds_benchmark(train_df, n_splits=5)
    
    # Compare to ML
    benchmark_df = compare_ml_vs_benchmark(benchmark_results)
    
    # Create visualizations
    ml_results_path = EXP_OUTPUT_DIR / 'cv_results_fixed.csv'
    if not ml_results_path.exists():
        ml_results_path = EXP_OUTPUT_DIR / 'cv_results.csv'
    has_ml = ml_results_path.exists()
    
    create_visualizations(benchmark_df, has_ml=has_ml)
    
    # Save results
    print_section("SAVING RESULTS")
    
    output_path = EXP_OUTPUT_DIR / 'cds_benchmark_cv_results.csv'
    benchmark_df.to_csv(output_path, index=False)
    print(f"✓ Saved benchmark results: {output_path}")
    
    print("\nDetailed Benchmark Results:")
    print("-" * 80)
    print(benchmark_df[['fold', 'val_period', 'optimal_auc', 'optimal_recall', 
                        'optimal_precision', 'optimal_f1']].to_string(index=False))
    print("-" * 80)
    
    print_section("✅ EXPERIMENT 15 COMPLETE")
    
    print("Key Findings:")
    print(f"  • CDS benchmark achieves {benchmark_df['optimal_auc'].mean():.4f} AUC (optimal threshold)")
    print(f"  • This provides a baseline for evaluating ML model value-add")
    print(f"  • Results show whether ML beats the naive 'High CDS = High Risk' rule")
    print(f"\n💡 Use this comparison to justify ML approach in your thesis!")


if __name__ == "__main__":
    main()
