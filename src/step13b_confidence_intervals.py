"""
STEP 13B: Bootstrap Confidence Intervals

Compute statistical confidence intervals for model performance metrics using bootstrap resampling.

This provides:
- Mean and standard deviation of metrics
- 95% confidence intervals
- Statistical rigor for academic reporting

Outputs:
    - CSV: output/confidence_intervals.csv
    - Console: Detailed CI statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_model_and_data():
    """Load trained model and test data."""
    print("Loading model and test data...")
    
    # Load model
    with open(MODELS_DIR / 'lightgbm_optimized.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle dict format
    if isinstance(model_data, dict):
        model = model_data.get('model', model_data.get('lightgbm', None))
    else:
        model = model_data
    
    # Load test data
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
    
    # Load feature list
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    features = feature_list['feature'].tolist()
    
    # Extract X and y
    X_test = test_df[features]
    y_test = test_df['distress_flag']
    
    print(f"  ✓ Model loaded")
    print(f"  ✓ Test data: {len(X_test):,} observations")
    print(f"  ✓ Features: {len(features)}\n")
    
    return model, X_test, y_test


def compute_bootstrap_ci(model, X_test, y_test, n_bootstrap=1000, random_state=42):
    """
    Compute bootstrap confidence intervals for model metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with mean, std, and 95% CI for each metric
    """
    print_section("BOOTSTRAP CONFIDENCE INTERVALS")
    
    print(f"Computing {n_bootstrap:,} bootstrap samples...")
    print("This may take a few minutes...\n")
    
    # Initialize storage for metrics
    metrics = {
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Bootstrap resampling
    np.random.seed(random_state)
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1:,}/{n_bootstrap:,} ({(i+1)/n_bootstrap*100:.0f}%)")
        
        # Resample with replacement
        indices = resample(range(len(X_test)), random_state=random_state + i)
        X_boot = X_test.iloc[indices]
        y_boot = y_test.iloc[indices]
        
        # Skip if bootstrap sample has only one class
        if len(y_boot.unique()) < 2:
            continue
        
        try:
            # Get predictions
            y_pred = model.predict(X_boot)
            y_proba = model.predict_proba(X_boot)[:, 1]
            
            # Compute metrics
            metrics['auc'].append(roc_auc_score(y_boot, y_proba))
            metrics['accuracy'].append(accuracy_score(y_boot, y_pred))
            
            # Handle zero division in precision/recall
            precision = precision_score(y_boot, y_pred, zero_division=0)
            recall = recall_score(y_boot, y_pred, zero_division=0)
            f1 = f1_score(y_boot, y_pred, zero_division=0)
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            
        except Exception as e:
            # Skip problematic bootstrap samples
            continue
    
    print(f"\n✓ Completed {len(metrics['auc']):,} successful bootstrap iterations\n")
    
    # Compute statistics
    results = {}
    
    print("="*80)
    print("CONFIDENCE INTERVAL RESULTS".center(80))
    print("="*80 + "\n")
    
    for metric_name, values in metrics.items():
        if len(values) == 0:
            continue
            
        mean = np.mean(values)
        std = np.std(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        median = np.median(values)
        
        results[metric_name] = {
            'mean': mean,
            'std': std,
            'median': median,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }
        
        # Print results
        print(f"{metric_name.upper()}:")
        print(f"  Mean:      {mean:.4f}")
        print(f"  Std Dev:   {std:.4f}")
        print(f"  Median:    {median:.4f}")
        print(f"  95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  CI Width:  {ci_upper - ci_lower:.4f}")
        print()
    
    return results


def save_results(results):
    """Save confidence interval results to CSV."""
    print_section("SAVING RESULTS")
    
    # Convert to DataFrame
    rows = []
    for metric, stats in results.items():
        rows.append({
            'metric': metric,
            'mean': stats['mean'],
            'std': stats['std'],
            'median': stats['median'],
            'ci_95_lower': stats['ci_95_lower'],
            'ci_95_upper': stats['ci_95_upper'],
            'ci_width': stats['ci_width']
        })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = OUTPUT_DIR / 'confidence_intervals.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved confidence intervals: {output_path}")
    print()
    
    # Print formatted table
    print("Summary Table:")
    print("-" * 80)
    print(f"{'Metric':<12} {'Mean':<10} {'Std':<10} {'95% CI':<25}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        ci_str = f"[{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}]"
        print(f"{row['metric']:<12} {row['mean']:<10.4f} {row['std']:<10.4f} {ci_str:<25}")
    
    print("-" * 80)
    print()


def create_interpretation():
    """Provide interpretation of confidence intervals."""
    print_section("INTERPRETATION")
    
    print("What do these confidence intervals mean?")
    print()
    print("1. **Mean**: Average performance across 1,000 bootstrap samples")
    print("   - This is our best estimate of the model's true performance")
    print()
    print("2. **95% Confidence Interval**: Range where we expect the true")
    print("   performance to lie with 95% confidence")
    print("   - Narrower intervals = more precise estimates")
    print("   - Wider intervals = more uncertainty")
    print()
    print("3. **Standard Deviation**: Variability in performance estimates")
    print("   - Lower std = more stable performance")
    print("   - Higher std = performance varies more across samples")
    print()
    print("4. **Statistical Significance**:")
    print("   - If the CI for AUC doesn't include 0.5, the model is")
    print("     significantly better than random")
    print("   - If two models' CIs don't overlap, they are significantly different")
    print()
    print("✓ These intervals provide statistical rigor for academic reporting")
    print()


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("STEP 13B: BOOTSTRAP CONFIDENCE INTERVALS".center(80))
    print("="*80 + "\n")
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Compute confidence intervals
    results = compute_bootstrap_ci(model, X_test, y_test, n_bootstrap=1000)
    
    # Save results
    save_results(results)
    
    # Provide interpretation
    create_interpretation()
    
    print("="*80)
    print("STEP 13B COMPLETE".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
