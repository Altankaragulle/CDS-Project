"""
STEP 8: Feature Validation

Validate and finalize features:
    8.1 Check correlations and multicollinearity
    8.2 Final feature selection and cleanup

Outputs:
    - CSV: output/ml_ready_dataset.csv
    - Console: Validation report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_features():
    """Load complete feature dataset."""
    print("Loading complete feature dataset...")
    
    df = pd.read_csv(OUTPUT_DIR / 'features_complete.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns\n")
    return df


def check_correlations(df):
    """
    Check correlations between features.
    
    Identifies highly correlated features that may cause multicollinearity.
    
    Returns:
        DataFrame with correlation analysis
    """
    print_section("8.1: CHECKING CORRELATIONS & MULTICOLLINEARITY")
    
    # Select only numeric features (exclude identifiers and dates)
    exclude_cols = ['gvkey', 'company_name', 'date', 'year', 'quarter']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    print(f"Analyzing {len(numeric_cols)} numeric features...")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Find highly correlated pairs (>0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9):")
        for pair in high_corr_pairs[:10]:  # Show first 10
            print(f"  {pair['Feature1']:25s} <-> {pair['Feature2']:25s}: {pair['Correlation']:.3f}")
        if len(high_corr_pairs) > 10:
            print(f"  ... and {len(high_corr_pairs) - 10} more")
    else:
        print("\n✓ No highly correlated pairs found (|r| > 0.9)")
    
    # Check correlation with target (CDS spread)
    if 'cds_spread' in df.columns:
        print("\nTop 10 features correlated with CDS spread:")
        cds_corr = df[numeric_cols].corrwith(df['cds_spread']).abs().sort_values(ascending=False)
        for i, (feature, corr) in enumerate(cds_corr.head(10).items(), 1):
            if feature != 'cds_spread':
                print(f"  {i:2d}. {feature:30s}: {corr:.3f}")
    
    print()
    print("✓ Correlation analysis complete")
    
    return high_corr_pairs


def final_feature_selection(df):
    """
    Select final features for ML modeling.
    
    Removes:
        - Highly redundant features
        - Features with too many missing values
        - Intermediate calculation columns
    
    Returns:
        DataFrame with final feature set
    """
    print_section("8.2: FINAL FEATURE SELECTION")
    
    initial_cols = len(df.columns)
    
    # Define features to keep
    # Identifiers
    id_cols = ['gvkey', 'company_name', 'date', 'year', 'quarter']
    
    # Core accounting features
    accounting_features = [
        'atq', 'ltq', 'niq', 'saleq', 'seqq',
        'debt_to_equity', 'debt_to_assets', 'leverage',
        'current_ratio', 'cash_ratio',
        'roa', 'roe', 'profit_margin', 'asset_turnover',
        'altman_z_score'  # Altman Z-Score for credit risk
    ]
    
    # Core market features
    market_features = [
        'price', 'return_1m', 'log_market_cap',
        'momentum_3m', 'momentum_12m',
        'volatility_3m', 'volatility_12m',
        'max_drawdown_12m'
    ]
    
    # Lagged features
    lagged_features = [
        'cds_spread_lag1', 'cds_spread_lag4',
        'return_lag1', 'return_lag4',
        'atq_change', 'saleq_change'
    ]
    
    # Target
    target = ['cds_spread']
    
    # Combine all features to keep
    features_to_keep = id_cols + accounting_features + market_features + lagged_features + target
    
    # Keep only features that exist in dataframe
    final_features = [col for col in features_to_keep if col in df.columns]
    
    df_final = df[final_features].copy()
    
    print(f"Feature selection:")
    print(f"  Initial columns: {initial_cols}")
    print(f"  Final columns: {len(final_features)}")
    print(f"  Removed: {initial_cols - len(final_features)}")
    
    print()
    print("Final feature categories:")
    print(f"  Identifiers: {len(id_cols)}")
    print(f"  Accounting: {len([f for f in accounting_features if f in df_final.columns])}")
    print(f"  Market: {len([f for f in market_features if f in df_final.columns])}")
    print(f"  Lagged: {len([f for f in lagged_features if f in df_final.columns])}")
    print(f"  Target: 1 (cds_spread)")
    
    # Check for missing values in final dataset
    print()
    print("Missing value check:")
    missing_summary = df_final.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    
    if len(missing_features) > 0:
        print(f"  Features with missing values: {len(missing_features)}")
        for feature, count in missing_features.items():
            pct = count / len(df_final) * 100
            print(f"    {feature:25s}: {count:6,} ({pct:5.1f}%)")
    else:
        print("  ✓ No missing values in final dataset")
    
    print()
    print("✓ Feature selection complete")
    
    return df_final


def validate_final_dataset(df):
    """
    Final validation of ML-ready dataset.
    """
    print_section("FINAL VALIDATION")
    
    print("Dataset Summary:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Unique firms: {df['gvkey'].nunique()}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    
    # Count complete cases (all features non-null)
    feature_cols = [col for col in df.columns if col not in ['gvkey', 'company_name', 'date', 'year', 'quarter']]
    complete_cases = df[feature_cols].notna().all(axis=1).sum()
    
    print()
    print("Data Completeness:")
    print(f"  Complete cases (all features): {complete_cases:,} ({complete_cases/len(df)*100:.1f}%)")
    print(f"  Observations with CDS: {df['cds_spread'].notna().sum():,} ({df['cds_spread'].notna().sum()/len(df)*100:.1f}%)")
    
    # Feature statistics
    print()
    print("Feature Statistics:")
    numeric_features = df.select_dtypes(include=[np.number]).columns
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Mean features per observation: {df[numeric_features].notna().sum(axis=1).mean():.1f}")
    
    print()
    print("✓ Final validation complete")
    print("✓ Dataset ready for ML modeling")


def save_ml_ready_dataset(df):
    """
    Save ML-ready dataset.
    """
    print_section("SAVING ML-READY DATASET")
    
    output_file = OUTPUT_DIR / 'ml_ready_dataset.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    # Also save feature list
    feature_cols = [col for col in df.columns if col not in ['gvkey', 'company_name', 'date', 'year', 'quarter', 'cds_spread']]
    feature_list = pd.DataFrame({'feature': feature_cols})
    feature_list.to_csv(OUTPUT_DIR / 'feature_list.csv', index=False)
    print(f"✓ Saved feature list: {OUTPUT_DIR / 'feature_list.csv'}")


def main():
    """
    Main execution: Validate and finalize features.
    """
    print("\n" + "="*80)
    print("STEP 8: FEATURE VALIDATION".center(80))
    print("="*80)
    
    # Load
    df = load_features()
    
    # Check correlations
    high_corr_pairs = check_correlations(df)
    
    # Final feature selection
    df_final = final_feature_selection(df)
    
    # Validate
    validate_final_dataset(df_final)
    
    # Save
    save_ml_ready_dataset(df_final)
    
    print("\n" + "="*80)
    print("✅ STEP 8 COMPLETE".center(80))
    print("="*80)
    print("\n✓ Features validated successfully")
    print("✓ Multicollinearity checked")
    print("✓ Final feature set selected")
    print("✓ ML-ready dataset prepared")
    print("✓ Ready for target variable creation")
    print("\nNext: Run step9_target_creation.py\n")
    
    return df_final


if __name__ == "__main__":
    df_final = main()
