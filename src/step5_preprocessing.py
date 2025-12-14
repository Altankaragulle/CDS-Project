"""
STEP 5: Preprocessing

Prepare data for feature engineering:
    5.1 Forward-fill accounting data (quarterly → all periods)
    5.2 Winsorize continuous variables (handle outliers)
    5.3 Handle remaining missing values
    5.4 Create lagged variables

Outputs:
    - CSV: output/preprocessed_dataset.csv
    - Console: Preprocessing report
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


def load_merged_data():
    """Load merged dataset."""
    print("Loading merged dataset...")
    
    df = pd.read_csv(OUTPUT_DIR / 'merged_dataset.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  ✓ Loaded: {len(df):,} rows, {df['gvkey'].nunique()} firms\n")
    return df


def forward_fill_accounting(df):
    """
    Forward-fill accounting variables within each firm.
    
    Accounting data is quarterly, so we fill forward to ensure
    each quarter has the most recent accounting values.
    
    Returns:
        DataFrame with forward-filled accounting data
    """
    print_section("5.1: FORWARD-FILLING ACCOUNTING DATA")
    
    # Accounting variables to forward-fill
    accounting_vars = ['atq', 'ltq', 'niq', 'saleq', 'cheq', 'dlttq', 'dlcq', 
                       'actq', 'lctq', 'seqq', 'cshoq', 'prccq']
    
    # Sort by firm and date
    df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    # Forward-fill within each firm
    print("Forward-filling accounting variables within each firm...")
    for var in accounting_vars:
        if var in df.columns:
            before_na = df[var].isna().sum()
            df[var] = df.groupby('gvkey')[var].ffill()
            after_na = df[var].isna().sum()
            filled = before_na - after_na
            if filled > 0:
                print(f"  {var.upper():10s}: filled {filled:,} values ({before_na:,} → {after_na:,})")
    
    print()
    print("✓ Forward-fill complete")
    
    return df


def winsorize_variables(df):
    """
    Winsorize continuous variables at 1st/99th percentile.
    
    This handles extreme outliers that could distort the model.
    
    Returns:
        DataFrame with winsorized variables
    """
    print_section("5.2: WINSORIZING CONTINUOUS VARIABLES")
    
    # Variables to winsorize
    winsorize_vars = ['atq', 'ltq', 'saleq', 'cheq', 'dlttq', 'dlcq', 
                      'actq', 'lctq', 'seqq', 'price', 'return_1m', 'cds_spread']
    
    print("Winsorizing at 1st/99th percentile...")
    for var in winsorize_vars:
        if var in df.columns and df[var].notna().sum() > 0:
            p1 = df[var].quantile(0.01)
            p99 = df[var].quantile(0.99)
            
            outliers_low = (df[var] < p1).sum()
            outliers_high = (df[var] > p99).sum()
            total_outliers = outliers_low + outliers_high
            
            if total_outliers > 0:
                df[var] = df[var].clip(lower=p1, upper=p99)
                print(f"  {var.upper():15s}: {total_outliers:,} outliers winsorized ({outliers_low:,} low, {outliers_high:,} high)")
    
    print()
    print("✓ Winsorization complete")
    
    return df


def handle_missing_values(df):
    """
    Handle remaining missing values.
    
    Strategy:
        - Drop rows with missing key variables (atq, ltq, price)
        - Fill other missing with median (within firm)
    
    Returns:
        DataFrame with handled missing values
    """
    print_section("5.3: HANDLING MISSING VALUES")
    
    initial_rows = len(df)
    
    # Key variables that must be present
    required_vars = ['atq', 'ltq', 'price']
    
    print("Dropping rows with missing key variables...")
    for var in required_vars:
        if var in df.columns:
            missing = df[var].isna().sum()
            if missing > 0:
                df = df[df[var].notna()].copy()
                print(f"  {var.upper():10s}: dropped {missing:,} rows")
    
    rows_dropped = initial_rows - len(df)
    print(f"\nTotal rows dropped: {rows_dropped:,} ({rows_dropped/initial_rows*100:.1f}%)")
    
    # Fill remaining missing with firm median
    print("\nFilling remaining missing values with firm median...")
    fill_vars = ['niq', 'saleq', 'cheq', 'dlttq', 'dlcq', 'seqq', 'return_1m']
    
    for var in fill_vars:
        if var in df.columns:
            missing_before = df[var].isna().sum()
            if missing_before > 0:
                # Fill with firm median, then overall median if still missing
                df[var] = df.groupby('gvkey')[var].transform(lambda x: x.fillna(x.median()))
                df[var] = df[var].fillna(df[var].median())
                missing_after = df[var].isna().sum()
                filled = missing_before - missing_after
                if filled > 0:
                    print(f"  {var.upper():10s}: filled {filled:,} values")
    
    print()
    print(f"Final dataset: {len(df):,} rows")
    print("✓ Missing value handling complete")
    
    return df


def create_lagged_variables(df):
    """
    Create lagged variables for time-series modeling.
    
    Creates:
        - Lagged CDS spread (1, 2, 4 quarters)
        - Lagged returns (1, 2, 4 quarters)
        - Change in key variables
    
    Returns:
        DataFrame with lagged variables
    """
    print_section("5.4: CREATING LAGGED VARIABLES")
    
    # Sort by firm and date
    df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    print("Creating lagged CDS spreads...")
    for lag in [1, 2, 4]:
        df[f'cds_spread_lag{lag}'] = df.groupby('gvkey')['cds_spread'].shift(lag)
        non_null = df[f'cds_spread_lag{lag}'].notna().sum()
        print(f"  cds_spread_lag{lag}: {non_null:,} non-null values")
    
    print("\nCreating lagged returns...")
    for lag in [1, 2, 4]:
        df[f'return_lag{lag}'] = df.groupby('gvkey')['return_1m'].shift(lag)
        non_null = df[f'return_lag{lag}'].notna().sum()
        print(f"  return_lag{lag}: {non_null:,} non-null values")
    
    print("\nCreating change variables...")
    # Change in total assets (growth)
    df['atq_change'] = df.groupby('gvkey')['atq'].pct_change()
    # Change in sales (revenue growth)
    df['saleq_change'] = df.groupby('gvkey')['saleq'].pct_change()
    # Change in CDS spread
    df['cds_change'] = df.groupby('gvkey')['cds_spread'].diff()
    
    print(f"  atq_change: {df['atq_change'].notna().sum():,} non-null values")
    print(f"  saleq_change: {df['saleq_change'].notna().sum():,} non-null values")
    print(f"  cds_change: {df['cds_change'].notna().sum():,} non-null values")
    
    print()
    print("✓ Lagged variables created")
    
    return df


def validate_preprocessed_data(df):
    """
    Validate preprocessed dataset.
    """
    print_section("VALIDATION")
    
    print("Dataset Structure:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Unique firms: {df['gvkey'].nunique()}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"  Total columns: {len(df.columns)}")
    
    print()
    print("Missing Value Summary:")
    
    key_vars = ['atq', 'ltq', 'niq', 'saleq', 'price', 'return_1m', 'cds_spread']
    for var in key_vars:
        if var in df.columns:
            missing = df[var].isna().sum()
            pct = missing / len(df) * 100
            print(f"  {var:15s}: {missing:6,} missing ({pct:5.1f}%)")
    
    print()
    print("Lagged Variables Coverage:")
    lag_vars = ['cds_spread_lag1', 'return_lag1', 'atq_change', 'saleq_change']
    for var in lag_vars:
        if var in df.columns:
            non_null = df[var].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {var:20s}: {non_null:6,} non-null ({pct:5.1f}%)")
    
    print()
    print("✓ Validation complete")


def save_preprocessed_data(df):
    """
    Save preprocessed dataset.
    """
    print_section("SAVING PREPROCESSED DATA")
    
    output_file = OUTPUT_DIR / 'preprocessed_dataset.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")


def main():
    """
    Main execution: Run all preprocessing steps.
    """
    print("\n" + "="*80)
    print("STEP 5: PREPROCESSING".center(80))
    print("="*80)
    
    # Load
    df = load_merged_data()
    
    # Forward-fill accounting
    df = forward_fill_accounting(df)
    
    # Winsorize
    df = winsorize_variables(df)
    
    # Handle missing
    df = handle_missing_values(df)
    
    # Create lags
    df = create_lagged_variables(df)
    
    # Validate
    validate_preprocessed_data(df)
    
    # Save
    save_preprocessed_data(df)
    
    print("\n" + "="*80)
    print("✅ STEP 5 COMPLETE".center(80))
    print("="*80)
    print("\n✓ Data preprocessed successfully")
    print("✓ Accounting data forward-filled")
    print("✓ Outliers winsorized")
    print("✓ Missing values handled")
    print("✓ Lagged variables created")
    print("✓ Ready for feature engineering")
    print("\nNext: Run step6_accounting_features.py\n")
    
    return df


if __name__ == "__main__":
    df = main()
