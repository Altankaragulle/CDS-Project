"""
STEP 9: Target Variable Creation

Create binary target variable for CDS spread widening prediction:
    9.1 Compute future CDS spread (12 months ahead)
    9.2 Create distress flag based on widening thresholds

Target Definition:
    distress_flag = 1 if:
        - Absolute widening >= 50 bps OR
        - Relative widening >= 25%
    
    This aligns with credit-risk literature and CDS jump-risk conventions.

Outputs:
    - CSV: output/step9_target.csv
    - Console: Target distribution analysis
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


def load_ml_ready_data():
    """Load ML-ready dataset."""
    print("Loading ML-ready dataset...")
    
    df = pd.read_csv(OUTPUT_DIR / 'ml_ready_dataset.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  ✓ Loaded: {len(df):,} rows, {df['gvkey'].nunique()} firms")
    print(f"  ✓ Observations with CDS: {df['cds_spread'].notna().sum():,}\n")
    
    return df


def compute_future_cds(df):
    """
    Compute future CDS spread (12 months = 4 quarters ahead).
    
    For each firm-quarter observation, calculate the CDS spread
    4 quarters in the future.
    
    Returns:
        DataFrame with cds_spread_lead_4q column
    """
    print_section("9.1: COMPUTING FUTURE CDS SPREAD (+12 MONTHS)")
    
    # Sort by firm and date to ensure correct ordering
    df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    print("Creating 4-quarter lead for CDS spread...")
    
    # Shift CDS spread -4 quarters (look ahead)
    df['cds_spread_lead_4q'] = df.groupby('gvkey')['cds_spread'].shift(-4)
    
    # Calculate changes
    df['future_cds_change_abs'] = df['cds_spread_lead_4q'] - df['cds_spread']
    
    # Calculate percentage change (handle division by zero)
    df['future_cds_change_pct'] = np.where(
        df['cds_spread'] > 0,
        df['future_cds_change_abs'] / df['cds_spread'],
        np.nan
    )
    
    # Remove infinities
    df['future_cds_change_pct'] = df['future_cds_change_pct'].replace([np.inf, -np.inf], np.nan)
    
    print(f"✓ Created lead variables")
    print(f"  Current CDS available: {df['cds_spread'].notna().sum():,}")
    print(f"  Future CDS available: {df['cds_spread_lead_4q'].notna().sum():,}")
    print(f"  Lost observations (end of series): {df['cds_spread'].notna().sum() - df['cds_spread_lead_4q'].notna().sum():,}")
    
    return df


def create_distress_flag(df):
    """
    Create binary distress flag based on CDS spread widening.
    
    Distress = 1 if:
        - Absolute widening >= 50 bps OR
        - Relative widening >= 25%
    
    Returns:
        DataFrame with distress_flag column
    """
    print_section("9.2: CREATING DISTRESS FLAG")
    
    print("Applying distress thresholds:")
    print("  Absolute threshold: >= 50 bps")
    print("  Relative threshold: >= 25%")
    print()
    
    # Create distress flag
    df['distress_flag'] = np.where(
        (df['future_cds_change_abs'] >= 50) | (df['future_cds_change_pct'] >= 0.25),
        1,
        0
    )
    
    # Set to NaN where we don't have future CDS data
    df.loc[df['cds_spread_lead_4q'].isna(), 'distress_flag'] = np.nan
    
    # Count distress events
    total_labeled = df['distress_flag'].notna().sum()
    distress_count = (df['distress_flag'] == 1).sum()
    non_distress_count = (df['distress_flag'] == 0).sum()
    
    print("Distress Flag Statistics:")
    print(f"  Total labeled observations: {total_labeled:,}")
    print(f"  Distress events (1): {distress_count:,} ({distress_count/total_labeled*100:.1f}%)")
    print(f"  Non-distress events (0): {non_distress_count:,} ({non_distress_count/total_labeled*100:.1f}%)")
    print(f"  Missing labels (NaN): {df['distress_flag'].isna().sum():,}")
    
    # Breakdown by threshold
    abs_only = ((df['future_cds_change_abs'] >= 50) & (df['future_cds_change_pct'] < 0.25)).sum()
    pct_only = ((df['future_cds_change_abs'] < 50) & (df['future_cds_change_pct'] >= 0.25)).sum()
    both = ((df['future_cds_change_abs'] >= 50) & (df['future_cds_change_pct'] >= 0.25)).sum()
    
    print()
    print("Threshold Breakdown:")
    print(f"  Triggered by absolute only (>=50 bps): {abs_only:,}")
    print(f"  Triggered by relative only (>=25%): {pct_only:,}")
    print(f"  Triggered by both thresholds: {both:,}")
    
    return df


def analyze_target_distribution(df):
    """
    Analyze target variable distribution.
    """
    print_section("TARGET DISTRIBUTION ANALYSIS")
    
    # Filter to labeled observations
    labeled_df = df[df['distress_flag'].notna()].copy()
    
    print("Overall Statistics:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Labeled observations: {len(labeled_df):,}")
    print(f"  Unique firms: {labeled_df['gvkey'].nunique()}")
    print(f"  Date range: {labeled_df['date'].min().strftime('%Y-%m')} to {labeled_df['date'].max().strftime('%Y-%m')}")
    
    # Distress rate by year
    print()
    print("Distress Rate by Year:")
    yearly_distress = labeled_df.groupby(labeled_df['date'].dt.year)['distress_flag'].agg(['sum', 'count', 'mean'])
    yearly_distress.columns = ['Distress_Events', 'Total_Obs', 'Distress_Rate']
    yearly_distress['Distress_Rate'] = yearly_distress['Distress_Rate'] * 100
    
    for year, row in yearly_distress.iterrows():
        print(f"  {year}: {row['Distress_Events']:4.0f} / {row['Total_Obs']:5.0f} = {row['Distress_Rate']:5.1f}%")
    
    # CDS change statistics
    print()
    print("CDS Change Statistics (for labeled observations):")
    print(f"  Mean absolute change: {labeled_df['future_cds_change_abs'].mean():.2f} bps")
    print(f"  Median absolute change: {labeled_df['future_cds_change_abs'].median():.2f} bps")
    print(f"  Mean relative change: {labeled_df['future_cds_change_pct'].mean()*100:.2f}%")
    print(f"  Median relative change: {labeled_df['future_cds_change_pct'].median()*100:.2f}%")
    
    # Extreme changes
    print()
    print("Extreme Changes:")
    print(f"  Max absolute widening: {labeled_df['future_cds_change_abs'].max():.2f} bps")
    print(f"  Max absolute tightening: {labeled_df['future_cds_change_abs'].min():.2f} bps")
    print(f"  Max relative widening: {labeled_df['future_cds_change_pct'].max()*100:.2f}%")
    
    # Class balance check
    print()
    print("Class Balance Assessment:")
    distress_rate = labeled_df['distress_flag'].mean()
    if distress_rate < 0.1:
        print(f"  ⚠️  Severe imbalance: {distress_rate*100:.1f}% distress rate")
        print("     → Will need SMOTE or class weights in training")
    elif distress_rate < 0.3:
        print(f"  ✓ Moderate imbalance: {distress_rate*100:.1f}% distress rate")
        print("     → Manageable with class weights")
    else:
        print(f"  ✓ Good balance: {distress_rate*100:.1f}% distress rate")


def save_target_data(df):
    """
    Save dataset with target variable.
    """
    print_section("SAVING TARGET DATA")
    
    # Select columns for output
    output_cols = [
        'gvkey', 'company_name', 'date', 'year', 'quarter',
        'cds_spread', 'cds_spread_lead_4q',
        'future_cds_change_abs', 'future_cds_change_pct',
        'distress_flag'
    ]
    
    # Keep only columns that exist
    output_cols = [col for col in output_cols if col in df.columns]
    
    # Save target summary
    target_summary = df[output_cols].copy()
    output_file = OUTPUT_DIR / 'step9_target.csv'
    target_summary.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    print(f"  Rows: {len(target_summary):,}")
    print(f"  Columns: {len(output_cols)}")
    
    # Also save full dataset with target
    full_output = OUTPUT_DIR / 'ml_dataset_with_target.csv'
    df.to_csv(full_output, index=False)
    
    print(f"✓ Saved full dataset: {full_output}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")


def main():
    """
    Main execution: Create target variable.
    """
    print("\n" + "="*80)
    print("STEP 9: TARGET VARIABLE CREATION".center(80))
    print("="*80)
    
    # Load
    df = load_ml_ready_data()
    
    # Compute future CDS
    df = compute_future_cds(df)
    
    # Create distress flag
    df = create_distress_flag(df)
    
    # Analyze distribution
    analyze_target_distribution(df)
    
    # Save
    save_target_data(df)
    
    print("\n" + "="*80)
    print("✅ STEP 9 COMPLETE".center(80))
    print("="*80)
    print("\n✓ Target variable created successfully")
    print("✓ Distress flag based on 50 bps OR 25% widening")
    print("✓ Target distribution analyzed")
    print("✓ Ready for ML data construction")
    print("\nNext: Run step10_ml_construction.py\n")
    
    return df


if __name__ == "__main__":
    df = main()
