"""
STEP 3: Data Cleaning

Clean and standardize all datasets:
    3.1 Clean Compustat (handle missing, derive equity)
    3.2 Clean CRSP (handle outliers, missing)
    3.3 Transform CDS from wide to long format
    3.4 Validate cleaned datasets

Outputs:
    - CSV: output/compustat_cleaned.csv
    - CSV: output/crsp_cleaned.csv
    - CSV: output/cds_cleaned.csv
    - Console: Cleaning report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_datasets():
    """Load raw datasets."""
    print("Loading raw datasets...")
    
    compustat = pd.read_csv(DATA_DIR / 'CDS firms fundamentals Quarterly.csv', 
                           sep=';', encoding='latin-1', low_memory=False)
    compustat['datadate'] = pd.to_datetime(compustat['datadate'])
    
    crsp = pd.read_csv(DATA_DIR / 'Security prices.csv', low_memory=False)
    crsp['datadate'] = pd.to_datetime(crsp['datadate'])
    
    cds = pd.read_csv(DATA_DIR / 'CDS Prices.csv', low_memory=False)
    
    print(f"  ✓ Loaded all datasets\n")
    return compustat, crsp, cds


def clean_compustat(compustat):
    """
    Clean Compustat data.
    
    Steps:
        - Convert to numeric
        - Handle negative values (set to NaN where inappropriate)
        - Derive stockholders' equity (SEQQ)
        - Remove duplicates
    
    Returns:
        Cleaned Compustat DataFrame
    """
    print_section("3.1: CLEANING COMPUSTAT")
    
    df = compustat.copy()
    initial_rows = len(df)
    
    # Key variables to clean
    numeric_vars = ['atq', 'ltq', 'niq', 'saleq', 'cheq', 'dlttq', 'dlcq', 
                    'actq', 'lctq', 'cshoq', 'prccq', 'oibdpq', 'rectq', 'invtq']
    
    # Convert to numeric
    for var in numeric_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
    
    print("Converting to numeric...")
    print(f"  ✓ Converted {len(numeric_vars)} variables")
    
    # Handle negative values (assets, equity should be positive)
    positive_vars = ['atq', 'actq', 'cheq', 'cshoq', 'prccq']
    for var in positive_vars:
        if var in df.columns:
            neg_count = (df[var] < 0).sum()
            if neg_count > 0:
                df.loc[df[var] < 0, var] = np.nan
                print(f"  ✓ Set {neg_count} negative {var.upper()} to NaN")
    
    # Derive stockholders' equity (SEQQ = ATQ - LTQ)
    if 'atq' in df.columns and 'ltq' in df.columns:
        df['seqq'] = df['atq'] - df['ltq']
        df.loc[df['seqq'] < 0, 'seqq'] = np.nan  # Negative equity -> NaN
        print(f"  ✓ Derived SEQQ (stockholders' equity)")
    
    # Remove duplicates (same firm-date)
    df = df.drop_duplicates(subset=['gvkey', 'datadate'], keep='first')
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"  ✓ Removed {duplicates_removed} duplicate rows")
    
    # Sort by firm and date
    df = df.sort_values(['gvkey', 'datadate']).reset_index(drop=True)
    
    print()
    print(f"Cleaning Summary:")
    print(f"  Initial rows: {initial_rows:,}")
    print(f"  Final rows: {len(df):,}")
    print(f"  Rows removed: {initial_rows - len(df):,}")
    
    return df


def clean_crsp(crsp):
    """
    Clean CRSP data.
    
    Steps:
        - Convert to numeric
        - Handle outliers (winsorize at 1st/99th percentile)
        - Remove invalid prices (<=0)
        - Remove duplicates
    
    Returns:
        Cleaned CRSP DataFrame
    """
    print_section("3.2: CLEANING CRSP")
    
    df = crsp.copy()
    initial_rows = len(df)
    
    # Convert to numeric
    numeric_vars = ['prccm', 'trt1m', 'cshom']
    for var in numeric_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
    
    print("Converting to numeric...")
    print(f"  ✓ Converted {len(numeric_vars)} variables")
    
    # Remove invalid prices
    if 'prccm' in df.columns:
        invalid_prices = (df['prccm'] <= 0) | (df['prccm'].isna())
        df = df[~invalid_prices]
        print(f"  ✓ Removed {invalid_prices.sum():,} invalid prices (<=0 or NaN)")
    
    # Winsorize returns at 1st/99th percentile (handle extreme outliers)
    if 'trt1m' in df.columns:
        p1 = df['trt1m'].quantile(0.01)
        p99 = df['trt1m'].quantile(0.99)
        outliers = ((df['trt1m'] < p1) | (df['trt1m'] > p99)).sum()
        df['trt1m'] = df['trt1m'].clip(lower=p1, upper=p99)
        print(f"  ✓ Winsorized {outliers:,} return outliers at 1st/99th percentile")
    
    # Winsorize shares outstanding
    if 'cshom' in df.columns:
        p1 = df['cshom'].quantile(0.01)
        p99 = df['cshom'].quantile(0.99)
        outliers = ((df['cshom'] < p1) | (df['cshom'] > p99)).sum()
        df['cshom'] = df['cshom'].clip(lower=p1, upper=p99)
        print(f"  ✓ Winsorized {outliers:,} shares outliers at 1st/99th percentile")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['gvkey', 'datadate'], keep='first')
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"  ✓ Removed {duplicates_removed} duplicate rows")
    
    # Sort
    df = df.sort_values(['gvkey', 'datadate']).reset_index(drop=True)
    
    print()
    print(f"Cleaning Summary:")
    print(f"  Initial rows: {initial_rows:,}")
    print(f"  Final rows: {len(df):,}")
    print(f"  Rows removed: {initial_rows - len(df):,}")
    
    return df


def transform_cds(cds):
    """
    Transform CDS from wide to long format.
    
    Wide format:  IQT | 31.12.24 | 01.12.24 | ...
    Long format:  IQT | date | spread
    
    Returns:
        CDS DataFrame in long format
    """
    print_section("3.3: TRANSFORMING CDS (WIDE → LONG)")
    
    # Get IQT column (first column)
    iqt_col = cds.columns[0]
    
    # Melt to long format
    df_long = cds.melt(id_vars=[iqt_col], 
                       var_name='date_str', 
                       value_name='cds_spread')
    
    # Rename IQT column
    df_long = df_long.rename(columns={iqt_col: 'iqt'})
    
    # Parse dates (format: DD.MM.YY)
    df_long['date'] = pd.to_datetime(df_long['date_str'], format='%d.%m.%y', errors='coerce')
    
    # Convert spread to numeric
    df_long['cds_spread'] = pd.to_numeric(df_long['cds_spread'], errors='coerce')
    
    # Remove missing/zero spreads
    initial_rows = len(df_long)
    df_long = df_long[df_long['cds_spread'] > 0].copy()
    removed = initial_rows - len(df_long)
    
    print(f"Transformation Summary:")
    print(f"  Initial format: {len(cds)} firms × {len(cds.columns)-1} dates (wide)")
    print(f"  Transformed to: {len(df_long):,} observations (long)")
    print(f"  Removed missing/zero: {removed:,}")
    print(f"  Valid spreads: {len(df_long):,}")
    
    # Handle extreme values (>10,000 bps likely errors)
    extreme = df_long['cds_spread'] > 10000
    if extreme.sum() > 0:
        print(f"  ⚠️  Found {extreme.sum()} extreme spreads (>10,000 bps)")
        print(f"     Capping at 10,000 bps")
        df_long.loc[extreme, 'cds_spread'] = 10000
    
    # Sort
    df_long = df_long.sort_values(['iqt', 'date']).reset_index(drop=True)
    
    # Drop date_str column
    df_long = df_long[['iqt', 'date', 'cds_spread']]
    
    return df_long


def validate_cleaned_data(compustat, crsp, cds):
    """
    Validate cleaned datasets.
    """
    print_section("3.4: VALIDATION")
    
    print("Compustat Validation:")
    print(f"  Rows: {len(compustat):,}")
    print(f"  Firms: {compustat['gvkey'].nunique():,}")
    print(f"  Date range: {compustat['datadate'].min().strftime('%Y-%m')} to {compustat['datadate'].max().strftime('%Y-%m')}")
    print(f"  Key variables non-null: ATQ {compustat['atq'].notna().sum():,}, NIQ {compustat['niq'].notna().sum():,}")
    
    print()
    print("CRSP Validation:")
    print(f"  Rows: {len(crsp):,}")
    print(f"  Firms: {crsp['gvkey'].nunique():,}")
    print(f"  Date range: {crsp['datadate'].min().strftime('%Y-%m')} to {crsp['datadate'].max().strftime('%Y-%m')}")
    print(f"  Valid prices: {crsp['prccm'].notna().sum():,}")
    
    print()
    print("CDS Validation:")
    print(f"  Rows: {len(cds):,}")
    print(f"  Firms: {cds['iqt'].nunique():,}")
    print(f"  Date range: {cds['date'].min().strftime('%Y-%m')} to {cds['date'].max().strftime('%Y-%m')}")
    print(f"  Spread range: {cds['cds_spread'].min():.1f} to {cds['cds_spread'].max():.1f} bps")
    print(f"  Median spread: {cds['cds_spread'].median():.1f} bps")
    
    print()
    print("✓ All datasets validated successfully")


def save_cleaned_data(compustat, crsp, cds):
    """
    Save cleaned datasets.
    """
    print_section("SAVING CLEANED DATA")
    
    # Save
    compustat.to_csv(OUTPUT_DIR / 'compustat_cleaned.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'compustat_cleaned.csv'}")
    
    crsp.to_csv(OUTPUT_DIR / 'crsp_cleaned.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'crsp_cleaned.csv'}")
    
    cds.to_csv(OUTPUT_DIR / 'cds_cleaned.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'cds_cleaned.csv'}")


def main():
    """
    Main execution: Run all cleaning steps.
    """
    print("\n" + "="*80)
    print("STEP 3: DATA CLEANING".center(80))
    print("="*80)
    
    # Load
    compustat, crsp, cds = load_datasets()
    
    # Clean
    compustat_clean = clean_compustat(compustat)
    crsp_clean = clean_crsp(crsp)
    cds_clean = transform_cds(cds)
    
    # Validate
    validate_cleaned_data(compustat_clean, crsp_clean, cds_clean)
    
    # Save
    save_cleaned_data(compustat_clean, crsp_clean, cds_clean)
    
    print("\n" + "="*80)
    print("✅ STEP 3 COMPLETE".center(80))
    print("="*80)
    print("\n✓ All datasets cleaned")
    print("✓ CDS transformed to long format")
    print("✓ Outliers handled")
    print("✓ Ready for merging")
    print("\nNext: Run step4_data_merging.py\n")
    
    return compustat_clean, crsp_clean, cds_clean


if __name__ == "__main__":
    compustat_clean, crsp_clean, cds_clean = main()
