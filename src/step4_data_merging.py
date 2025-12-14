"""
STEP 4: Data Merging

Merge all datasets into one unified dataset:
    4.1 Merge Compustat + CRSP on GVKEY
    4.2 Add CDS using GVKEY-IQT mapping
    4.3 Align to quarterly frequency
    4.4 Validate merged dataset

Outputs:
    - CSV: output/merged_dataset.csv
    - Console: Merge report
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


def load_cleaned_data():
    """Load cleaned datasets."""
    print("Loading cleaned datasets...")
    
    compustat = pd.read_csv(OUTPUT_DIR / 'compustat_cleaned.csv', low_memory=False)
    compustat['datadate'] = pd.to_datetime(compustat['datadate'])
    
    crsp = pd.read_csv(OUTPUT_DIR / 'crsp_cleaned.csv', low_memory=False)
    crsp['datadate'] = pd.to_datetime(crsp['datadate'])
    
    cds = pd.read_csv(OUTPUT_DIR / 'cds_cleaned.csv', low_memory=False)
    cds['date'] = pd.to_datetime(cds['date'])
    
    mapping = pd.read_csv(DATA_DIR / 'GVKEY IQT Name matching .csv', 
                         header=None, names=['gvkey', 'iqt', 'company_name', 'extra'])
    
    print(f"  ✓ Compustat: {len(compustat):,} rows")
    print(f"  ✓ CRSP: {len(crsp):,} rows")
    print(f"  ✓ CDS: {len(cds):,} rows")
    print(f"  ✓ Mapping: {len(mapping):,} firms\n")
    
    return compustat, crsp, cds, mapping


def merge_compustat_crsp(compustat, crsp):
    """
    Merge Compustat (quarterly) with CRSP (monthly).
    
    Strategy:
        - Aggregate CRSP to quarterly (last month of quarter)
        - Merge on gvkey + quarter
    
    Returns:
        Merged DataFrame
    """
    print_section("4.1: MERGING COMPUSTAT + CRSP")
    
    # Normalize gvkey to integers
    compustat['gvkey'] = pd.to_numeric(compustat['gvkey'], errors='coerce').astype('Int64')
    crsp['gvkey'] = pd.to_numeric(crsp['gvkey'], errors='coerce').astype('Int64')
    
    # Create quarter identifier for Compustat
    compustat['year'] = compustat['datadate'].dt.year
    compustat['quarter'] = compustat['datadate'].dt.quarter
    
    print(f"Compustat: {len(compustat):,} quarterly observations")
    
    # Aggregate CRSP to quarterly (use last month of quarter)
    crsp['year'] = crsp['datadate'].dt.year
    crsp['quarter'] = crsp['datadate'].dt.quarter
    
    # For each firm-quarter, take the last available month
    crsp_quarterly = crsp.sort_values(['gvkey', 'datadate']).groupby(['gvkey', 'year', 'quarter']).last().reset_index()
    
    # Select key CRSP variables
    crsp_vars = ['gvkey', 'year', 'quarter', 'prccm', 'trt1m', 'cshom']
    crsp_quarterly = crsp_quarterly[crsp_vars].copy()
    
    # Rename to avoid confusion
    crsp_quarterly = crsp_quarterly.rename(columns={
        'prccm': 'price',
        'trt1m': 'return_1m',
        'cshom': 'shares_out'
    })
    
    print(f"CRSP aggregated to quarterly: {len(crsp_quarterly):,} observations")
    
    # Merge
    merged = compustat.merge(crsp_quarterly, on=['gvkey', 'year', 'quarter'], how='inner')
    
    print(f"\nMerge Result:")
    print(f"  Compustat firms: {compustat['gvkey'].nunique()}")
    print(f"  CRSP firms: {crsp_quarterly['gvkey'].nunique()}")
    print(f"  Merged firms: {merged['gvkey'].nunique()}")
    print(f"  Merged observations: {len(merged):,}")
    print(f"  Coverage: {len(merged)/len(compustat)*100:.1f}% of Compustat rows")
    
    return merged


def add_cds_data(merged, cds, mapping):
    """
    Add CDS data using GVKEY-IQT mapping.
    
    Strategy:
        - Map IQT to GVKEY using mapping file
        - Add company names
        - Aggregate CDS to quarterly (average of quarter)
        - Merge on gvkey + quarter
    
    Returns:
        Merged DataFrame with CDS and company names
    """
    print_section("4.2: ADDING CDS DATA & COMPANY NAMES")
    
    # Prepare mapping
    mapping['gvkey'] = pd.to_numeric(mapping['gvkey'], errors='coerce').astype('Int64')
    mapping['iqt'] = mapping['iqt'].astype(str)
    mapping_dict = dict(zip(mapping['iqt'], mapping['gvkey']))
    
    # Create company name mapping (gvkey -> company_name)
    company_name_dict = dict(zip(mapping['gvkey'], mapping['company_name']))
    
    # Add company names to merged dataset
    merged['company_name'] = merged['gvkey'].map(company_name_dict)
    print(f"Added company names: {merged['company_name'].notna().sum():,} / {len(merged):,} observations")
    
    print(f"Mapping: {len(mapping_dict)} IQT → GVKEY mappings")
    
    # Map IQT to GVKEY in CDS
    cds['iqt'] = cds['iqt'].astype(str)
    cds['gvkey'] = cds['iqt'].map(mapping_dict)
    
    # Remove unmapped
    cds_mapped = cds[cds['gvkey'].notna()].copy()
    print(f"CDS mapped: {len(cds_mapped):,} / {len(cds):,} observations ({len(cds_mapped)/len(cds)*100:.1f}%)")
    
    # Create quarter identifier
    cds_mapped['year'] = cds_mapped['date'].dt.year
    cds_mapped['quarter'] = cds_mapped['date'].dt.quarter
    
    # Aggregate to quarterly (mean spread for the quarter)
    cds_quarterly = cds_mapped.groupby(['gvkey', 'year', 'quarter']).agg({
        'cds_spread': 'mean'
    }).reset_index()
    
    print(f"CDS aggregated to quarterly: {len(cds_quarterly):,} observations")
    
    # Merge with main dataset
    merged_with_cds = merged.merge(cds_quarterly, on=['gvkey', 'year', 'quarter'], how='left')
    
    print(f"\nMerge Result:")
    print(f"  Before CDS: {len(merged):,} observations")
    print(f"  After CDS: {len(merged_with_cds):,} observations")
    print(f"  Observations with CDS: {merged_with_cds['cds_spread'].notna().sum():,} ({merged_with_cds['cds_spread'].notna().sum()/len(merged_with_cds)*100:.1f}%)")
    print(f"  Firms with CDS: {merged_with_cds[merged_with_cds['cds_spread'].notna()]['gvkey'].nunique()}")
    
    return merged_with_cds


def align_to_quarterly(merged):
    """
    Ensure quarterly frequency and create date column.
    
    Returns:
        DataFrame with proper quarterly date
    """
    print_section("4.3: ALIGNING TO QUARTERLY FREQUENCY")
    
    # Create proper quarter-end date
    merged['date'] = pd.to_datetime(merged['datadate'])
    
    # Sort by firm and date
    merged = merged.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    print(f"Date range: {merged['date'].min().strftime('%Y-%m')} to {merged['date'].max().strftime('%Y-%m')}")
    print(f"Total quarters: {len(merged):,}")
    print(f"Firms: {merged['gvkey'].nunique()}")
    
    # Calculate observations per firm
    obs_per_firm = merged.groupby('gvkey').size()
    print(f"\nObservations per firm:")
    print(f"  Mean: {obs_per_firm.mean():.1f}")
    print(f"  Median: {obs_per_firm.median():.0f}")
    print(f"  Min: {obs_per_firm.min()}")
    print(f"  Max: {obs_per_firm.max()}")
    
    return merged


def validate_merged_data(merged):
    """
    Validate the merged dataset.
    """
    print_section("4.4: VALIDATION")
    
    print("Dataset Structure:")
    print(f"  Total observations: {len(merged):,}")
    print(f"  Unique firms: {merged['gvkey'].nunique()}")
    print(f"  Date range: {merged['date'].min().strftime('%Y-%m')} to {merged['date'].max().strftime('%Y-%m')}")
    print(f"  Years covered: {merged['date'].dt.year.nunique()}")
    
    print()
    print("Variable Coverage:")
    
    # Key variables
    key_vars = {
        'atq': 'Total Assets',
        'ltq': 'Total Liabilities',
        'niq': 'Net Income',
        'saleq': 'Sales',
        'seqq': 'Stockholders Equity',
        'price': 'Stock Price',
        'return_1m': 'Monthly Return',
        'cds_spread': 'CDS Spread'
    }
    
    for var, name in key_vars.items():
        if var in merged.columns:
            non_null = merged[var].notna().sum()
            pct = non_null / len(merged) * 100
            print(f"  {name:25s}: {non_null:6,} ({pct:5.1f}%)")
    
    print()
    print("Data Quality Checks:")
    
    # Check for complete cases (all key variables present)
    required_vars = ['atq', 'ltq', 'niq', 'saleq', 'price', 'cds_spread']
    complete_cases = merged[required_vars].notna().all(axis=1).sum()
    print(f"  Complete cases (all key vars): {complete_cases:,} ({complete_cases/len(merged)*100:.1f}%)")
    
    # Check for firms with CDS data
    firms_with_cds = merged[merged['cds_spread'].notna()]['gvkey'].nunique()
    print(f"  Firms with CDS data: {firms_with_cds}")
    
    print()
    print("✓ Validation complete")


def save_merged_data(merged):
    """
    Save merged dataset.
    """
    print_section("SAVING MERGED DATA")
    
    # Select key columns for final dataset (company_name first for readability)
    key_cols = ['gvkey', 'company_name', 'date', 'year', 'quarter', 
                'atq', 'ltq', 'niq', 'saleq', 'cheq', 'dlttq', 'dlcq', 
                'actq', 'lctq', 'seqq', 'cshoq', 'prccq',
                'price', 'return_1m', 'shares_out', 'cds_spread']
    
    # Keep only columns that exist
    final_cols = [col for col in key_cols if col in merged.columns]
    merged_final = merged[final_cols].copy()
    
    # Save
    output_file = OUTPUT_DIR / 'merged_dataset.csv'
    merged_final.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    print(f"  Rows: {len(merged_final):,}")
    print(f"  Columns: {len(merged_final.columns)}")


def main():
    """
    Main execution: Run all merging steps.
    """
    print("\n" + "="*80)
    print("STEP 4: DATA MERGING".center(80))
    print("="*80)
    
    # Load
    compustat, crsp, cds, mapping = load_cleaned_data()
    
    # Merge Compustat + CRSP
    merged = merge_compustat_crsp(compustat, crsp)
    
    # Add CDS
    merged = add_cds_data(merged, cds, mapping)
    
    # Align to quarterly
    merged = align_to_quarterly(merged)
    
    # Validate
    validate_merged_data(merged)
    
    # Save
    save_merged_data(merged)
    
    print("\n" + "="*80)
    print("✅ STEP 4 COMPLETE".center(80))
    print("="*80)
    print("\n✓ All datasets merged successfully")
    print("✓ Quarterly frequency aligned")
    print("✓ Ready for preprocessing and feature engineering")
    print("\nNext: Run step5_preprocessing.py\n")
    
    return merged


if __name__ == "__main__":
    merged = main()
