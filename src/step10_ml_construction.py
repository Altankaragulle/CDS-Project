"""
STEP 10: ML Data Construction

Prepare final datasets for machine learning:
    10.1 Create train/test split (temporal split)
    10.2 Prepare X (features) and y (target) matrices

Strategy:
    - Temporal split: Train on 2010-2020, Test on 2021-2023
    - Remove observations with missing target
    - Standardize feature names
    - Save train/test sets separately

Outputs:
    - CSV: output/train_data.csv
    - CSV: output/test_data.csv
    - Console: Dataset statistics
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


def load_data_with_target():
    """Load dataset with target variable."""
    print("Loading dataset with target variable...")
    
    df = pd.read_csv(OUTPUT_DIR / 'ml_dataset_with_target.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  ✓ Unique firms: {df['gvkey'].nunique()}")
    print(f"  ✓ Observations with target: {df['distress_flag'].notna().sum():,}\n")
    
    return df


def create_train_test_split(df):
    """
    Create temporal train/test split.
    
    Strategy:
        - Train: 2010-2020 (11 years)
        - Test: 2021-2023 (3 years)
    
    This ensures:
        - No data leakage (test is strictly future)
        - Realistic evaluation (predict future distress)
        - Sufficient training data
    
    Returns:
        train_df, test_df
    """
    print_section("10.1: CREATING TRAIN/TEST SPLIT (TEMPORAL)")
    
    # Remove observations without target
    df_labeled = df[df['distress_flag'].notna()].copy()
    
    print(f"Total labeled observations: {len(df_labeled):,}")
    print(f"Date range: {df_labeled['date'].min().strftime('%Y-%m')} to {df_labeled['date'].max().strftime('%Y-%m')}")
    
    # Define split date
    split_date = pd.Timestamp('2021-01-01')
    
    # Create train/test split
    train_df = df_labeled[df_labeled['date'] < split_date].copy()
    test_df = df_labeled[df_labeled['date'] >= split_date].copy()
    
    print()
    print("Temporal Split:")
    print(f"  Split date: {split_date.strftime('%Y-%m-%d')}")
    print(f"  Train: {train_df['date'].min().strftime('%Y-%m')} to {train_df['date'].max().strftime('%Y-%m')}")
    print(f"  Test:  {test_df['date'].min().strftime('%Y-%m')} to {test_df['date'].max().strftime('%Y-%m')}")
    
    print()
    print("Dataset Sizes:")
    print(f"  Train: {len(train_df):,} observations ({len(train_df)/len(df_labeled)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} observations ({len(test_df)/len(df_labeled)*100:.1f}%)")
    
    # Check class distribution
    train_distress_rate = train_df['distress_flag'].mean()
    test_distress_rate = test_df['distress_flag'].mean()
    
    print()
    print("Class Distribution:")
    print(f"  Train distress rate: {train_distress_rate*100:.1f}%")
    print(f"  Test distress rate:  {test_distress_rate*100:.1f}%")
    
    # Check firm overlap
    train_firms = set(train_df['gvkey'].unique())
    test_firms = set(test_df['gvkey'].unique())
    overlap_firms = train_firms & test_firms
    
    print()
    print("Firm Distribution:")
    print(f"  Train firms: {len(train_firms)}")
    print(f"  Test firms:  {len(test_firms)}")
    print(f"  Overlap:     {len(overlap_firms)} ({len(overlap_firms)/len(test_firms)*100:.1f}% of test)")
    
    print()
    print("✓ Train/test split created")
    
    return train_df, test_df


def prepare_feature_matrices(train_df, test_df):
    """
    Prepare X (features) and y (target) matrices.
    
    Separates:
        - Identifiers (gvkey, company_name, date, year, quarter)
        - Features (all numeric predictors)
        - Target (distress_flag)
        - Metadata (CDS values, changes)
    
    Returns:
        train_df, test_df with organized columns
    """
    print_section("10.2: PREPARING FEATURE MATRICES")
    
    # Define column groups
    id_cols = ['gvkey', 'company_name', 'date', 'year', 'quarter']
    
    target_col = 'distress_flag'
    
    # Metadata columns (not used for training but useful for analysis)
    # CRITICAL: Exclude future CDS values to prevent data leakage!
    metadata_cols = [
        'cds_spread', 'cds_spread_lead_4q',  # Current and future CDS (12-month)
        'future_cds_change_abs', 'future_cds_change_pct'  # Future changes
    ]
    
    # Feature columns (everything else that's numeric)
    exclude_cols = id_cols + [target_col] + metadata_cols
    feature_cols = [col for col in train_df.columns 
                   if col not in exclude_cols 
                   and train_df[col].dtype in ['float64', 'int64', 'Int64']]
    
    print(f"Column organization:")
    print(f"  Identifiers: {len(id_cols)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: 1")
    print(f"  Metadata: {len(metadata_cols)}")
    
    # Check for missing values in features
    print()
    print("Missing value check in features:")
    train_missing = train_df[feature_cols].isnull().sum().sum()
    test_missing = test_df[feature_cols].isnull().sum().sum()
    
    print(f"  Train: {train_missing:,} missing values")
    print(f"  Test:  {test_missing:,} missing values")
    
    if train_missing > 0 or test_missing > 0:
        print()
        print("  ⚠️  Features with missing values:")
        for col in feature_cols:
            train_na = train_df[col].isnull().sum()
            test_na = test_df[col].isnull().sum()
            if train_na > 0 or test_na > 0:
                print(f"    {col:30s}: Train {train_na:5,}, Test {test_na:5,}")
        print()
        print("  → These will be handled during model training (imputation)")
    
    # Print feature list
    print()
    print(f"Feature list ({len(feature_cols)} features):")
    for i, feature in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feature}")
    
    print()
    print("✓ Feature matrices prepared")
    
    # Save feature list
    feature_list = pd.DataFrame({'feature': feature_cols})
    feature_list.to_csv(OUTPUT_DIR / 'ml_feature_list.csv', index=False)
    print(f"✓ Saved feature list: {OUTPUT_DIR / 'ml_feature_list.csv'}")
    
    return train_df, test_df, feature_cols


def validate_datasets(train_df, test_df, feature_cols):
    """
    Final validation of train/test datasets.
    """
    print_section("VALIDATION")
    
    print("Train Dataset:")
    print(f"  Observations: {len(train_df):,}")
    print(f"  Firms: {train_df['gvkey'].nunique()}")
    print(f"  Date range: {train_df['date'].min().strftime('%Y-%m')} to {train_df['date'].max().strftime('%Y-%m')}")
    print(f"  Distress events: {(train_df['distress_flag']==1).sum():,} ({train_df['distress_flag'].mean()*100:.1f}%)")
    print(f"  Features: {len(feature_cols)}")
    
    print()
    print("Test Dataset:")
    print(f"  Observations: {len(test_df):,}")
    print(f"  Firms: {test_df['gvkey'].nunique()}")
    print(f"  Date range: {test_df['date'].min().strftime('%Y-%m')} to {test_df['date'].max().strftime('%Y-%m')}")
    print(f"  Distress events: {(test_df['distress_flag']==1).sum():,} ({test_df['distress_flag'].mean()*100:.1f}%)")
    print(f"  Features: {len(feature_cols)}")
    
    # Feature statistics
    print()
    print("Feature Statistics (Train):")
    print(f"  Mean non-null per observation: {train_df[feature_cols].notna().sum(axis=1).mean():.1f} / {len(feature_cols)}")
    print(f"  Complete cases (all features): {train_df[feature_cols].notna().all(axis=1).sum():,} ({train_df[feature_cols].notna().all(axis=1).sum()/len(train_df)*100:.1f}%)")
    
    print()
    print("✓ Validation complete")
    print("✓ Datasets ready for model training")


def save_datasets(train_df, test_df):
    """
    Save train and test datasets.
    """
    print_section("SAVING DATASETS")
    
    # Save train
    train_file = OUTPUT_DIR / 'train_data.csv'
    train_df.to_csv(train_file, index=False)
    print(f"✓ Saved: {train_file}")
    print(f"  Rows: {len(train_df):,}")
    print(f"  Columns: {len(train_df.columns)}")
    
    # Save test
    test_file = OUTPUT_DIR / 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    print(f"✓ Saved: {test_file}")
    print(f"  Rows: {len(test_df):,}")
    print(f"  Columns: {len(test_df.columns)}")


def main():
    """
    Main execution: Construct ML datasets.
    """
    print("\n" + "="*80)
    print("STEP 10: ML DATA CONSTRUCTION".center(80))
    print("="*80)
    
    # Load
    df = load_data_with_target()
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df)
    
    # Prepare feature matrices
    train_df, test_df, feature_cols = prepare_feature_matrices(train_df, test_df)
    
    # Validate
    validate_datasets(train_df, test_df, feature_cols)
    
    # Save
    save_datasets(train_df, test_df)
    
    print("\n" + "="*80)
    print("✅ STEP 10 COMPLETE".center(80))
    print("="*80)
    print("\n✓ Train/test split created (temporal)")
    print("✓ Feature matrices prepared")
    print("✓ Datasets validated and saved")
    print("✓ Ready for model training")
    print("\nNext: Run step11_model_training.py\n")
    
    return train_df, test_df, feature_cols


if __name__ == "__main__":
    train_df, test_df, feature_cols = main()
