"""
STEP 7: Market Features

Create market-based features:
    7.1 Size and momentum features (market cap, returns)
    7.2 Volatility features (return volatility)
    7.3 Drawdown features (max drawdown)

Outputs:
    - CSV: output/features_complete.csv
    - Console: Feature creation report
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


def load_accounting_features():
    """Load dataset with accounting features."""
    print("Loading dataset with accounting features...")
    
    df = pd.read_csv(OUTPUT_DIR / 'features_accounting.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  ✓ Loaded: {len(df):,} rows, {df['gvkey'].nunique()} firms\n")
    return df


def create_size_momentum_features(df):
    """
    Create size and momentum features.
    
    Features:
        - market_cap: Market capitalization (price * shares)
        - log_market_cap: Log of market cap
        - momentum_3m: 3-month return
        - momentum_6m: 6-month return
        - momentum_12m: 12-month return
    
    Returns:
        DataFrame with size and momentum features
    """
    print_section("7.1: CREATING SIZE & MOMENTUM FEATURES")
    
    # Sort by firm and date
    df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    # Market capitalization (price * shares outstanding)
    df['market_cap'] = df['price'] * df['shares_out']
    
    # Log market cap (for better distribution)
    df['log_market_cap'] = np.log(df['market_cap'] + 1)
    
    print("Created size features:")
    print(f"  market_cap:         {df['market_cap'].notna().sum():,} non-null")
    print(f"  log_market_cap:     {df['log_market_cap'].notna().sum():,} non-null")
    
    # Momentum features (cumulative returns over different periods)
    # 3-month momentum (3 quarters back)
    df['momentum_3m'] = df.groupby('gvkey')['return_1m'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    
    # 6-month momentum (6 quarters back)
    df['momentum_6m'] = df.groupby('gvkey')['return_1m'].transform(
        lambda x: x.rolling(window=6, min_periods=1).sum()
    )
    
    # 12-month momentum (12 quarters back)
    df['momentum_12m'] = df.groupby('gvkey')['return_1m'].transform(
        lambda x: x.rolling(window=12, min_periods=1).sum()
    )
    
    print("\nCreated momentum features:")
    print(f"  momentum_3m:        {df['momentum_3m'].notna().sum():,} non-null")
    print(f"  momentum_6m:        {df['momentum_6m'].notna().sum():,} non-null")
    print(f"  momentum_12m:       {df['momentum_12m'].notna().sum():,} non-null")
    
    print()
    print("Summary statistics:")
    print(f"  log_market_cap:     mean={df['log_market_cap'].mean():.2f}, median={df['log_market_cap'].median():.2f}")
    print(f"  momentum_3m:        mean={df['momentum_3m'].mean():.4f}, median={df['momentum_3m'].median():.4f}")
    print(f"  momentum_12m:       mean={df['momentum_12m'].mean():.4f}, median={df['momentum_12m'].median():.4f}")
    
    return df


def create_volatility_features(df):
    """
    Create volatility features.
    
    Features:
        - volatility_3m: 3-month return volatility
        - volatility_6m: 6-month return volatility
        - volatility_12m: 12-month return volatility
    
    Returns:
        DataFrame with volatility features
    """
    print_section("7.2: CREATING VOLATILITY FEATURES")
    
    # Sort by firm and date
    df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    # 3-month volatility
    df['volatility_3m'] = df.groupby('gvkey')['return_1m'].transform(
        lambda x: x.rolling(window=3, min_periods=2).std()
    )
    
    # 6-month volatility
    df['volatility_6m'] = df.groupby('gvkey')['return_1m'].transform(
        lambda x: x.rolling(window=6, min_periods=3).std()
    )
    
    # 12-month volatility
    df['volatility_12m'] = df.groupby('gvkey')['return_1m'].transform(
        lambda x: x.rolling(window=12, min_periods=6).std()
    )
    
    print("Created volatility features:")
    print(f"  volatility_3m:      {df['volatility_3m'].notna().sum():,} non-null")
    print(f"  volatility_6m:      {df['volatility_6m'].notna().sum():,} non-null")
    print(f"  volatility_12m:     {df['volatility_12m'].notna().sum():,} non-null")
    
    print()
    print("Summary statistics:")
    print(f"  volatility_3m:      mean={df['volatility_3m'].mean():.4f}, median={df['volatility_3m'].median():.4f}")
    print(f"  volatility_6m:      mean={df['volatility_6m'].mean():.4f}, median={df['volatility_6m'].median():.4f}")
    print(f"  volatility_12m:     mean={df['volatility_12m'].mean():.4f}, median={df['volatility_12m'].median():.4f}")
    
    return df


def create_drawdown_features(df):
    """
    Create drawdown features.
    
    Features:
        - max_drawdown_3m: Maximum drawdown over 3 months
        - max_drawdown_6m: Maximum drawdown over 6 months
        - max_drawdown_12m: Maximum drawdown over 12 months
    
    Returns:
        DataFrame with drawdown features
    """
    print_section("7.3: CREATING DRAWDOWN FEATURES")
    
    # Sort by firm and date
    df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    # Calculate cumulative returns for drawdown calculation
    df['cum_return'] = df.groupby('gvkey')['return_1m'].transform(
        lambda x: (1 + x).cumprod()
    )
    
    # 3-month max drawdown
    df['max_drawdown_3m'] = df.groupby('gvkey')['cum_return'].transform(
        lambda x: (x / x.rolling(window=3, min_periods=1).max() - 1)
    )
    
    # 6-month max drawdown
    df['max_drawdown_6m'] = df.groupby('gvkey')['cum_return'].transform(
        lambda x: (x / x.rolling(window=6, min_periods=1).max() - 1)
    )
    
    # 12-month max drawdown
    df['max_drawdown_12m'] = df.groupby('gvkey')['cum_return'].transform(
        lambda x: (x / x.rolling(window=12, min_periods=1).max() - 1)
    )
    
    # Drop temporary column
    df = df.drop('cum_return', axis=1)
    
    print("Created drawdown features:")
    print(f"  max_drawdown_3m:    {df['max_drawdown_3m'].notna().sum():,} non-null")
    print(f"  max_drawdown_6m:    {df['max_drawdown_6m'].notna().sum():,} non-null")
    print(f"  max_drawdown_12m:   {df['max_drawdown_12m'].notna().sum():,} non-null")
    
    print()
    print("Summary statistics:")
    print(f"  max_drawdown_3m:    mean={df['max_drawdown_3m'].mean():.4f}, median={df['max_drawdown_3m'].median():.4f}")
    print(f"  max_drawdown_6m:    mean={df['max_drawdown_6m'].mean():.4f}, median={df['max_drawdown_6m'].median():.4f}")
    print(f"  max_drawdown_12m:   mean={df['max_drawdown_12m'].mean():.4f}, median={df['max_drawdown_12m'].median():.4f}")
    
    return df


def validate_features(df):
    """
    Validate created features.
    """
    print_section("VALIDATION")
    
    # List all new features
    size_momentum_features = ['market_cap', 'log_market_cap', 'momentum_3m', 'momentum_6m', 'momentum_12m']
    volatility_features = ['volatility_3m', 'volatility_6m', 'volatility_12m']
    drawdown_features = ['max_drawdown_3m', 'max_drawdown_6m', 'max_drawdown_12m']
    
    all_features = size_momentum_features + volatility_features + drawdown_features
    
    print("Feature Coverage:")
    for feature in all_features:
        if feature in df.columns:
            non_null = df[feature].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {feature:25s}: {non_null:6,} ({pct:5.1f}%)")
    
    print()
    print("Feature Statistics:")
    print(f"  Total market features created: {len(all_features)}")
    print(f"  Total observations: {len(df):,}")
    print(f"  Firms: {df['gvkey'].nunique()}")
    print(f"  Total columns: {len(df.columns)}")
    
    print()
    print("✓ Validation complete")


def save_features(df):
    """
    Save complete feature dataset.
    """
    print_section("SAVING COMPLETE FEATURES")
    
    output_file = OUTPUT_DIR / 'features_complete.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")


def main():
    """
    Main execution: Create all market features.
    """
    print("\n" + "="*80)
    print("STEP 7: MARKET FEATURES".center(80))
    print("="*80)
    
    # Load
    df = load_accounting_features()
    
    # Create features
    df = create_size_momentum_features(df)
    df = create_volatility_features(df)
    df = create_drawdown_features(df)
    
    # Validate
    validate_features(df)
    
    # Save
    save_features(df)
    
    print("\n" + "="*80)
    print("✅ STEP 7 COMPLETE".center(80))
    print("="*80)
    print("\n✓ Market features created successfully")
    print("✓ 11 market-based features generated")
    print("✓ Size, momentum, volatility, and drawdown metrics ready")
    print("✓ Complete feature set ready for validation")
    print("\nNext: Run step8_feature_validation.py\n")
    
    return df


if __name__ == "__main__":
    df = main()
