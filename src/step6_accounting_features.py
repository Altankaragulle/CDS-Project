"""
STEP 6: Accounting Features

Create financial ratios and accounting-based features:
    6.1 Leverage ratios (debt/equity, debt/assets)
    6.2 Liquidity ratios (current ratio, quick ratio, cash ratio)
    6.3 Profitability ratios (ROA, ROE, profit margin)
    6.4 Altman Z-Score (credit risk composite score)

Outputs:
    - CSV: output/features_accounting.csv
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


def load_preprocessed_data():
    """Load preprocessed dataset."""
    print("Loading preprocessed dataset...")
    
    df = pd.read_csv(OUTPUT_DIR / 'preprocessed_dataset.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  ✓ Loaded: {len(df):,} rows, {df['gvkey'].nunique()} firms\n")
    return df


def create_leverage_ratios(df):
    """
    Create leverage ratios.
    
    Ratios:
        - debt_to_equity: Total Debt / Stockholders' Equity
        - debt_to_assets: Total Debt / Total Assets
        - leverage: Total Assets / Stockholders' Equity
        - long_term_debt_ratio: Long-term Debt / Total Assets
    
    Returns:
        DataFrame with leverage ratios
    """
    print_section("6.1: CREATING LEVERAGE RATIOS")
    
    # Total debt = Long-term debt + Short-term debt
    df['total_debt'] = df['dlttq'] + df['dlcq']
    
    # Debt-to-Equity ratio
    df['debt_to_equity'] = df['total_debt'] / df['seqq']
    df['debt_to_equity'] = df['debt_to_equity'].replace([np.inf, -np.inf], np.nan)
    
    # Debt-to-Assets ratio
    df['debt_to_assets'] = df['total_debt'] / df['atq']
    
    # Leverage (Assets/Equity)
    df['leverage'] = df['atq'] / df['seqq']
    df['leverage'] = df['leverage'].replace([np.inf, -np.inf], np.nan)
    
    # Long-term debt ratio
    df['lt_debt_ratio'] = df['dlttq'] / df['atq']
    
    print("Created leverage ratios:")
    print(f"  debt_to_equity:     {df['debt_to_equity'].notna().sum():,} non-null")
    print(f"  debt_to_assets:     {df['debt_to_assets'].notna().sum():,} non-null")
    print(f"  leverage:           {df['leverage'].notna().sum():,} non-null")
    print(f"  lt_debt_ratio:      {df['lt_debt_ratio'].notna().sum():,} non-null")
    
    print()
    print("Summary statistics:")
    print(f"  debt_to_equity:     mean={df['debt_to_equity'].mean():.2f}, median={df['debt_to_equity'].median():.2f}")
    print(f"  debt_to_assets:     mean={df['debt_to_assets'].mean():.2f}, median={df['debt_to_assets'].median():.2f}")
    print(f"  leverage:           mean={df['leverage'].mean():.2f}, median={df['leverage'].median():.2f}")
    
    return df


def create_liquidity_ratios(df):
    """
    Create liquidity ratios.
    
    Ratios:
        - current_ratio: Current Assets / Current Liabilities
        - quick_ratio: (Current Assets - Inventory) / Current Liabilities
        - cash_ratio: Cash / Current Liabilities
        - working_capital_ratio: (Current Assets - Current Liabilities) / Total Assets
    
    Returns:
        DataFrame with liquidity ratios
    """
    print_section("6.2: CREATING LIQUIDITY RATIOS")
    
    # Current ratio
    df['current_ratio'] = df['actq'] / df['lctq']
    df['current_ratio'] = df['current_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Quick ratio (assuming invtq is inventory)
    # If invtq not available, use actq directly
    if 'invtq' in df.columns:
        df['quick_ratio'] = (df['actq'] - df['invtq']) / df['lctq']
    else:
        df['quick_ratio'] = df['actq'] / df['lctq']
    df['quick_ratio'] = df['quick_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Cash ratio
    df['cash_ratio'] = df['cheq'] / df['lctq']
    df['cash_ratio'] = df['cash_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Working capital ratio
    df['working_capital_ratio'] = (df['actq'] - df['lctq']) / df['atq']
    
    print("Created liquidity ratios:")
    print(f"  current_ratio:      {df['current_ratio'].notna().sum():,} non-null")
    print(f"  quick_ratio:        {df['quick_ratio'].notna().sum():,} non-null")
    print(f"  cash_ratio:         {df['cash_ratio'].notna().sum():,} non-null")
    print(f"  working_cap_ratio:  {df['working_capital_ratio'].notna().sum():,} non-null")
    
    print()
    print("Summary statistics:")
    print(f"  current_ratio:      mean={df['current_ratio'].mean():.2f}, median={df['current_ratio'].median():.2f}")
    print(f"  cash_ratio:         mean={df['cash_ratio'].mean():.2f}, median={df['cash_ratio'].median():.2f}")
    
    return df


def create_profitability_ratios(df):
    """
    Create profitability ratios.
    
    Ratios:
        - roa: Return on Assets (Net Income / Total Assets)
        - roe: Return on Equity (Net Income / Stockholders' Equity)
        - profit_margin: Net Income / Sales
        - asset_turnover: Sales / Total Assets
    
    Returns:
        DataFrame with profitability ratios
    """
    print_section("6.3: CREATING PROFITABILITY RATIOS")
    
    # ROA (Return on Assets)
    df['roa'] = df['niq'] / df['atq']
    
    # ROE (Return on Equity)
    df['roe'] = df['niq'] / df['seqq']
    df['roe'] = df['roe'].replace([np.inf, -np.inf], np.nan)
    
    # Profit margin
    df['profit_margin'] = df['niq'] / df['saleq']
    df['profit_margin'] = df['profit_margin'].replace([np.inf, -np.inf], np.nan)
    
    # Asset turnover
    df['asset_turnover'] = df['saleq'] / df['atq']
    
    print("Created profitability ratios:")
    print(f"  roa:                {df['roa'].notna().sum():,} non-null")
    print(f"  roe:                {df['roe'].notna().sum():,} non-null")
    print(f"  profit_margin:      {df['profit_margin'].notna().sum():,} non-null")
    print(f"  asset_turnover:     {df['asset_turnover'].notna().sum():,} non-null")
    
    print()
    print("Summary statistics:")
    print(f"  roa:                mean={df['roa'].mean():.4f}, median={df['roa'].median():.4f}")
    print(f"  roe:                mean={df['roe'].mean():.4f}, median={df['roe'].median():.4f}")
    print(f"  profit_margin:      mean={df['profit_margin'].mean():.4f}, median={df['profit_margin'].median():.4f}")
    
    return df


def create_altman_z_score(df):
    """
    Create Altman Z-Score for credit risk assessment.
    
    Altman Z-Score (1968) for public manufacturing firms:
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    
    Where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Book Value of Total Liabilities
        X5 = Sales / Total Assets
    
    Interpretation:
        Z > 2.99: Safe zone (low bankruptcy risk)
        1.81 < Z < 2.99: Grey zone (moderate risk)
        Z < 1.81: Distress zone (high bankruptcy risk)
    
    Returns:
        DataFrame with Altman Z-Score
    """
    print_section("6.4: CREATING ALTMAN Z-SCORE")
    
    print("Computing Altman Z-Score components...")
    
    # X1: Working Capital / Total Assets
    working_capital = df['actq'] - df['lctq']
    x1 = working_capital / df['atq']
    
    # X2: Retained Earnings / Total Assets
    # Use retained earnings (req) if available, otherwise approximate with equity - common stock
    if 'req' in df.columns:
        x2 = df['req'] / df['atq']
    else:
        # Approximate: use stockholders equity as proxy
        x2 = df['seqq'] / df['atq']
    
    # X3: EBIT / Total Assets
    # EBIT = Operating Income Before Depreciation (oibdpq) - Depreciation (dpq)
    # If not available, use Net Income + Interest Expense + Taxes
    if 'oibdpq' in df.columns:
        ebit = df['oibdpq']
    else:
        # Approximate EBIT using Net Income (niq)
        # EBIT ≈ NI + Interest + Taxes
        # For simplicity, use NI * 1.5 as rough approximation
        ebit = df['niq'] * 1.5
    
    x3 = ebit / df['atq']
    
    # X4: Market Value of Equity / Book Value of Total Liabilities
    # Market cap = price * shares outstanding (cshoq)
    # If price and shares available, compute market cap; otherwise use book value
    if 'price' in df.columns and 'cshoq' in df.columns:
        market_value_equity = df['price'] * df['cshoq']
    elif 'prccq' in df.columns and 'cshoq' in df.columns:
        market_value_equity = df['prccq'] * df['cshoq']
    else:
        # Fallback: use book value of equity
        market_value_equity = df['seqq']
    
    total_liabilities = df['ltq']  # Total liabilities from Compustat
    x4 = market_value_equity / total_liabilities
    x4 = x4.replace([np.inf, -np.inf], np.nan)
    
    # X5: Sales / Total Assets (Asset Turnover)
    x5 = df['saleq'] / df['atq']
    
    # Calculate Altman Z-Score
    df['altman_z_score'] = (
        1.2 * x1 + 
        1.4 * x2 + 
        3.3 * x3 + 
        0.6 * x4 + 
        1.0 * x5
    )
    
    # Replace infinities with NaN
    df['altman_z_score'] = df['altman_z_score'].replace([np.inf, -np.inf], np.nan)
    
    # Create categorical risk zones
    df['altman_risk_zone'] = pd.cut(
        df['altman_z_score'],
        bins=[-np.inf, 1.81, 2.99, np.inf],
        labels=['Distress', 'Grey', 'Safe']
    )
    
    print(f"✓ Altman Z-Score computed: {df['altman_z_score'].notna().sum():,} non-null values")
    print()
    
    # Summary statistics
    print("Altman Z-Score Statistics:")
    print(f"  Mean:   {df['altman_z_score'].mean():.2f}")
    print(f"  Median: {df['altman_z_score'].median():.2f}")
    print(f"  Std:    {df['altman_z_score'].std():.2f}")
    print(f"  Min:    {df['altman_z_score'].min():.2f}")
    print(f"  Max:    {df['altman_z_score'].max():.2f}")
    print()
    
    # Risk zone distribution
    print("Risk Zone Distribution:")
    if df['altman_risk_zone'].notna().sum() > 0:
        zone_counts = df['altman_risk_zone'].value_counts()
        zone_pcts = df['altman_risk_zone'].value_counts(normalize=True) * 100
        
        for zone in ['Distress', 'Grey', 'Safe']:
            if zone in zone_counts.index:
                count = zone_counts[zone]
                pct = zone_pcts[zone]
                print(f"  {zone:10s}: {count:6,} ({pct:5.1f}%)")
    
    print()
    print("Component Statistics:")
    print(f"  X1 (Working Capital/Assets): mean={x1.mean():.3f}")
    print(f"  X2 (Retained Earnings/Assets): mean={x2.mean():.3f}")
    print(f"  X3 (EBIT/Assets): mean={x3.mean():.3f}")
    print(f"  X4 (Market Cap/Liabilities): mean={x4.mean():.3f}")
    print(f"  X5 (Sales/Assets): mean={x5.mean():.3f}")
    
    return df


def validate_features(df):
    """
    Validate created features.
    """
    print_section("VALIDATION")
    
    # List all new features
    leverage_features = ['debt_to_equity', 'debt_to_assets', 'leverage', 'lt_debt_ratio']
    liquidity_features = ['current_ratio', 'quick_ratio', 'cash_ratio', 'working_capital_ratio']
    profitability_features = ['roa', 'roe', 'profit_margin', 'asset_turnover']
    credit_risk_features = ['altman_z_score']
    
    all_features = leverage_features + liquidity_features + profitability_features + credit_risk_features
    
    print("Feature Coverage:")
    for feature in all_features:
        if feature in df.columns:
            non_null = df[feature].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {feature:25s}: {non_null:6,} ({pct:5.1f}%)")
    
    print()
    print("Feature Statistics:")
    print(f"  Total features created: {len(all_features)}")
    print(f"  Total observations: {len(df):,}")
    print(f"  Firms: {df['gvkey'].nunique()}")
    
    # Check for extreme values
    print()
    print("Extreme Value Check:")
    for feature in all_features:
        if feature in df.columns:
            extreme_low = (df[feature] < -10).sum()
            extreme_high = (df[feature] > 10).sum()
            if extreme_low > 0 or extreme_high > 0:
                print(f"  {feature:25s}: {extreme_low:,} < -10, {extreme_high:,} > 10")
    
    print()
    print("✓ Validation complete")


def save_features(df):
    """
    Save dataset with accounting features.
    """
    print_section("SAVING FEATURES")
    
    output_file = OUTPUT_DIR / 'features_accounting.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")


def main():
    """
    Main execution: Create all accounting features.
    """
    print("\n" + "="*80)
    print("STEP 6: ACCOUNTING FEATURES".center(80))
    print("="*80)
    
    # Load
    df = load_preprocessed_data()
    
    # Create features
    df = create_leverage_ratios(df)
    df = create_liquidity_ratios(df)
    df = create_profitability_ratios(df)
    df = create_altman_z_score(df)
    
    # Validate
    validate_features(df)
    
    # Save
    save_features(df)
    
    print("\n" + "="*80)
    print("✅ STEP 6 COMPLETE".center(80))
    print("="*80)
    print("\n✓ Accounting features created successfully")
    print("✓ 13 financial ratios generated")
    print("✓ Leverage, liquidity, profitability, and credit risk metrics ready")
    print("✓ Altman Z-Score added for bankruptcy prediction")
    print("✓ Ready for market feature engineering")
    print("\nNext: Run step7_market_features.py\n")
    
    return df


if __name__ == "__main__":
    df = main()
