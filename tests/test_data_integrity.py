"""
Unit Tests for Data Integrity and Leakage Prevention

Tests to ensure:
1. No future information leaked into features
2. Temporal train/test split is correct
3. No target leakage
4. Reasonable data splits
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'


class TestDataLeakage:
    """Test suite for data leakage prevention."""
    
    def test_no_future_data_in_features(self):
        """Verify no future information leaked into features."""
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        feature_names = features['feature'].tolist()
        
        # Check for forbidden terms that indicate future data
        forbidden_terms = ['lead', 'future', '_ahead', '_forward', 'next_']
        
        for term in forbidden_terms:
            violating_features = [f for f in feature_names if term in f.lower()]
            assert len(violating_features) == 0, \
                f"Found future information in features: {violating_features}"
        
        print(f"✓ Verified {len(feature_names)} features contain no future data")
    
    def test_no_target_in_features(self):
        """Verify target variable is not in feature list."""
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        feature_names = features['feature'].tolist()
        
        # Target and related variables should NOT be features
        forbidden_features = [
            'distress_flag',
            'cds_spread_lead_4q',
            'future_cds_change_abs',
            'future_cds_change_pct'
        ]
        
        for forbidden in forbidden_features:
            assert forbidden not in feature_names, \
                f"Target-related variable '{forbidden}' found in features!"
        
        print(f"✓ Verified target variables excluded from features")
    
    def test_lagged_features_use_past_data(self):
        """Verify lagged features look backward, not forward."""
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        feature_names = features['feature'].tolist()
        
        # Find lag features
        lag_features = [f for f in feature_names if 'lag' in f.lower()]
        
        # All lags should be positive (looking backward)
        # e.g., cds_spread_lag1, return_lag4
        for lag_feature in lag_features:
            # Extract lag number
            if 'lag' in lag_feature:
                parts = lag_feature.split('lag')
                if len(parts) > 1 and parts[1].isdigit():
                    lag_num = int(parts[1])
                    assert lag_num > 0, \
                        f"Feature '{lag_feature}' has non-positive lag: {lag_num}"
        
        print(f"✓ Verified {len(lag_features)} lagged features use past data")


class TestTemporalIntegrity:
    """Test suite for temporal train/test split."""
    
    def test_temporal_split_no_overlap(self):
        """Verify train/test temporal separation."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        
        train['date'] = pd.to_datetime(train['date'])
        test['date'] = pd.to_datetime(test['date'])
        
        # Train must be strictly before test
        train_max = train['date'].max()
        test_min = test['date'].min()
        
        assert train_max < test_min, \
            f"Train data overlaps with test! Train max: {train_max}, Test min: {test_min}"
        
        print(f"✓ Train period: {train['date'].min()} to {train_max}")
        print(f"✓ Test period: {test_min} to {test['date'].max()}")
        print(f"✓ No temporal overlap")
    
    def test_split_date_is_2021(self):
        """Verify split happens at 2021-01-01."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        
        train['date'] = pd.to_datetime(train['date'])
        test['date'] = pd.to_datetime(test['date'])
        
        split_date = pd.Timestamp('2021-01-01')
        
        # All train should be before 2021
        assert (train['date'] < split_date).all(), \
            "Some train observations are after 2021-01-01"
        
        # All test should be 2021 or after
        assert (test['date'] >= split_date).all(), \
            "Some test observations are before 2021-01-01"
        
        print(f"✓ Split date verified: {split_date}")
    
    def test_train_test_size_reasonable(self):
        """Verify reasonable train/test split proportions."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        
        total = len(train) + len(test)
        train_pct = len(train) / total
        test_pct = len(test) / total
        
        # Train should be 70-85% of data (typical for temporal split)
        assert 0.70 <= train_pct <= 0.85, \
            f"Train proportion {train_pct:.1%} outside acceptable range [70%, 85%]"
        
        print(f"✓ Train: {len(train):,} ({train_pct:.1%})")
        print(f"✓ Test: {len(test):,} ({test_pct:.1%})")


class TestDataQuality:
    """Test suite for data quality."""
    
    def test_target_is_binary(self):
        """Verify target variable is binary (0 or 1)."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        
        # Check train
        train_unique = train['distress_flag'].dropna().unique()
        assert set(train_unique).issubset({0, 1}), \
            f"Train target contains non-binary values: {train_unique}"
        
        # Check test
        test_unique = test['distress_flag'].dropna().unique()
        assert set(test_unique).issubset({0, 1}), \
            f"Test target contains non-binary values: {test_unique}"
        
        print(f"✓ Target is binary in both train and test")
    
    def test_no_infinite_values_in_features(self):
        """Verify no infinite values in features."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        
        for feature in features['feature']:
            if feature in train.columns:
                has_inf = np.isinf(train[feature]).any()
                assert not has_inf, \
                    f"Feature '{feature}' contains infinite values"
        
        print(f"✓ No infinite values in {len(features)} features")
    
    def test_features_exist_in_data(self):
        """Verify all listed features exist in train/test data."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        
        for feature in features['feature']:
            assert feature in train.columns, \
                f"Feature '{feature}' not found in train data"
            assert feature in test.columns, \
                f"Feature '{feature}' not found in test data"
        
        print(f"✓ All {len(features)} features exist in train and test")
    
    def test_no_all_null_features(self):
        """Verify no features are completely null."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        
        for feature in features['feature']:
            if feature in train.columns:
                non_null_count = train[feature].notna().sum()
                assert non_null_count > 0, \
                    f"Feature '{feature}' is completely null"
        
        print(f"✓ All features have at least some non-null values")


class TestDistressDefinition:
    """Test suite for distress flag definition."""
    
    def test_distress_rate_reasonable(self):
        """Verify distress rate is reasonable (not 0% or 100%)."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        
        train_rate = train['distress_flag'].mean()
        test_rate = test['distress_flag'].mean()
        
        # Distress rate should be between 5% and 40% (reasonable for credit risk)
        assert 0.05 <= train_rate <= 0.40, \
            f"Train distress rate {train_rate:.1%} outside reasonable range"
        assert 0.05 <= test_rate <= 0.40, \
            f"Test distress rate {test_rate:.1%} outside reasonable range"
        
        print(f"✓ Train distress rate: {train_rate:.1%}")
        print(f"✓ Test distress rate: {test_rate:.1%}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
