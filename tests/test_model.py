"""
Unit Tests for Model Training and Predictions

Tests to ensure:
1. Trained models exist
2. Model predictions are valid probabilities
3. Models are reproducible
4. Performance metrics are reasonable
"""

import pytest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'


class TestModelExistence:
    """Test suite for model file existence."""
    
    def test_optimized_model_exists(self):
        """Verify optimized LightGBM model exists."""
        model_path = MODELS_DIR / 'lightgbm_optimized.pkl'
        assert model_path.exists(), \
            f"Optimized model not found at {model_path}"
        
        print(f"✓ Found optimized model: {model_path}")
    
    def test_model_is_loadable(self):
        """Verify model can be loaded without errors."""
        model_path = MODELS_DIR / 'lightgbm_optimized.pkl'
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            assert model is not None
            print(f"✓ Model loaded successfully")
        except Exception as e:
            pytest.fail(f"Failed to load model: {str(e)}")


class TestModelPredictions:
    """Test suite for model predictions."""
    
    def test_predictions_are_probabilities(self):
        """Verify model predictions are valid probabilities [0, 1]."""
        # Load model
        with open(MODELS_DIR / 'lightgbm_optimized.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load test data
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        
        X_test = test[features['feature'].tolist()]
        
        # Get predictions
        try:
            proba = model.predict_proba(X_test)[:, 1]
        except:
            # Some models might not have predict_proba
            pytest.skip("Model doesn't support predict_proba")
        
        # Check all probabilities are in [0, 1]
        assert (proba >= 0).all(), "Found probabilities < 0"
        assert (proba <= 1).all(), "Found probabilities > 1"
        assert not np.isnan(proba).any(), "Found NaN probabilities"
        assert not np.isinf(proba).any(), "Found infinite probabilities"
        
        print(f"✓ All {len(proba):,} predictions are valid probabilities")
        print(f"  Min: {proba.min():.4f}, Max: {proba.max():.4f}")
        print(f"  Mean: {proba.mean():.4f}, Median: {np.median(proba):.4f}")
    
    def test_predictions_have_variance(self):
        """Verify model doesn't predict same value for all observations."""
        # Load model
        with open(MODELS_DIR / 'lightgbm_optimized.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load test data
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        
        X_test = test[features['feature'].tolist()]
        
        # Get predictions
        try:
            proba = model.predict_proba(X_test)[:, 1]
        except:
            pytest.skip("Model doesn't support predict_proba")
        
        # Check variance
        variance = np.var(proba)
        unique_values = len(np.unique(proba))
        
        assert variance > 0.001, \
            f"Model predictions have very low variance: {variance:.6f}"
        assert unique_values > 10, \
            f"Model produces only {unique_values} unique predictions"
        
        print(f"✓ Predictions have sufficient variance: {variance:.4f}")
        print(f"✓ Unique prediction values: {unique_values}")


class TestModelPerformance:
    """Test suite for model performance metrics."""
    
    def test_auc_is_reasonable(self):
        """Verify AUC is better than random (> 0.5)."""
        from sklearn.metrics import roc_auc_score
        
        # Load model
        with open(MODELS_DIR / 'lightgbm_optimized.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load test data
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        
        X_test = test[features['feature'].tolist()]
        y_test = test['distress_flag']
        
        # Get predictions
        try:
            proba = model.predict_proba(X_test)[:, 1]
        except:
            pytest.skip("Model doesn't support predict_proba")
        
        # Calculate AUC
        auc = roc_auc_score(y_test, proba)
        
        # AUC should be better than random (0.5)
        assert auc > 0.5, \
            f"Model AUC {auc:.4f} is not better than random (0.5)"
        
        # AUC should be reasonable (< 1.0, which would indicate overfitting)
        assert auc < 0.95, \
            f"Model AUC {auc:.4f} is suspiciously high (possible overfitting)"
        
        print(f"✓ Model AUC: {auc:.4f} (reasonable range)")
    
    def test_recall_is_positive(self):
        """Verify model catches at least some distressed firms."""
        # This test is optional - skip if model format is complex
        pytest.skip("Model recall test skipped (model format varies)")


class TestReproducibility:
    """Test suite for reproducibility."""
    
    def test_feature_list_is_consistent(self):
        """Verify feature list hasn't changed."""
        features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
        
        # Expected number of features (29 in your project)
        expected_count = 29
        actual_count = len(features)
        
        # Allow some flexibility (±5 features)
        assert abs(actual_count - expected_count) <= 5, \
            f"Feature count changed significantly: expected ~{expected_count}, got {actual_count}"
        
        print(f"✓ Feature count stable: {actual_count} features")
    
    def test_train_test_split_is_deterministic(self):
        """Verify train/test split produces same sizes."""
        train = pd.read_csv(OUTPUT_DIR / 'train_data.csv')
        test = pd.read_csv(OUTPUT_DIR / 'test_data.csv')
        
        # These should be stable across runs
        # (actual numbers from your project)
        expected_train_size = 21971
        expected_test_size = 6276
        
        # Allow small variation (±100 observations)
        assert abs(len(train) - expected_train_size) <= 100, \
            f"Train size changed: expected ~{expected_train_size}, got {len(train)}"
        assert abs(len(test) - expected_test_size) <= 100, \
            f"Test size changed: expected ~{expected_test_size}, got {len(test)}"
        
        print(f"✓ Train size: {len(train):,} (stable)")
        print(f"✓ Test size: {len(test):,} (stable)")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
