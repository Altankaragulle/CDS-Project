"""
Pytest Configuration and Fixtures

Provides shared fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def output_dir(project_root):
    """Return output directory."""
    return project_root / 'output'


@pytest.fixture
def models_dir(output_dir):
    """Return models directory."""
    return output_dir / 'models'


@pytest.fixture
def data_dir(project_root):
    """Return data directory."""
    return project_root / 'data'
