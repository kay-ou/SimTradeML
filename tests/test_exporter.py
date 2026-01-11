# -*- coding: utf-8 -*-
"""
Unit tests for PTradeModelExporter
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import RobustScaler

from simtrademl.core.models.metadata import ModelMetadata, create_model_id
from simtrademl.core.models.exporter import PTradeModelExporter


@pytest.fixture
def sample_model():
    """Create a simple XGBoost model for testing"""
    X = np.random.rand(100, 3)
    y = np.random.rand(100)

    dtrain = xgb.DMatrix(X, label=y)
    params = {'max_depth': 2, 'eta': 0.1, 'objective': 'reg:squarederror'}
    model = xgb.train(params, dtrain, num_boost_round=10)

    return model


@pytest.fixture
def sample_scaler():
    """Create a fitted scaler for testing"""
    X = np.random.rand(100, 3)
    scaler = RobustScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture
def sample_metadata():
    """Create sample metadata"""
    return ModelMetadata(
        model_id=create_model_id('test'),
        version='1.0',
        created_at='2026-01-11T10:00:00',
        model_type='xgboost',
        model_library_version='0.90',
        features=['feature1', 'feature2', 'feature3'],
        n_features=3,
        scaler_type='RobustScaler'
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


class TestPTradeModelExporter:
    """Test PTradeModelExporter class"""

    def test_export_json_format(self, sample_model, sample_scaler, sample_metadata, temp_dir):
        """Test exporting model in JSON format"""
        output_dir = temp_dir / 'test_model'
        exporter = PTradeModelExporter(str(output_dir))

        result_path = exporter.export(
            model=sample_model,
            metadata=sample_metadata,
            scaler=sample_scaler,
            model_format='json'
        )

        assert Path(result_path).exists()
        assert (output_dir / 'model.json').exists()
        assert (output_dir / 'scaler.pkl').exists()
        assert (output_dir / 'features.json').exists()
        assert (output_dir / 'metadata.json').exists()
        assert (output_dir / 'README.md').exists()
        assert (output_dir / 'usage_example.py').exists()

    def test_export_model_format(self, sample_model, sample_scaler, sample_metadata, temp_dir):
        """Test exporting model in .model format"""
        output_dir = temp_dir / 'test_model'
        exporter = PTradeModelExporter(str(output_dir))

        exporter.export(
            model=sample_model,
            metadata=sample_metadata,
            scaler=sample_scaler,
            model_format='model'
        )

        assert (output_dir / 'model.model').exists()

    def test_export_pickle_format(self, sample_model, sample_scaler, sample_metadata, temp_dir):
        """Test exporting model in pickle format"""
        output_dir = temp_dir / 'test_model'
        exporter = PTradeModelExporter(str(output_dir))

        exporter.export(
            model=sample_model,
            metadata=sample_metadata,
            scaler=sample_scaler,
            model_format='pickle'
        )

        assert (output_dir / 'model.pkl').exists()

    def test_invalid_format(self, sample_model, sample_metadata, temp_dir):
        """Test that invalid format raises error"""
        output_dir = temp_dir / 'test_model'
        exporter = PTradeModelExporter(str(output_dir))

        with pytest.raises(ValueError, match="Invalid model_format"):
            exporter.export(
                model=sample_model,
                metadata=sample_metadata,
                model_format='invalid'
            )

    def test_overwrite_protection(self, sample_model, sample_metadata, temp_dir):
        """Test that existing directory is protected"""
        output_dir = temp_dir / 'test_model'
        output_dir.mkdir()

        exporter = PTradeModelExporter(str(output_dir))

        with pytest.raises(FileExistsError, match="already exists"):
            exporter.export(
                model=sample_model,
                metadata=sample_metadata,
                overwrite=False
            )
