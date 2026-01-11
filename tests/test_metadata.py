# -*- coding: utf-8 -*-
"""
Unit tests for ModelMetadata
"""

import pytest
import json
import tempfile
import os
from datetime import datetime

from simtrademl.core.models.metadata import ModelMetadata, create_model_id


class TestModelMetadata:
    """Test ModelMetadata class"""

    def test_create_basic_metadata(self):
        """Test creating basic metadata"""
        metadata = ModelMetadata(
            model_id='test_model_001',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10', 'rsi14'],
            n_features=3
        )

        assert metadata.model_id == 'test_model_001'
        assert metadata.version == '1.0'
        assert len(metadata.features) == 3
        assert metadata.n_features == 3

    def test_feature_count_validation(self):
        """Test that n_features must match features list length"""
        with pytest.raises(ValueError, match="n_features.*does not match"):
            ModelMetadata(
                model_id='test',
                version='1.0',
                created_at='2026-01-11T10:00:00',
                model_type='xgboost',
                model_library_version='0.90',
                features=['ma5', 'ma10'],
                n_features=3  # Wrong!
            )

    def test_empty_features_validation(self):
        """Test that features cannot be empty"""
        with pytest.raises(ValueError, match="Feature list cannot be empty"):
            ModelMetadata(
                model_id='test',
                version='1.0',
                created_at='2026-01-11T10:00:00',
                model_type='xgboost',
                model_library_version='0.90',
                features=[],
                n_features=0
            )

    def test_duplicate_features_validation(self):
        """Test that duplicate features are detected"""
        with pytest.raises(ValueError, match="Duplicate features"):
            ModelMetadata(
                model_id='test',
                version='1.0',
                created_at='2026-01-11T10:00:00',
                model_type='xgboost',
                model_library_version='0.90',
                features=['ma5', 'ma10', 'ma5'],  # Duplicate!
                n_features=3
            )

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10'],
            n_features=2
        )

        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data['model_id'] == 'test'
        assert data['features'] == ['ma5', 'ma10']

    def test_to_json(self):
        """Test conversion to JSON"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10'],
            n_features=2
        )

        json_str = metadata.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data['model_id'] == 'test'

    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            'model_id': 'test',
            'version': '1.0',
            'created_at': '2026-01-11T10:00:00',
            'model_type': 'xgboost',
            'model_library_version': '0.90',
            'features': ['ma5', 'ma10'],
            'n_features': 2
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.model_id == 'test'
        assert metadata.features == ['ma5', 'ma10']

    def test_from_json(self):
        """Test creating from JSON string"""
        json_str = '''
        {
            "model_id": "test",
            "version": "1.0",
            "created_at": "2026-01-11T10:00:00",
            "model_type": "xgboost",
            "model_library_version": "0.90",
            "features": ["ma5", "ma10"],
            "n_features": 2
        }
        '''

        metadata = ModelMetadata.from_json(json_str)

        assert metadata.model_id == 'test'
        assert metadata.features == ['ma5', 'ma10']

    def test_save_and_load(self):
        """Test saving and loading from file"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10', 'rsi14'],
            n_features=3
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            metadata.save(filepath)

            # Load back
            loaded = ModelMetadata.load(filepath)

            assert loaded.model_id == metadata.model_id
            assert loaded.features == metadata.features
            assert loaded.n_features == metadata.n_features
        finally:
            os.unlink(filepath)

    def test_validate_features_success(self):
        """Test feature validation with matching features"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10', 'rsi14'],
            n_features=3
        )

        # Should not raise
        assert metadata.validate_features(['ma5', 'ma10', 'rsi14'])

    def test_validate_features_wrong_order(self):
        """Test feature validation with wrong order"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10', 'rsi14'],
            n_features=3
        )

        with pytest.raises(ValueError, match="Feature mismatch"):
            metadata.validate_features(['ma10', 'ma5', 'rsi14'])  # Wrong order!

    def test_validate_features_missing(self):
        """Test feature validation with missing features"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10', 'rsi14'],
            n_features=3
        )

        with pytest.raises(ValueError, match="Missing features"):
            metadata.validate_features(['ma5', 'ma10'])  # Missing rsi14

    def test_validate_features_extra(self):
        """Test feature validation with extra features"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10'],
            n_features=2
        )

        with pytest.raises(ValueError, match="Extra features"):
            metadata.validate_features(['ma5', 'ma10', 'rsi14'])  # Extra rsi14

    def test_add_metric(self):
        """Test adding metrics"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5'],
            n_features=1
        )

        metadata.add_metric('ic', 0.05)
        metadata.add_metric('icir', 1.2)

        assert metadata.metrics['ic'] == 0.05
        assert metadata.metrics['icir'] == 1.2

    def test_add_file(self):
        """Test adding file references"""
        metadata = ModelMetadata(
            model_id='test',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5'],
            n_features=1
        )

        metadata.add_file('model', 'model.json')
        metadata.add_file('scaler', 'scaler.pkl')

        assert metadata.files['model'] == 'model.json'
        assert metadata.files['scaler'] == 'scaler.pkl'

    def test_summary(self):
        """Test summary generation"""
        metadata = ModelMetadata(
            model_id='test_model',
            version='1.0',
            created_at='2026-01-11T10:00:00',
            model_type='xgboost',
            model_library_version='0.90',
            features=['ma5', 'ma10', 'rsi14'],
            n_features=3,
            scaler_type='RobustScaler',
            n_samples=1000
        )

        metadata.add_metric('ic', 0.05)
        metadata.add_file('model', 'model.json')

        summary = metadata.summary()

        assert 'test_model' in summary
        assert 'RobustScaler' in summary
        assert '1,000' in summary  # Formatted sample count
        assert 'ic: 0.0500' in summary


class TestCreateModelId:
    """Test create_model_id function"""

    def test_default_prefix(self):
        """Test model ID creation with default prefix"""
        model_id = create_model_id()

        assert model_id.startswith('model_')
        assert len(model_id.split('_')) == 3  # model_YYYYMMDD_HHMMSS

    def test_custom_prefix(self):
        """Test model ID creation with custom prefix"""
        model_id = create_model_id(prefix='xgb')

        assert model_id.startswith('xgb_')

    def test_uniqueness(self):
        """Test that generated IDs are unique"""
        import time

        id1 = create_model_id()
        time.sleep(1.1)  # Wait to ensure different timestamp
        id2 = create_model_id()

        assert id1 != id2
