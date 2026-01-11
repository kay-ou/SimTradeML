# -*- coding: utf-8 -*-
"""
Unit tests for PTradeModelPackage
"""

import pytest
import numpy as np
import xgboost as xgb
from pathlib import Path
import tempfile
import shutil
from sklearn.preprocessing import RobustScaler

from simtrademl.core.models.package import PTradeModelPackage
from simtrademl.core.models.metadata import ModelMetadata, create_model_id


@pytest.fixture
def sample_model():
    """Create a simple XGBoost model for testing"""
    X_train = np.random.randn(100, 3)
    y_train = np.random.randn(100)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'max_depth': 3, 'eta': 0.1, 'objective': 'reg:squarederror'}
    model = xgb.train(params, dtrain, num_boost_round=10)

    return model


@pytest.fixture
def sample_scaler():
    """Create a fitted scaler for testing"""
    scaler = RobustScaler()
    X = np.random.randn(100, 3)
    scaler.fit(X)
    return scaler


@pytest.fixture
def sample_metadata():
    """Create sample metadata"""
    from datetime import datetime
    return ModelMetadata(
        model_id=create_model_id('test'),
        version='1.0',
        created_at=datetime.now().isoformat(),
        model_type='xgboost',
        model_library_version='0.90',
        features=['feat_a', 'feat_b', 'feat_c'],
        n_features=3,
        scaler_type='RobustScaler',
        description='Test model'
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestPTradeModelPackage:
    """Test PTradeModelPackage class"""

    def test_create_package(self, sample_model, sample_scaler, sample_metadata):
        """Test creating a package"""
        package = PTradeModelPackage(
            model=sample_model,
            scaler=sample_scaler,
            metadata=sample_metadata
        )

        assert package.model is not None
        assert package.scaler is not None
        assert package.metadata is not None

    def test_save_and_load(self, sample_model, sample_scaler, sample_metadata, temp_dir):
        """Test saving and loading package"""
        # Create and save
        package = PTradeModelPackage(sample_model, sample_scaler, sample_metadata)
        save_path = temp_dir / 'test_model.ptp'
        package.save(str(save_path))

        assert save_path.exists()

        # Load
        loaded_package = PTradeModelPackage.load(str(save_path))

        assert loaded_package.model is not None
        assert loaded_package.scaler is not None
        assert loaded_package.metadata is not None
        assert loaded_package.metadata.features == sample_metadata.features

    def test_predict_single(self, sample_model, sample_scaler, sample_metadata):
        """Test single prediction"""
        package = PTradeModelPackage(sample_model, sample_scaler, sample_metadata)

        features = {
            'feat_a': 1.0,
            'feat_b': 2.0,
            'feat_c': 3.0
        }

        prediction = package.predict(features)

        assert isinstance(prediction, float)
        assert np.isfinite(prediction)

    def test_predict_batch(self, sample_model, sample_scaler, sample_metadata):
        """Test batch prediction"""
        package = PTradeModelPackage(sample_model, sample_scaler, sample_metadata)

        features_list = [
            {'feat_a': 1.0, 'feat_b': 2.0, 'feat_c': 3.0},
            {'feat_a': 1.5, 'feat_b': 2.5, 'feat_c': 3.5},
            {'feat_a': 2.0, 'feat_b': 3.0, 'feat_c': 4.0}
        ]

        predictions = package.predict_batch(features_list)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3
        assert all(np.isfinite(predictions))

    def test_predict_batch_empty(self, sample_model, sample_scaler, sample_metadata):
        """Test batch prediction with empty list"""
        package = PTradeModelPackage(sample_model, sample_scaler, sample_metadata)

        predictions = package.predict_batch([])

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 0

    def test_predict_without_scaler(self, sample_model, sample_metadata):
        """Test prediction without scaler"""
        package = PTradeModelPackage(sample_model, scaler=None, metadata=sample_metadata)

        features = {
            'feat_a': 1.0,
            'feat_b': 2.0,
            'feat_c': 3.0
        }

        prediction = package.predict(features)

        assert isinstance(prediction, float)
        assert np.isfinite(prediction)

    def test_predict_wrong_features(self, sample_model, sample_scaler, sample_metadata):
        """Test prediction with wrong features raises error"""
        package = PTradeModelPackage(sample_model, sample_scaler, sample_metadata)

        features = {
            'wrong_a': 1.0,
            'wrong_b': 2.0,
            'wrong_c': 3.0
        }

        with pytest.raises(ValueError, match="Feature mismatch"):
            package.predict(features)

    def test_predict_missing_features(self, sample_model, sample_scaler, sample_metadata):
        """Test prediction with missing features raises error"""
        package = PTradeModelPackage(sample_model, sample_scaler, sample_metadata)

        features = {
            'feat_a': 1.0,
            'feat_b': 2.0
            # Missing feat_c
        }

        with pytest.raises(ValueError, match="Missing features"):
            package.predict(features)

    def test_predict_without_model(self, sample_metadata):
        """Test prediction without model raises error"""
        package = PTradeModelPackage(model=None, metadata=sample_metadata)

        features = {'feat_a': 1.0, 'feat_b': 2.0, 'feat_c': 3.0}

        with pytest.raises(ValueError, match="Model not loaded"):
            package.predict(features)

    def test_predict_without_metadata(self, sample_model):
        """Test prediction without metadata raises error"""
        package = PTradeModelPackage(model=sample_model, metadata=None)

        features = {'feat_a': 1.0, 'feat_b': 2.0, 'feat_c': 3.0}

        with pytest.raises(ValueError, match="Metadata not loaded"):
            package.predict(features)

    def test_load_from_files_json(self, sample_model, sample_metadata, temp_dir):
        """Test loading from separate JSON files"""
        # Save model and metadata separately
        model_path = temp_dir / 'model.json'
        metadata_path = temp_dir / 'metadata.json'

        sample_model.save_model(str(model_path))
        sample_metadata.save(metadata_path)

        # Load using load_from_files
        package = PTradeModelPackage.load_from_files(
            str(model_path),
            str(metadata_path)
        )

        assert package.model is not None
        assert package.metadata is not None
        assert package.metadata.features == sample_metadata.features

    def test_load_from_files_model_format(self, sample_model, temp_dir):
        """Test loading from .model file"""
        model_path = temp_dir / 'model.model'
        sample_model.save_model(str(model_path))

        package = PTradeModelPackage.load_from_files(str(model_path))

        assert package.model is not None

    def test_load_from_files_with_scaler(
        self, sample_model, sample_scaler, sample_metadata, temp_dir
    ):
        """Test loading with scaler"""
        import pickle

        # Save all components
        model_path = temp_dir / 'model.json'
        metadata_path = temp_dir / 'metadata.json'
        scaler_path = temp_dir / 'scaler.pkl'

        sample_model.save_model(str(model_path))
        sample_metadata.save(metadata_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(sample_scaler, f)

        # Load
        package = PTradeModelPackage.load_from_files(
            str(model_path),
            str(metadata_path),
            str(scaler_path)
        )

        assert package.model is not None
        assert package.metadata is not None
        assert package.scaler is not None

    def test_load_from_files_nonexistent(self, temp_dir):
        """Test loading nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            PTradeModelPackage.load_from_files(str(temp_dir / 'nonexistent.json'))

    def test_load_from_files_unsupported_format(self, temp_dir):
        """Test loading unsupported format raises error"""
        bad_file = temp_dir / 'model.txt'
        bad_file.write_text('not a model')

        with pytest.raises(ValueError, match="Unsupported model format"):
            PTradeModelPackage.load_from_files(str(bad_file))

    def test_summary(self, sample_model, sample_metadata):
        """Test package summary"""
        package = PTradeModelPackage(sample_model, metadata=sample_metadata)

        summary = package.summary()

        assert isinstance(summary, str)
        assert sample_metadata.model_id in summary
        assert 'feat_a' in summary

    def test_summary_without_metadata(self, sample_model):
        """Test summary without metadata"""
        package = PTradeModelPackage(sample_model, metadata=None)

        summary = package.summary()

        assert isinstance(summary, str)
        assert 'No metadata' in summary

    def test_roundtrip_consistency(
        self, sample_model, sample_scaler, sample_metadata, temp_dir
    ):
        """Test prediction consistency after save/load"""
        # Create package and make prediction
        package1 = PTradeModelPackage(sample_model, sample_scaler, sample_metadata)
        features = {'feat_a': 1.0, 'feat_b': 2.0, 'feat_c': 3.0}
        pred1 = package1.predict(features)

        # Save and load
        save_path = temp_dir / 'model.ptp'
        package1.save(str(save_path))
        package2 = PTradeModelPackage.load(str(save_path))

        # Make same prediction
        pred2 = package2.predict(features)

        # Should be identical
        assert np.isclose(pred1, pred2, rtol=1e-6)
