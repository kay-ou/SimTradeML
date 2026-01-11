# -*- coding: utf-8 -*-
"""
Unit tests for Config class
"""

import pytest
import tempfile
from pathlib import Path
from simtrademl.core.utils.config import Config


@pytest.mark.unit
class TestConfig:
    """Test configuration management"""

    def test_from_dict(self, sample_config_dict):
        """Test creating config from dictionary"""
        config = Config.from_dict(sample_config_dict)
        assert config.get('data.lookback_days') == 60
        assert config.get('model.type') == 'xgboost'

    def test_get_with_default(self):
        """Test get with default value"""
        config = Config.from_dict({})
        assert config.get('nonexistent.key', 'default') == 'default'

    def test_get_nested_value(self, sample_config_dict):
        """Test getting nested value with dot notation"""
        config = Config.from_dict(sample_config_dict)
        assert config.get('model.params.max_depth') == 4
        assert config.get('model.params.learning_rate') == 0.04

    def test_set_value(self):
        """Test setting configuration value"""
        config = Config.from_dict({})
        config.set('data.lookback_days', 90)
        assert config.get('data.lookback_days') == 90

    def test_set_nested_value(self):
        """Test setting nested value"""
        config = Config.from_dict({})
        config.set('model.params.max_depth', 6)
        assert config.get('model.params.max_depth') == 6

    def test_update(self, sample_config_dict):
        """Test updating config with new values"""
        config = Config.from_dict(sample_config_dict)
        config.update({'data': {'lookback_days': 120}})
        assert config.get('data.lookback_days') == 120
        # Other values should remain
        assert config.get('data.predict_days') == 5

    def test_to_dict(self, sample_config_dict):
        """Test exporting config to dictionary"""
        config = Config.from_dict(sample_config_dict)
        exported = config.to_dict()
        assert exported['data']['lookback_days'] == 60
        assert isinstance(exported, dict)

    def test_yaml_roundtrip(self, sample_config_dict):
        """Test saving and loading YAML"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / 'config.yml'

            # Save
            config = Config.from_dict(sample_config_dict)
            config.save(yaml_path)

            # Load
            loaded_config = Config.from_yaml(yaml_path)
            assert loaded_config.get('data.lookback_days') == 60
            assert loaded_config.get('model.type') == 'xgboost'

    def test_defaults(self):
        """Test default values"""
        config = Config.from_dict({})
        # Should get default value
        assert config.get('data.lookback_days') == 60  # From defaults
        assert config.get('model.type') == 'xgboost'

    def test_file_not_found(self):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml('/nonexistent/path/config.yml')
