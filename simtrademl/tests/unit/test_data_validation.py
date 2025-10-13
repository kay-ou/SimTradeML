"""Unit tests for DataValidator."""

import pytest
import pandas as pd
import numpy as np

from simtrademl.data.validation import DataValidator, ValidationResult


@pytest.mark.unit
class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_init_success(self) -> None:
        """Test ValidationResult initialization with success=True."""
        result = ValidationResult(success=True)
        assert result.success is True
        assert result.errors == []
        assert result.warnings == []
        assert result.statistics == {}

    def test_add_error(self) -> None:
        """Test adding an error marks success as False."""
        result = ValidationResult(success=True)
        result.add_error("Test error")

        assert result.success is False
        assert "Test error" in result.errors
        assert len(result.errors) == 1

    def test_add_warning(self) -> None:
        """Test adding a warning doesn't affect success."""
        result = ValidationResult(success=True)
        result.add_warning("Test warning")

        assert result.success is True
        assert "Test warning" in result.warnings
        assert len(result.warnings) == 1


@pytest.mark.unit
class TestDataValidatorInit:
    """Tests for DataValidator initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values from settings."""
        validator = DataValidator()

        # Should load defaults from settings
        assert validator.missing_threshold >= 0.0
        assert validator.missing_threshold <= 1.0
        assert validator.ks_pvalue_threshold >= 0.0
        assert validator.ks_pvalue_threshold <= 1.0

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        validator = DataValidator(
            missing_threshold=0.2,
            ks_pvalue_threshold=0.01,
        )

        assert validator.missing_threshold == 0.2
        assert validator.ks_pvalue_threshold == 0.01


@pytest.mark.unit
class TestValidateSchema:
    """Tests for schema validation."""

    def test_validate_schema_success(self) -> None:
        """Test schema validation with all required columns present."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'close': [150.0, 250.0],
            'volume': [1000, 2000],
        })

        validator = DataValidator()
        result = validator.validate_schema(
            df,
            required_columns=['symbol', 'close', 'volume'],
        )

        assert result.success is True
        assert len(result.errors) == 0
        assert result.statistics['num_rows'] == 2
        assert result.statistics['num_columns'] == 3

    def test_validate_schema_missing_columns(self) -> None:
        """Test schema validation with missing required columns."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'close': [150.0, 250.0],
        })

        validator = DataValidator()
        result = validator.validate_schema(
            df,
            required_columns=['symbol', 'close', 'volume'],
        )

        assert result.success is False
        assert len(result.errors) == 1
        assert 'volume' in result.errors[0]

    def test_validate_schema_with_types(self) -> None:
        """Test schema validation with type checking."""
        df = pd.DataFrame({
            'close': [150.0, 250.0],
            'volume': [1000, 2000],
        })

        validator = DataValidator()
        result = validator.validate_schema(
            df,
            required_columns=['close', 'volume'],
            column_types={'close': 'float64', 'volume': 'int64'},
        )

        assert result.success is True
        # Type mismatches should be warnings, not errors
        assert len(result.errors) == 0

    def test_validate_schema_type_mismatch(self) -> None:
        """Test schema validation with type mismatches."""
        df = pd.DataFrame({
            'close': [150, 250],  # int instead of float
            'volume': [1000, 2000],
        })

        validator = DataValidator()
        result = validator.validate_schema(
            df,
            required_columns=['close', 'volume'],
            column_types={'close': 'float64', 'volume': 'int64'},
        )

        # Should succeed but have warnings
        assert result.success is True
        assert len(result.warnings) > 0


@pytest.mark.unit
class TestCheckMissingValues:
    """Tests for missing value detection."""

    def test_check_missing_no_missing(self) -> None:
        """Test missing value check with no missing values."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200],
        })

        validator = DataValidator(missing_threshold=0.1)
        ratios = validator.check_missing_values(df)

        assert ratios['close'] == 0.0
        assert ratios['volume'] == 0.0

    def test_check_missing_some_missing(self) -> None:
        """Test missing value check with some missing values."""
        df = pd.DataFrame({
            'close': [100.0, None, 102.0],
            'volume': [1000, 1100, None],
        })

        validator = DataValidator(missing_threshold=0.5)
        ratios = validator.check_missing_values(df)

        assert ratios['close'] == pytest.approx(1/3, rel=1e-2)
        assert ratios['volume'] == pytest.approx(1/3, rel=1e-2)

    def test_check_missing_all_missing(self) -> None:
        """Test missing value check with all missing values."""
        df = pd.DataFrame({
            'close': [None, None, None],
            'volume': [None, None, None],
        })

        validator = DataValidator(missing_threshold=0.5)
        ratios = validator.check_missing_values(df)

        assert ratios['close'] == 1.0
        assert ratios['volume'] == 1.0

    def test_check_missing_specific_columns(self) -> None:
        """Test missing value check for specific columns only."""
        df = pd.DataFrame({
            'close': [100.0, None, 102.0],
            'volume': [1000, 1100, None],
            'high': [105.0, 106.0, 107.0],
        })

        validator = DataValidator()
        ratios = validator.check_missing_values(df, columns=['close', 'volume'])

        assert 'close' in ratios
        assert 'volume' in ratios
        assert 'high' not in ratios

    def test_check_missing_nonexistent_column(self) -> None:
        """Test missing value check with nonexistent column."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
        })

        validator = DataValidator()
        ratios = validator.check_missing_values(df, columns=['close', 'volume'])

        # Should only return ratios for existing columns
        assert 'close' in ratios
        assert 'volume' not in ratios


@pytest.mark.unit
class TestDetectDistributionShift:
    """Tests for distribution shift detection."""

    def test_detect_shift_no_shift(self) -> None:
        """Test distribution shift detection with no shift."""
        np.random.seed(42)
        df_reference = pd.DataFrame({
            'close': np.random.normal(100, 10, 1000),
            'volume': np.random.normal(1000, 100, 1000),
        })
        df_new = pd.DataFrame({
            'close': np.random.normal(100, 10, 1000),
            'volume': np.random.normal(1000, 100, 1000),
        })

        validator = DataValidator(ks_pvalue_threshold=0.05)
        pvalues = validator.detect_distribution_shift(
            df_reference, df_new, ['close', 'volume']
        )

        # With same distribution, p-values should be high (>0.05)
        assert pvalues['close'] > 0.01
        assert pvalues['volume'] > 0.01

    def test_detect_shift_with_shift(self) -> None:
        """Test distribution shift detection with significant shift."""
        np.random.seed(42)
        df_reference = pd.DataFrame({
            'close': np.random.normal(100, 10, 1000),
        })
        df_new = pd.DataFrame({
            'close': np.random.normal(150, 10, 1000),  # Shifted mean
        })

        validator = DataValidator(ks_pvalue_threshold=0.05)
        pvalues = validator.detect_distribution_shift(
            df_reference, df_new, ['close']
        )

        # With shifted distribution, p-value should be very low (<0.05)
        assert pvalues['close'] < 0.05

    def test_detect_shift_identical_data(self) -> None:
        """Test distribution shift detection with identical data."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
        })

        validator = DataValidator(ks_pvalue_threshold=0.05)
        pvalues = validator.detect_distribution_shift(
            df, df, ['close', 'volume']
        )

        # Identical distributions should have p-value = 1.0
        assert pvalues['close'] == 1.0
        assert pvalues['volume'] == 1.0

    def test_detect_shift_with_nans(self) -> None:
        """Test distribution shift detection handles NaN values."""
        df_reference = pd.DataFrame({
            'close': [100.0, 101.0, None, 103.0, 104.0],
        })
        df_new = pd.DataFrame({
            'close': [100.0, None, 102.0, 103.0, 104.0],
        })

        validator = DataValidator()
        pvalues = validator.detect_distribution_shift(
            df_reference, df_new, ['close']
        )

        # Should handle NaNs and compute KS test on valid values
        assert 'close' in pvalues
        assert pvalues['close'] > 0.0

    def test_detect_shift_missing_column(self) -> None:
        """Test distribution shift detection with missing column."""
        df_reference = pd.DataFrame({
            'close': [100, 101, 102],
        })
        df_new = pd.DataFrame({
            'volume': [1000, 1100, 1200],
        })

        validator = DataValidator()
        pvalues = validator.detect_distribution_shift(
            df_reference, df_new, ['close', 'volume']
        )

        # Should not return p-values for missing columns
        assert len(pvalues) == 0

    def test_detect_shift_empty_column(self) -> None:
        """Test distribution shift detection with empty column (all NaNs)."""
        df_reference = pd.DataFrame({
            'close': [None, None, None],
        })
        df_new = pd.DataFrame({
            'close': [None, None, None],
        })

        validator = DataValidator()
        pvalues = validator.detect_distribution_shift(
            df_reference, df_new, ['close']
        )

        # Should not return p-value for empty columns
        assert len(pvalues) == 0


@pytest.mark.unit
class TestDataValidatorIntegration:
    """Integration tests for DataValidator with multiple methods."""

    def test_full_validation_workflow(self) -> None:
        """Test complete validation workflow."""
        # Create training data
        np.random.seed(42)
        df_train = pd.DataFrame({
            'symbol': ['AAPL'] * 100,
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'close': np.random.normal(100, 10, 100),
            'volume': np.random.normal(1000, 100, 100),
        })

        # Create test data with some issues
        df_test = pd.DataFrame({
            'symbol': ['AAPL'] * 50,
            'timestamp': pd.date_range('2024-04-11', periods=50),
            'close': [100.0] * 48 + [None, None],  # 2 missing values
            'volume': np.random.normal(1000, 100, 50),
        })

        validator = DataValidator(
            missing_threshold=0.05,  # 5% threshold
            ks_pvalue_threshold=0.05,
        )

        # 1. Validate schema
        schema_result = validator.validate_schema(
            df_test,
            required_columns=['symbol', 'timestamp', 'close', 'volume'],
        )
        assert schema_result.success is True

        # 2. Check missing values
        missing_ratios = validator.check_missing_values(df_test)
        assert missing_ratios['close'] == pytest.approx(2/50, rel=1e-2)

        # 3. Detect distribution shift
        pvalues = validator.detect_distribution_shift(
            df_train, df_test, ['close', 'volume']
        )
        # close should have no shift (same values)
        # volume should have no shift (same distribution)
        assert 'close' in pvalues
        assert 'volume' in pvalues
