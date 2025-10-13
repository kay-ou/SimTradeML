"""Unit tests for DataCleaner."""

import pytest
import pandas as pd
import numpy as np

from simtrademl.data.cleaning import (
    DataCleaner,
    CleaningResult,
    MissingStrategy,
    OutlierStrategy,
)


@pytest.mark.unit
class TestCleaningResult:
    """Tests for CleaningResult dataclass."""

    def test_init(self) -> None:
        """Test CleaningResult initialization."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = CleaningResult(data=df)

        assert len(result.data) == 3
        assert result.rows_removed == 0
        assert result.missing_filled == {}
        assert result.outliers_handled == {}


@pytest.mark.unit
class TestDataCleanerInit:
    """Tests for DataCleaner initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with default values."""
        cleaner = DataCleaner()

        assert cleaner.missing_strategy == MissingStrategy.FORWARD_FILL
        assert cleaner.outlier_strategy == OutlierStrategy.CLIP
        assert cleaner.outlier_method == "iqr"
        assert cleaner.iqr_multiplier == 1.5
        assert cleaner.zscore_threshold == 3.0

    def test_init_custom(self) -> None:
        """Test initialization with custom values."""
        cleaner = DataCleaner(
            missing_strategy=MissingStrategy.MEAN,
            outlier_strategy=OutlierStrategy.REMOVE,
            outlier_method="zscore",
            iqr_multiplier=2.0,
            zscore_threshold=2.5,
        )

        assert cleaner.missing_strategy == MissingStrategy.MEAN
        assert cleaner.outlier_strategy == OutlierStrategy.REMOVE
        assert cleaner.outlier_method == "zscore"
        assert cleaner.iqr_multiplier == 2.0
        assert cleaner.zscore_threshold == 2.5

    def test_init_invalid_method(self) -> None:
        """Test initialization with invalid outlier method."""
        with pytest.raises(ValueError, match="Invalid outlier_method"):
            DataCleaner(outlier_method="invalid")


@pytest.mark.unit
class TestMissingValueStrategies:
    """Tests for missing value handling strategies."""

    def test_forward_fill(self) -> None:
        """Test forward fill strategy."""
        df = pd.DataFrame({'price': [100.0, None, None, 104.0]})
        cleaner = DataCleaner(missing_strategy=MissingStrategy.FORWARD_FILL)

        result = cleaner.handle_missing_values(df, ['price'])

        assert result['price'].tolist() == [100.0, 100.0, 100.0, 104.0]

    def test_backward_fill(self) -> None:
        """Test backward fill strategy."""
        df = pd.DataFrame({'price': [100.0, None, None, 104.0]})
        cleaner = DataCleaner(missing_strategy=MissingStrategy.BACKWARD_FILL)

        result = cleaner.handle_missing_values(df, ['price'])

        assert result['price'].tolist() == [100.0, 104.0, 104.0, 104.0]

    def test_interpolate(self) -> None:
        """Test interpolation strategy."""
        df = pd.DataFrame({'price': [100.0, None, None, 104.0]})
        cleaner = DataCleaner(missing_strategy=MissingStrategy.INTERPOLATE)

        result = cleaner.handle_missing_values(df, ['price'])

        # Linear interpolation: 100 -> 101.33 -> 102.67 -> 104
        expected = [100.0, pytest.approx(101.33, rel=1e-2),
                    pytest.approx(102.67, rel=1e-2), 104.0]
        assert result['price'].tolist() == expected

    def test_drop(self) -> None:
        """Test drop strategy."""
        df = pd.DataFrame({
            'price': [100.0, None, 102.0, 104.0],
            'volume': [1000, 1100, 1200, 1300],
        })
        cleaner = DataCleaner(missing_strategy=MissingStrategy.DROP)

        result = cleaner.handle_missing_values(df, ['price'])

        assert len(result) == 3  # One row dropped
        assert result['price'].isna().sum() == 0

    def test_mean(self) -> None:
        """Test mean imputation strategy."""
        df = pd.DataFrame({'price': [100.0, None, 102.0, None]})
        cleaner = DataCleaner(missing_strategy=MissingStrategy.MEAN)

        result = cleaner.handle_missing_values(df, ['price'])

        # Mean of 100 and 102 is 101
        assert result['price'].tolist() == [100.0, 101.0, 102.0, 101.0]

    def test_median(self) -> None:
        """Test median imputation strategy."""
        df = pd.DataFrame({'price': [100.0, None, 102.0, 110.0]})
        cleaner = DataCleaner(missing_strategy=MissingStrategy.MEDIAN)

        result = cleaner.handle_missing_values(df, ['price'])

        # Median of 100, 102, 110 is 102
        assert result['price'].tolist() == [100.0, 102.0, 102.0, 110.0]

    def test_no_missing_values(self) -> None:
        """Test handling when no missing values exist."""
        df = pd.DataFrame({'price': [100.0, 101.0, 102.0]})
        cleaner = DataCleaner()

        result = cleaner.handle_missing_values(df, ['price'])

        pd.testing.assert_frame_equal(result, df)

    def test_nonexistent_column(self) -> None:
        """Test handling nonexistent column."""
        df = pd.DataFrame({'price': [100.0, 101.0]})
        cleaner = DataCleaner()

        result = cleaner.handle_missing_values(df, ['volume'])

        # Should return unchanged dataframe
        pd.testing.assert_frame_equal(result, df)


@pytest.mark.unit
class TestOutlierDetection:
    """Tests for outlier detection methods."""

    def test_detect_outliers_iqr_no_outliers(self) -> None:
        """Test IQR method with no outliers."""
        np.random.seed(42)
        df = pd.DataFrame({'price': np.random.normal(100, 10, 100)})
        cleaner = DataCleaner(outlier_method="iqr")

        outliers = cleaner.detect_outliers(df, ['price'])

        # With normal distribution, should have few or no outliers
        assert outliers.sum() < 10  # Less than 10% outliers

    def test_detect_outliers_iqr_with_outliers(self) -> None:
        """Test IQR method with clear outliers."""
        df = pd.DataFrame({'price': [100, 101, 102, 103, 1000]})  # 1000 is outlier
        cleaner = DataCleaner(outlier_method="iqr")

        outliers = cleaner.detect_outliers(df, ['price'])

        assert outliers.iloc[4] == True  # Last value is outlier
        assert outliers.sum() == 1

    def test_detect_outliers_zscore_no_outliers(self) -> None:
        """Test Z-score method with no outliers."""
        np.random.seed(42)
        df = pd.DataFrame({'price': np.random.normal(100, 10, 100)})
        cleaner = DataCleaner(outlier_method="zscore", zscore_threshold=3.0)

        outliers = cleaner.detect_outliers(df, ['price'])

        # With threshold=3.0, should have very few outliers
        assert outliers.sum() < 5

    def test_detect_outliers_zscore_with_outliers(self) -> None:
        """Test Z-score method with clear outliers."""
        # With more normal values, the outlier becomes more apparent
        # Z-score for 500 with this data: ~2.65 > 2.0
        df = pd.DataFrame({'price': [100, 100, 100, 100, 100, 100, 100, 500]})
        cleaner = DataCleaner(outlier_method="zscore", zscore_threshold=2.0)

        outliers = cleaner.detect_outliers(df, ['price'])

        assert outliers.iloc[7] == True  # Last value is outlier

    def test_detect_outliers_empty_column(self) -> None:
        """Test outlier detection with empty column."""
        df = pd.DataFrame({'price': [None, None, None]})
        cleaner = DataCleaner()

        outliers = cleaner.detect_outliers(df, ['price'])

        assert outliers.sum() == 0  # No outliers in empty data

    def test_detect_outliers_multiple_columns(self) -> None:
        """Test outlier detection across multiple columns."""
        df = pd.DataFrame({
            'price': [100, 101, 102, 1000],  # Outlier at index 3
            'volume': [1000, 1100, 50000, 1300],  # Outlier at index 2
        })
        cleaner = DataCleaner(outlier_method="iqr")

        outliers = cleaner.detect_outliers(df, ['price', 'volume'])

        # Should detect outliers in both columns
        assert outliers.iloc[2] == True  # Volume outlier
        assert outliers.iloc[3] == True  # Price outlier


@pytest.mark.unit
class TestOutlierHandling:
    """Tests for outlier handling strategies."""

    def test_clip_strategy(self) -> None:
        """Test clipping outliers to bounds."""
        df = pd.DataFrame({'price': [100, 101, 102, 103, 1000]})
        cleaner = DataCleaner(
            outlier_strategy=OutlierStrategy.CLIP,
            outlier_method="iqr",
        )
        outliers = cleaner.detect_outliers(df, ['price'])

        result = cleaner.handle_outliers(df, outliers, ['price'])

        # Outlier should be clipped to upper bound
        assert result['price'].iloc[4] < 1000
        assert result['price'].iloc[4] > 103
        assert len(result) == len(df)  # No rows removed

    def test_remove_strategy(self) -> None:
        """Test removing outlier rows."""
        df = pd.DataFrame({'price': [100, 101, 102, 103, 1000]})
        cleaner = DataCleaner(
            outlier_strategy=OutlierStrategy.REMOVE,
            outlier_method="iqr",
        )
        outliers = cleaner.detect_outliers(df, ['price'])

        result = cleaner.handle_outliers(df, outliers, ['price'])

        assert len(result) == 4  # One row removed
        assert 1000 not in result['price'].values

    def test_transform_strategy(self) -> None:
        """Test log transform strategy."""
        df = pd.DataFrame({'price': [100.0, 101.0, 102.0, 103.0, 1000.0]})
        cleaner = DataCleaner(
            outlier_strategy=OutlierStrategy.TRANSFORM,
            outlier_method="iqr",
        )
        outliers = cleaner.detect_outliers(df, ['price'])

        result = cleaner.handle_outliers(df, outliers, ['price'])

        # All values should be log transformed
        assert result['price'].iloc[0] < 100  # log1p(100) < 100
        assert result['price'].iloc[4] < 1000  # log1p(1000) < 1000

    def test_transform_negative_values(self) -> None:
        """Test transform strategy skips negative values."""
        df = pd.DataFrame({'price': [-10.0, 100.0, 101.0]})
        cleaner = DataCleaner(outlier_strategy=OutlierStrategy.TRANSFORM)
        outliers = pd.Series([False, False, False])

        result = cleaner.handle_outliers(df, outliers, ['price'])

        # Should not transform due to negative values
        pd.testing.assert_frame_equal(result, df)


@pytest.mark.unit
class TestCleanMethod:
    """Tests for the main clean() method."""

    def test_clean_no_issues(self) -> None:
        """Test cleaning data with no issues."""
        df = pd.DataFrame({'price': [100, 101, 102, 103]})
        cleaner = DataCleaner()

        result = cleaner.clean(df)

        assert len(result.data) == 4
        assert result.rows_removed == 0
        assert len(result.missing_filled) == 0
        assert len(result.outliers_handled) == 0

    def test_clean_with_missing_and_outliers(self) -> None:
        """Test cleaning data with both missing values and outliers."""
        df = pd.DataFrame({
            'price': [100.0, None, 102.0, 103.0, 104.0, 105.0, 106.0, 1000.0],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        })
        cleaner = DataCleaner(
            missing_strategy=MissingStrategy.MEAN,
            outlier_strategy=OutlierStrategy.CLIP,
        )

        result = cleaner.clean(df)

        # Missing values should be filled
        assert result.data['price'].isna().sum() == 0
        # Outlier should be clipped (with more data points, 1000 is clearly an outlier)
        assert result.data['price'].max() < 1000

    def test_clean_specific_columns(self) -> None:
        """Test cleaning only specific columns."""
        df = pd.DataFrame({
            'price': [100.0, None, 102.0],
            'volume': [1000, None, 1200],
        })
        cleaner = DataCleaner(missing_strategy=MissingStrategy.MEAN)

        result = cleaner.clean(df, columns=['price'])

        # Only price should be cleaned
        assert result.data['price'].isna().sum() == 0
        assert result.data['volume'].isna().sum() == 1

    def test_clean_non_numeric_columns(self) -> None:
        """Test cleaning with non-numeric columns."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'price': [100.0, 101.0, 102.0],
        })
        cleaner = DataCleaner()

        result = cleaner.clean(df)

        # Should only clean numeric columns
        assert 'symbol' in result.data.columns
        assert result.data['symbol'].tolist() == ['AAPL', 'MSFT', 'GOOGL']

    def test_clean_empty_dataframe(self) -> None:
        """Test cleaning empty dataframe."""
        df = pd.DataFrame()
        cleaner = DataCleaner()

        result = cleaner.clean(df)

        assert len(result.data) == 0
        assert result.rows_removed == 0

    def test_clean_all_missing(self) -> None:
        """Test cleaning with all missing values."""
        # Use float type explicitly so pandas recognizes it as numeric
        df = pd.DataFrame({'price': [np.nan, np.nan, np.nan]})
        cleaner = DataCleaner(missing_strategy=MissingStrategy.DROP)

        result = cleaner.clean(df)

        assert len(result.data) == 0
        assert result.rows_removed == 3


@pytest.mark.unit
class TestDataCleanerIntegration:
    """Integration tests for DataCleaner."""

    def test_full_pipeline(self) -> None:
        """Test complete cleaning pipeline."""
        np.random.seed(42)
        df = pd.DataFrame({
            'price': [100, None, 102, 103, 1000, 105, None],
            'volume': [1000, 1100, None, 1300, 1400, 50000, 1600],
        })

        cleaner = DataCleaner(
            missing_strategy=MissingStrategy.MEAN,
            outlier_strategy=OutlierStrategy.REMOVE,
            outlier_method="iqr",
        )

        result = cleaner.clean(df)

        # Should have no missing values
        assert result.data['price'].isna().sum() == 0
        assert result.data['volume'].isna().sum() == 0

        # Should have removed outlier rows
        assert result.rows_removed > 0

        # Should track statistics
        assert len(result.missing_filled) > 0

    def test_strategy_combinations(self) -> None:
        """Test various strategy combinations."""
        df = pd.DataFrame({'price': [100.0, None, 102.0, 1000.0]})

        # Test 1: Forward fill + Clip
        cleaner1 = DataCleaner(
            missing_strategy=MissingStrategy.FORWARD_FILL,
            outlier_strategy=OutlierStrategy.CLIP,
        )
        result1 = cleaner1.clean(df)
        assert result1.data['price'].isna().sum() == 0
        assert len(result1.data) == 4

        # Test 2: Drop + Remove
        cleaner2 = DataCleaner(
            missing_strategy=MissingStrategy.DROP,
            outlier_strategy=OutlierStrategy.REMOVE,
        )
        result2 = cleaner2.clean(df)
        assert len(result2.data) < 4  # Rows removed

        # Test 3: Median + Transform
        cleaner3 = DataCleaner(
            missing_strategy=MissingStrategy.MEDIAN,
            outlier_strategy=OutlierStrategy.TRANSFORM,
        )
        result3 = cleaner3.clean(df)
        assert result3.data['price'].isna().sum() == 0
