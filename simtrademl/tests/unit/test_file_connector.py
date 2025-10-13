"""Unit tests for FileDataConnector."""

from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from simtrademl.data.connectors.file import FileDataConnector


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100),
            "symbol": ["AAPL"] * 100,
            "open": range(100, 200),
            "high": range(105, 205),
            "low": range(95, 195),
            "close": range(100, 200),
            "volume": range(1000, 1100),
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a sample Parquet file for testing."""
    parquet_file = tmp_path / "test_data.parquet"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100),
            "symbol": ["AAPL"] * 100,
            "open": range(100, 200),
            "high": range(105, 205),
            "low": range(95, 195),
            "close": range(100, 200),
            "volume": range(1000, 1100),
        }
    )
    df.to_parquet(parquet_file, index=False)
    return parquet_file


class TestFileDataConnectorInit:
    """Tests for FileDataConnector initialization."""

    def test_init_with_csv_file(self, sample_csv: Path) -> None:
        """Test initialization with CSV file."""
        connector = FileDataConnector(sample_csv)
        assert connector.file_path == sample_csv
        assert connector.file_format == "csv"

    def test_init_with_parquet_file(self, sample_parquet: Path) -> None:
        """Test initialization with Parquet file."""
        connector = FileDataConnector(sample_parquet)
        assert connector.file_path == sample_parquet
        assert connector.file_format == "parquet"

    def test_init_with_string_path(self, sample_csv: Path) -> None:
        """Test initialization with string path."""
        connector = FileDataConnector(str(sample_csv))
        assert connector.file_path == sample_csv
        assert connector.file_format == "csv"

    def test_init_with_nonexistent_file(self, tmp_path: Path) -> None:
        """Test initialization with nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            FileDataConnector(tmp_path / "nonexistent.csv")

    def test_init_with_unsupported_format(self, tmp_path: Path) -> None:
        """Test initialization with unsupported format raises error."""
        unsupported_file = tmp_path / "data.txt"
        unsupported_file.write_text("test")

        with pytest.raises(ValueError, match="Unsupported file format"):
            FileDataConnector(unsupported_file)


class TestFileDataConnectorConnect:
    """Tests for connect method."""

    def test_connect_csv(self, sample_csv: Path) -> None:
        """Test connect method for CSV (no-op)."""
        connector = FileDataConnector(sample_csv)
        connector.connect({})  # Should not raise any error

    def test_connect_parquet(self, sample_parquet: Path) -> None:
        """Test connect method for Parquet (no-op)."""
        connector = FileDataConnector(sample_parquet)
        connector.connect({})  # Should not raise any error


class TestCSVFetch:
    """Tests for CSV file fetching."""

    def test_fetch_entire_csv(self, sample_csv: Path) -> None:
        """Test fetching entire CSV file."""
        connector = FileDataConnector(sample_csv)
        df = connector.fetch()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]

    def test_fetch_csv_with_columns(self, sample_csv: Path) -> None:
        """Test fetching CSV with specific columns."""
        connector = FileDataConnector(sample_csv)
        df = connector.fetch(query={"columns": ["date", "close", "volume"]})

        assert list(df.columns) == ["date", "close", "volume"]
        assert len(df) == 100

    def test_fetch_csv_with_nrows(self, sample_csv: Path) -> None:
        """Test fetching CSV with limited rows."""
        connector = FileDataConnector(sample_csv)
        df = connector.fetch(query={"nrows": 10})

        assert len(df) == 10

    def test_fetch_csv_with_skiprows(self, sample_csv: Path) -> None:
        """Test fetching CSV with skipped rows."""
        connector = FileDataConnector(sample_csv)
        df = connector.fetch(query={"skiprows": 10})

        assert len(df) == 90

    def test_fetch_csv_with_parse_dates(self, sample_csv: Path) -> None:
        """Test fetching CSV with date parsing."""
        connector = FileDataConnector(sample_csv)
        df = connector.fetch(query={"parse_dates": ["date"]})

        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_fetch_csv_chunked(self, sample_csv: Path) -> None:
        """Test fetching CSV in chunks."""
        connector = FileDataConnector(sample_csv)
        chunks = connector.fetch(chunksize=20)

        assert isinstance(chunks, Iterator)

        chunk_list = list(chunks)
        assert len(chunk_list) == 5  # 100 rows / 20 per chunk

        for chunk in chunk_list:
            assert isinstance(chunk, pd.DataFrame)
            assert len(chunk) <= 20

    def test_fetch_csv_with_custom_separator(self, tmp_path: Path) -> None:
        """Test fetching CSV with custom separator."""
        csv_file = tmp_path / "semicolon.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(csv_file, sep=";", index=False)

        connector = FileDataConnector(csv_file)
        result = connector.fetch(query={"sep": ";"})

        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]


class TestParquetFetch:
    """Tests for Parquet file fetching."""

    def test_fetch_entire_parquet(self, sample_parquet: Path) -> None:
        """Test fetching entire Parquet file."""
        connector = FileDataConnector(sample_parquet)
        df = connector.fetch()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "date" in df.columns
        assert "close" in df.columns

    def test_fetch_parquet_with_columns(self, sample_parquet: Path) -> None:
        """Test fetching Parquet with specific columns."""
        connector = FileDataConnector(sample_parquet)
        df = connector.fetch(query={"columns": ["date", "close", "volume"]})

        assert list(df.columns) == ["date", "close", "volume"]
        assert len(df) == 100

    def test_fetch_parquet_with_nrows(self, sample_parquet: Path) -> None:
        """Test fetching Parquet with limited rows."""
        connector = FileDataConnector(sample_parquet)
        df = connector.fetch(query={"nrows": 10})

        assert len(df) == 10

    def test_fetch_parquet_with_skiprows(self, sample_parquet: Path) -> None:
        """Test fetching Parquet with skipped rows."""
        connector = FileDataConnector(sample_parquet)
        df = connector.fetch(query={"skiprows": 10})

        assert len(df) == 90

    def test_fetch_parquet_chunked(self, sample_parquet: Path) -> None:
        """Test fetching Parquet in chunks."""
        connector = FileDataConnector(sample_parquet)
        chunks = connector.fetch(chunksize=20)

        assert isinstance(chunks, Iterator)

        chunk_list = list(chunks)
        assert len(chunk_list) >= 5  # Should have at least 5 chunks

        total_rows = sum(len(chunk) for chunk in chunk_list)
        assert total_rows == 100


class TestDisconnect:
    """Tests for disconnect method."""

    def test_disconnect_csv(self, sample_csv: Path) -> None:
        """Test disconnect method for CSV (no-op)."""
        connector = FileDataConnector(sample_csv)
        connector.disconnect()  # Should not raise any error

    def test_disconnect_parquet(self, sample_parquet: Path) -> None:
        """Test disconnect method for Parquet (no-op)."""
        connector = FileDataConnector(sample_parquet)
        connector.disconnect()  # Should not raise any error


@pytest.mark.unit
class TestFileDataConnectorIntegration:
    """Integration tests for FileDataConnector."""

    def test_complete_csv_workflow(self, sample_csv: Path) -> None:
        """Test complete workflow with CSV file."""
        connector = FileDataConnector(sample_csv)
        connector.connect({})

        # Fetch entire file
        df = connector.fetch()
        assert len(df) == 100

        # Fetch with filters
        df_filtered = connector.fetch(
            query={"columns": ["date", "close"], "nrows": 10, "parse_dates": ["date"]}
        )
        assert len(df_filtered) == 10
        assert list(df_filtered.columns) == ["date", "close"]

        # Fetch in chunks
        total_rows = 0
        for chunk in connector.fetch(chunksize=25):
            total_rows += len(chunk)
        assert total_rows == 100

        connector.disconnect()

    def test_complete_parquet_workflow(self, sample_parquet: Path) -> None:
        """Test complete workflow with Parquet file."""
        connector = FileDataConnector(sample_parquet)
        connector.connect({})

        # Fetch entire file
        df = connector.fetch()
        assert len(df) == 100

        # Fetch with filters
        df_filtered = connector.fetch(query={"columns": ["date", "close"], "nrows": 10})
        assert len(df_filtered) == 10
        assert list(df_filtered.columns) == ["date", "close"]

        connector.disconnect()

    def test_large_file_chunking(self, tmp_path: Path) -> None:
        """Test chunking with larger file."""
        large_csv = tmp_path / "large.csv"
        df = pd.DataFrame(
            {
                "id": range(1000),
                "value": range(1000, 2000),
                "category": ["A", "B", "C"] * 333 + ["A"],
            }
        )
        df.to_csv(large_csv, index=False)

        connector = FileDataConnector(large_csv)

        chunks = list(connector.fetch(chunksize=100))
        assert len(chunks) == 10

        # Verify total rows
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 1000

    def test_parquet_alternative_extension(self, tmp_path: Path) -> None:
        """Test Parquet file with .parq extension."""
        parquet_file = tmp_path / "data.parq"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_parquet(parquet_file, index=False)

        connector = FileDataConnector(parquet_file)
        assert connector.file_format == "parquet"

        result = connector.fetch()
        assert len(result) == 3
