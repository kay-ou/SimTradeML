"""Test basic import."""
import pytest


def test_version() -> None:
    """Test that version is defined."""
    import simtrademl

    assert hasattr(simtrademl, "__version__")
    assert isinstance(simtrademl.__version__, str)
