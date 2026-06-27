"""Initial package smoke tests."""

from scholaragent import __version__


def test_package_version() -> None:
    """The package exposes the expected initial version."""
    assert __version__ == "0.1.0"
