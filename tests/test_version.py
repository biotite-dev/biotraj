from importlib.metadata import version
import biotraj


def test_version():
    """
    Check if version imported from version.py is correct.
    """
    assert biotraj.__version__ == version("biotraj")
