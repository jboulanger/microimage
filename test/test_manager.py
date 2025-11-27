import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from microimage import ExperimentManager, MetadataParser


@pytest.fixture
def mock_dirs(tmp_path):
    """Create temporary data, results, and db directories for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    db_path = tmp_path / "project.db"
    return data_dir, results_dir, db_path


def test_manager_scan_with_parser(mock_dirs):
    """
    Tests that the ExperimentManager can scan a directory and use the MetadataParser
    to register new items in the database.
    """
    data_dir, results_dir, db_path = mock_dirs

    # Create a dummy file for the scanner to find
    dummy_file = data_dir / "test_image.lsm"
    dummy_file.touch()

    # Mock the MetadataParser to avoid actual file reading
    mock_parser = MagicMock(spec=MetadataParser)
    mock_parser.return_value = [
        {"internal_index": "0", "metadata": {"channel": "DAPI"}},
        {"internal_index": "1", "metadata": {"channel": "GFP"}},
    ]

    manager = ExperimentManager(data_dir, results_dir, db_path)
    manager.scan_data_directory(extensions=[".lsm"], multi_file_parser=mock_parser)

    assert len(manager) == 2
    assert "test_image.lsm:0" in manager
    assert "test_image.lsm:1" in manager
    assert manager["test_image.lsm:1"].metadata["channel"] == "GFP"
    mock_parser.assert_called_once_with(dummy_file)
