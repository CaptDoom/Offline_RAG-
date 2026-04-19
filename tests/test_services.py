import pytest
from local_archive_ai.services import get_index_status

def test_index_status_missing_directory(tmp_path):
    # Pass a Path that doesn't exist to get_index_status
    status = get_index_status(str(tmp_path / "non_existent"))
    assert status["exists"] is False
    assert status["file_count"] == 0
    assert status["chunk_count"] == 0
