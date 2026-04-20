import pytest
import shutil
import uuid
from local_archive_ai.services import get_index_status

def test_index_status_missing_directory():
    temp_root = "tests/.tmp"
    target = f"{temp_root}/missing_{uuid.uuid4().hex}"
    shutil.rmtree(temp_root, ignore_errors=True)
    # Pass a Path that doesn't exist to get_index_status
    status = get_index_status(target)
    assert status["exists"] is False
    assert status["file_count"] == 0
    assert status["chunk_count"] == 0
