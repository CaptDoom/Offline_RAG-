import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from api import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
