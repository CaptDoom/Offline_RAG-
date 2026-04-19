def test_read_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_get_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "index" in data
    assert "ollama_active" in data

def test_chat_empty_query(client):
    response = client.post("/api/chat", json={"query": "   "})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "cannot be empty" in data["error"].lower()

def test_batch_empty_queries(client):
    response = client.post("/api/batch", json={"queries": []})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["results"] == []

def test_invalid_config_update(client):
    # Missing fields should return 422
    response = client.post("/api/config", json={"chunk_size": 100})
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert "Validation error" in data["error"]
