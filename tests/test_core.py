from fastapi.testclient import TestClient
from main import app
from session_store import init_db, save_session, load_session

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_session_persistence(tmp_path, monkeypatch):
    import session_store
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "test.db")
    init_db()
    save_session("abc123", {"cols": ["a", "b"], "rows": 100})
    result = load_session("abc123")
    assert result["rows"] == 100


def test_missing_session():
    token_resp = client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    token = token_resp.json()["access_token"]
    r = client.get(
        "/sessions/nonexistent-id-xyz",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404
