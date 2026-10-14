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
    r = client.get("/sessions/nonexistent-id-xyz")
    assert r.status_code == 404


def test_upload_without_auth(tmp_path):
    init_db()
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    with open(csv_path, "rb") as fp:
        response = client.post(
            "/upload",
            files={"file": (csv_path.name, fp, "text/csv")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["summary"]["rows"] == 2
