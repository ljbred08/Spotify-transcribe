"""API tests via FastAPI TestClient (no network, no models)."""
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_index_serves_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "Refrain" in res.text


def test_tracks_empty_listing():
    res = client.get("/api/tracks")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_job_roundtrip_and_unknown_track_404():
    res = client.post("/api/jobs", json={"url": "https://example.invalid/x"})
    assert res.status_code == 200
    job = res.json()
    assert job["status"] == "queued"
    assert client.get(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get("/api/tracks/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.delete("/api/tracks/nope").status_code == 404


def test_export_missing_track_is_404():
    res = client.get("/api/tracks/x/export", params={"fmt": "xml"})
    assert res.status_code == 404  # unknown track wins over unknown format
    # format validation itself is covered in test_exporters.py
