import json

import pytest
from fastapi.testclient import TestClient

from backend import repository
from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_start_requires_source():
    response = client.post("/api/sessions/start", data={})
    assert response.status_code == 400


def test_start_rejects_unsupported_file():
    response = client.post(
        "/api/sessions/start",
        data={"save_video": "false"},
        files={"file": ("catatan.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_session_list_is_a_list():
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_missing_session_returns_404():
    assert client.get("/api/sessions/tidak-ada").status_code == 404
    assert client.post("/api/sessions/tidak-ada/review", json={"acked": []}).status_code == 404


def test_repository_rejects_path_traversal():
    with pytest.raises(ValueError):
        repository.get_session("../rahasia")


def test_save_review_keeps_labels_when_omitted(tmp_path, monkeypatch):
    session = tmp_path / "sesi-1"
    session.mkdir()
    (session / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(repository, "OUTPUTS_DIR", tmp_path)

    repository.save_review("sesi-1", ["1"], {"1": "true_fall"})
    repository.save_review("sesi-1", ["1", "2"])

    review = json.loads((session / "review.json").read_text(encoding="utf-8"))
    assert review["acked"] == ["1", "2"]
    assert review["labels"] == {"1": "true_fall"}
