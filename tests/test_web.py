from collections import defaultdict

from fastapi.testclient import TestClient

from findex.index import DocMeta, Index, Posting
from findex.web.app import app, get_index


def build_test_index() -> Index:
    idx = Index(
        postings=defaultdict(
            list,
            {
                "whale": [Posting(doc_id=0, tf=3, positions=(1, 5, 10))],
                "captain": [Posting(doc_id=0, tf=1, positions=(2,))],
            },
        ),
        doc_lengths={0: 100},
        doc_meta={0: DocMeta(doc_id=0, title="Moby Dick", path="moby.txt")},
        raw_docs={0: "The captain sailed after the white whale on the high seas."},
    )
    return idx


def test_health_503_when_unloaded() -> None:
    with TestClient(app) as client:
        app.state.index = None
        res = client.get("/health")
        assert res.status_code == 503
        assert res.json()["detail"] == "Index not loaded"


def test_health_200_when_loaded() -> None:
    with TestClient(app) as client:
        app.state.index = build_test_index()
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


def test_search_api_success() -> None:
    mock_idx = build_test_index()
    app.dependency_overrides[get_index] = lambda: mock_idx
    try:
        with TestClient(app) as client:
            res = client.get("/search", params={"q": "whale", "k": 5, "page": 1})
            assert res.status_code == 200
            data = res.json()
            assert data["total"] == 1
            assert data["results"][0]["doc_id"] == 0
            assert data["results"][0]["title"] == "Moby Dick"
            assert "whale" in data["results"][0]["snippet"].casefold()
    finally:
        app.dependency_overrides.clear()


def test_search_api_validation_errors_422() -> None:
    mock_idx = build_test_index()
    app.dependency_overrides[get_index] = lambda: mock_idx
    try:
        with TestClient(app) as client:
            assert client.get("/search").status_code == 422
            assert client.get("/search", params={"q": ""}).status_code == 422
            assert (
                client.get("/search", params={"q": "whale", "k": 150}).status_code
                == 422
            )
            assert (
                client.get("/search", params={"q": "whale", "page": 0}).status_code
                == 422
            )
    finally:
        app.dependency_overrides.clear()


def test_get_document_success_and_404() -> None:
    mock_idx = build_test_index()
    app.dependency_overrides[get_index] = lambda: mock_idx
    try:
        with TestClient(app) as client:
            res_ok = client.get("/docs/0")
            assert res_ok.status_code == 200
            assert "captain sailed" in res_ok.text

            res_404 = client.get("/docs/999")
            assert res_404.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_stats_route() -> None:
    mock_idx = build_test_index()
    app.dependency_overrides[get_index] = lambda: mock_idx
    try:
        with TestClient(app) as client:
            res = client.get("/stats")
            assert res.status_code == 200
            data = res.json()
            assert data["doc_count"] == 1
            assert data["vocabulary_size"] == 2
            assert data["uptime_seconds"] >= 0
    finally:
        app.dependency_overrides.clear()


def test_html_ui_renders() -> None:
    mock_idx = build_test_index()
    app.dependency_overrides[get_index] = lambda: mock_idx
    try:
        with TestClient(app) as client:
            res_home = client.get("/")
            assert res_home.status_code == 200
            assert "findex" in res_home.text

            res_search = client.get("/", params={"q": "whale"})
            assert res_search.status_code == 200
            assert "Moby Dick" in res_search.text

            res_doc = client.get("/doc/0")
            assert res_doc.status_code == 200
            assert "Moby Dick" in res_doc.text
    finally:
        app.dependency_overrides.clear()
