import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex.index import Index
from findex.ranking import BM25, Scorer, TfIdf
from findex.web.config import Settings, get_settings
from findex.web.schemas import SearchItem, SearchResponse, StatsResponse

log = logging.getLogger("findex.web")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    app.state.start_time = time.time()
    if settings.index_path.exists():
        log.info("Loading index from %s...", settings.index_path)
        app.state.index = Index.load(settings.index_path)
    else:
        log.warning(
            "Index file '%s' not found. /health will report 503.", settings.index_path
        )
        app.state.index = None
    yield


app = FastAPI(
    title="findex Search Service",
    version="0.7.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index is not loaded",
        )
    return index


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Any:
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = getattr(request.state, "request_id", "unknown")
    log.exception(
        "Unhandled error on %s (request_id=%s): %s", request.url.path, req_id, exc
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": req_id},
    )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "index", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Index not loaded"
        )
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def get_stats(
    request: Request,
    index: Annotated[Index, Depends(get_index)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StatsResponse:
    uptime = time.time() - getattr(request.app.state, "start_time", time.time())
    size = settings.index_path.stat().st_size if settings.index_path.exists() else 0
    return StatsResponse(
        index_size_bytes=size,
        doc_count=index.num_docs,
        vocabulary_size=len(index),
        uptime_seconds=round(uptime, 2),
    )


@app.get("/docs/{doc_id}", response_class=PlainTextResponse)
def get_document(
    doc_id: int,
    index: Annotated[Index, Depends(get_index)],
) -> str:
    text = index.raw_docs.get(doc_id)
    if text is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return text


@app.get("/search", response_model=SearchResponse)
def search_api(
    q: Annotated[str, Query(min_length=1, max_length=200, description="Query string")],
    k: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 10,
    scorer: Annotated[
        Literal["bm25", "tfidf"], Query(description="Ranking algorithm")
    ] = "bm25",
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    index: Index = Depends(get_index),  # noqa: B008
) -> SearchResponse:
    t0 = time.perf_counter()
    scorer_obj: Scorer = BM25() if scorer == "bm25" else TfIdf()

    fetch_depth = page * k
    all_results = index.search(q, scorer=scorer_obj, k=fetch_depth)
    total = len(all_results)

    start = (page - 1) * k
    page_results = all_results[start : start + k]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    items = [
        SearchItem(
            doc_id=r.doc_id, title=r.title, score=round(r.score, 4), snippet=r.snippet
        )
        for r in page_results
    ]
    return SearchResponse(
        query=q,
        total=total,
        page=page,
        took_ms=round(elapsed_ms, 2),
        results=items,
    )


@app.get("/", response_class=HTMLResponse)
def search_ui(
    request: Request,
    q: str = "",
    page: int = 1,
    k: int = 10,
    index: Index = Depends(get_index),  # noqa: B008
) -> HTMLResponse:
    results: list[dict[str, Any]] = []
    total = 0
    elapsed_ms = 0.0

    if q.strip():
        t0 = time.perf_counter()
        raw_results = index.search(q, k=page * k)
        total = len(raw_results)
        start = (page - 1) * k
        page_results = raw_results[start : start + k]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        for r in page_results:
            results.append(
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "score": r.score,
                    "snippet": r.snippet.replace("**", "<mark>").replace(
                        "</mark><mark>", ""
                    ),
                }
            )

    total_pages = (total + k - 1) // k if total > 0 else 1

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "query": q,
            "results": results,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )


@app.get("/doc/{doc_id}", response_class=HTMLResponse)
def document_view_ui(
    request: Request,
    doc_id: int,
    index: Index = Depends(get_index),  # noqa: B008
) -> HTMLResponse:
    text = index.raw_docs.get(doc_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Document not found")

    title = (
        index.doc_meta[doc_id].title
        if doc_id in index.doc_meta
        else f"Document {doc_id}"
    )

    return templates.TemplateResponse(
        request=request,
        name="doc.html",
        context={"title": title, "doc_id": doc_id, "text": text},
    )
