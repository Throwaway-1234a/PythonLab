import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from findex.crawler.crawl import crawl
from findex.index import Index
from findex.parallel import build_index_parallel
from findex.semantic import SemanticSearcher, build_embeddings, reciprocal_rank_fusion
from findex.vectorized import VectorizedIndex

app = typer.Typer(
    help="findex: full-text search engine CLI",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def setup_logging(verbose_level: int) -> None:
    if verbose_level == 1:
        level = logging.INFO
    elif verbose_level >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level, format="%(levelname)s: %(message)s", stream=sys.stderr
    )


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase verbosity (-v: INFO, -vv: DEBUG)",
        ),
    ] = 0,
) -> None:
    setup_logging(verbose)


@app.command()
def index(
    corpus: Annotated[Path, typer.Argument(help="Directory containing text documents")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output index destination path")
    ] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Record token positions for phrase matching"),
    ] = True,
    workers: Annotated[
        int, typer.Option("--workers", "-w", help="Number of concurrent workers")
    ] = os.cpu_count() or 4,
    executor: Annotated[
        Literal["serial", "threads", "processes"],
        typer.Option("--executor", "-e", help="Execution backend"),
    ] = "processes",
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Limit indexing to first N documents"),
    ] = None,
) -> None:
    if not corpus.exists():
        err_console.print(
            f"[bold red]Error:[/bold red] Corpus directory '{corpus}' does not exist."
        )
        raise typer.Exit(code=1)

    t0 = time.perf_counter()
    try:
        idx, merge_time = build_index_parallel(
            corpus_dir=corpus,
            executor=executor,
            workers=workers,
            record_positions=positions,
        )
    except Exception as err:
        err_console.print(f"[bold red]Worker failure:[/bold red] {err}")
        raise

    idx.save(out)
    elapsed = time.perf_counter() - t0
    err_console.print(
        f"[green]✓[/green] Indexed [bold]{idx.num_docs}[/bold] documents in [yellow]{elapsed:.3f} s[/yellow] "
        f"([cyan]{executor}[/cyan], {workers} workers, merge: {merge_time:.3f} s)."
    )


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(help="Path to index file")],
    query: Annotated[str, typer.Argument(help="Search query expression")],
    k: Annotated[int, typer.Option("--k", help="Maximum results to return")] = 10,
    scorer_type: Annotated[
        Literal["bm25", "tfidf"], typer.Option("--scorer", help="Ranking scorer")
    ] = "bm25",
    mode: Annotated[
        Literal["keyword", "semantic", "hybrid"],
        typer.Option("--mode", "-m", help="Search mode"),
    ] = "keyword",
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit raw JSON lines")
    ] = False,
) -> None:
    if not index_path.exists():
        err_console.print(
            f"[bold red]Error:[/bold red] Index '{index_path}' not found."
        )
        raise typer.Exit(code=1)

    idx = Index.load(index_path)

    if mode == "keyword":
        vec_idx = VectorizedIndex.from_index(idx)
        results = vec_idx.search_vectorized(query, scorer=scorer_type, k=k)
    elif mode == "semantic":
        npy_path = Path("data/embeddings/embeddings.npy")
        meta_path = Path("data/embeddings/chunk_meta.json")
        if not npy_path.exists():
            err_console.print("[bold red]Error:[/bold red] Run 'findex embed' first.")
            raise typer.Exit(code=1)
        searcher = SemanticSearcher(npy_path, meta_path, idx)
        results = searcher.search_semantic(query, top_k=k)
    else:  # hybrid
        npy_path = Path("data/embeddings/embeddings.npy")
        meta_path = Path("data/embeddings/chunk_meta.json")
        if not npy_path.exists():
            err_console.print("[bold red]Error:[/bold red] Run 'findex embed' first.")
            raise typer.Exit(code=1)
        vec_idx = VectorizedIndex.from_index(idx)
        kw_res = vec_idx.search_vectorized(query, scorer=scorer_type, k=k * 2)
        searcher = SemanticSearcher(npy_path, meta_path, idx)
        sem_res = searcher.search_semantic(query, top_k=k * 2)
        results = reciprocal_rank_fusion(kw_res, sem_res, k=60, top_k=k)

    if as_json:
        for res in results:
            payload = {
                "doc_id": res.doc_id,
                "score": round(res.score, 4),
                "title": res.title,
                "snippet": res.snippet,
            }
            sys.stdout.write(json.dumps(payload) + "\n")
        return

    table = Table(title=f"Results for '{query}' (Mode: {mode.upper()})")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Title", style="green")
    table.add_column("Snippet", style="white")

    for rank, res in enumerate(results, start=1):
        table.add_row(str(rank), f"{res.score:.4f}", res.title, res.snippet)

    console.print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(help="Path to serialized index file")],
) -> None:
    if not index_path.exists():
        err_console.print(
            f"[bold red]Error:[/bold red] Index file '{index_path}' not found."
        )
        raise typer.Exit(code=1)

    try:
        idx = Index.load(index_path)
    except Exception as err:  # noqa: BLE001
        err_console.print(f"[bold red]Error loading index:[/bold red] {err}")
        raise typer.Exit(code=1)

    table = Table(title=f"Index Overview ({index_path.name})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Documents", f"{idx.num_docs:,}")
    table.add_row("Distinct Vocabulary", f"{len(idx):,} terms")
    table.add_row("Average Doc Length", f"{idx.avg_doc_length:.1f} tokens")

    console.print(table)


@app.command()
def crawl_cmd(
    seed_url: Annotated[
        str, typer.Argument(help="Initial URL seed to start crawling from")
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output .jsonl destination path")
    ] = Path("data/crawl.jsonl"),
    max_pages: Annotated[
        int, typer.Option("--max-pages", "-m", help="Maximum pages to fetch")
    ] = 100,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", help="Global concurrency limit")
    ] = 10,
    per_host: Annotated[
        int, typer.Option("--per-host", help="Per-host concurrency limit")
    ] = 2,
    log_file: Annotated[
        Path, typer.Option("--log", help="Per-URL log destination")
    ] = Path("crawl.log"),
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    host = urlparse(seed_url).netloc
    allowed_domains = {host} if host else None

    err_console.print(
        f"[bold green]Starting crawl:[/bold green] {seed_url} (max: {max_pages}, workers: {concurrency})"
    )

    t0 = time.perf_counter()
    count = 0

    async def _run() -> None:
        nonlocal count
        with open(out, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            async for page in crawl(
                seeds=[seed_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                allowed_domains=allowed_domains,
                log_file=log_file,
            ):
                count += 1
                record = {
                    "doc_id": page.url,
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "fetched_at": page.fetched_at,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                rate = count / max(time.perf_counter() - t0, 0.001)
                err_console.print(
                    f"  [cyan]{count:4d}/{max_pages}[/cyan] ({rate:.1f} pages/s) [dim]{page.url}[/dim]"
                )

    asyncio.run(_run())
    elapsed = time.perf_counter() - t0
    err_console.print(
        f"[green]✓[/green] Crawl complete: [bold]{count}[/bold] pages in [yellow]{elapsed:.2f} s[/yellow] -> [cyan]{out}[/cyan]"
    )


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", "-h", help="Bind network host")
    ] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Bind network port")] = 8000,
    workers: Annotated[
        int, typer.Option("--workers", "-w", help="Number of uvicorn worker processes")
    ] = 1,
    index_path: Annotated[
        Path, typer.Option("--index", "-i", help="Index file path")
    ] = Path("index.bin"),
) -> None:
    import os

    os.environ["FINDEX_INDEX_PATH"] = str(index_path)
    os.environ["FINDEX_HOST"] = host
    os.environ["FINDEX_PORT"] = str(port)
    os.environ["FINDEX_WORKERS"] = str(workers)

    err_console.print(
        f"[bold green]Serving findex at http://{host}:{port}[/bold green] ({workers} workers)"
    )
    uvicorn.run("findex.web.app:app", host=host, port=port, workers=workers)


@app.command()
def embed(
    index_path: Annotated[Path, typer.Argument(help="Path to serialized index")] = Path(
        "index.bin"
    ),
    output_dir: Annotated[
        Path, typer.Option("--out", "-o", help="Embeddings directory")
    ] = Path("data/embeddings"),
    model: Annotated[
        str, typer.Option("--model", help="Embedding model name")
    ] = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    if not index_path.exists():
        err_console.print(
            f"[bold red]Error:[/bold red] Index '{index_path}' not found."
        )
        raise typer.Exit(code=1)

    idx = Index.load(index_path)
    npy_path, meta_path = build_embeddings(idx, output_dir, model_name=model)
    err_console.print(
        f"[green]✓[/green] Saved embeddings to [cyan]{npy_path}[/cyan] and [cyan]{meta_path}[/cyan]"
    )


if __name__ == "__main__":
    app()
