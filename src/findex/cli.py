import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from findex.corpus import iter_documents
from findex.index import Index
from findex.ranking import BM25, Scorer, TfIdf

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

    files = list(corpus.rglob("*.txt"))
    if not files:
        err_console.print(
            f"[bold red]Error:[/bold red] No .txt files found in '{corpus}'."
        )
        raise typer.Exit(code=1)

    if limit is not None:
        files = files[:limit]

    idx = Index()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=err_console,
    ) as progress:
        task = progress.add_task("Indexing documents...", total=len(files))

        from findex.index import DocMeta, Posting
        from findex.tokenize import tokenize

        for doc in iter_documents(corpus):
            if limit is not None and len(idx.doc_meta) >= limit:
                break
            doc_id = len(idx.doc_meta)
            title = doc.doc_id.replace("-", " ").replace("_", " ").title()
            idx.doc_meta[doc_id] = DocMeta(
                doc_id=doc_id, title=title, path=str(doc.path)
            )
            idx.raw_docs[doc_id] = doc.text

            term_positions: dict[str, list[int]] = {}
            count = 0
            for pos, tok in enumerate(tokenize(doc.text)):
                term_positions.setdefault(tok, []).append(pos)
                count += 1

            idx.doc_lengths[doc_id] = count
            for term, pos_list in term_positions.items():
                if positions:
                    idx.postings[term].append(
                        Posting(
                            doc_id=doc_id, tf=len(pos_list), positions=tuple(pos_list)
                        )
                    )
                else:
                    idx.postings[term].append(Posting(doc_id=doc_id, tf=len(pos_list)))

            progress.advance(task)

    idx.save(out)
    err_console.print(
        f"[green]✓[/green] Successfully indexed [bold]{idx.num_docs}[/bold] documents into [cyan]{out}[/cyan]."
    )


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(help="Path to serialized index file")],
    query: Annotated[str, typer.Argument(help="Search query expression")],
    k: Annotated[int, typer.Option("--k", help="Maximum results to return")] = 10,
    scorer_type: Annotated[
        Literal["bm25", "tfidf"], typer.Option("--scorer", help="Ranking scorer")
    ] = "bm25",
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit raw JSON lines to stdout for piping")
    ] = False,
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

    scorer: Scorer = BM25() if scorer_type == "bm25" else TfIdf()
    results = idx.search(query, scorer=scorer, k=k)

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

    if not results:
        console.print(f"[yellow]No documents matched:[/yellow] '{query}'")
        return

    table = Table(title=f"Search Results: '{query}' (Scorer: {scorer_type.upper()})")
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
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


if __name__ == "__main__":
    app()
