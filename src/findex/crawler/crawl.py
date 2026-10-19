import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx

from findex.crawler.fetch import fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger("findex.crawler")


@dataclass(slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: float


def canonicalize_url(url: str) -> str:
    p = urlparse(url)
    scheme = p.scheme.lower()

    hostname = p.hostname.lower() if p.hostname else ""
    port = p.port
    if port and (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = hostname
    elif port:
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_params = sorted(parse_qsl(p.query))
    query = urlencode(query_params)
    return urlunparse((scheme, netloc, path, "", query, ""))


class _HTMLContentExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.title: str = ""
        self.links: list[str] = []
        self._text_chunks: list[str] = []
        self._tag_stack: list[str] = []
        self._is_title = False
        self._ignored = {"script", "style", "noscript", "svg", "head"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        self._tag_stack.append(t)
        if t == "title":
            self._is_title = True
        elif t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    full = urljoin(self.base_url, v)
                    self.links.append(canonicalize_url(full))

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self._is_title = False
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._is_title and not self.title:
            self.title = data.strip()
        if any(ign in self._tag_stack for ign in self._ignored):
            return
        cleaned = data.strip()
        if cleaned:
            self._text_chunks.append(cleaned)

    @property
    def clean_text(self) -> str:
        return " ".join(self._text_chunks)


async def crawl(
    seeds: list[str],
    *,
    max_pages: int = 100,
    concurrency: int = 10,
    per_host: int = 2,
    per_host_delay: float = 0.1,
    allowed_domains: set[str] | None = None,
    client: httpx.AsyncClient | None = None,
    log_file: Path | None = None,
) -> AsyncIterator[Page]:
    queue: asyncio.Queue[str] = asyncio.Queue()
    seen: set[str] = set()
    pages_emitted = 0

    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_last_req: dict[str, float] = {}

    for s in seeds:
        canon = canonicalize_url(s)
        seen.add(canon)
        await queue.put(canon)

    owns_client = client is None
    http_client = (
        client
        if client is not None
        else httpx.AsyncClient(timeout=10.0, follow_redirects=True)
    )
    robots = RobotsCache(http_client)

    out_channel: asyncio.Queue[Page | None] = asyncio.Queue()
    stop_event = asyncio.Event()

    async def worker() -> None:
        nonlocal pages_emitted
        while not stop_event.is_set():
            try:
                url = await asyncio.wait_for(queue.get(), timeout=0.2)
            except TimeoutError:
                if queue.empty() and pages_emitted >= max_pages:
                    break
                continue

            host = urlparse(url).netloc
            if host not in host_sems:
                host_sems[host] = asyncio.Semaphore(per_host)

            if not await robots.is_allowed(url):
                queue.task_done()
                continue

            now = time.monotonic()
            last = host_last_req.get(host, 0.0)
            if now - last < per_host_delay:
                await asyncio.sleep(per_host_delay - (now - last))

            async with global_sem, host_sems[host]:
                host_last_req[host] = time.monotonic()
                res = await fetch(http_client, url)

            if log_file:
                status = res.status_code if res else "FAIL"
                with open(log_file, "a", encoding="utf-8") as lf:  # noqa: ASYNC230
                    lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {status} {url}\n")

            if (
                res
                and res.status_code == 200
                and "text/html" in res.headers.get("content-type", "")
            ):
                extractor = _HTMLContentExtractor(base_url=res.url)
                extractor.feed(res.content.decode("utf-8", errors="replace"))

                page = Page(
                    url=res.url,
                    title=extractor.title or res.url,
                    text=extractor.clean_text,
                    fetched_at=time.time(),
                )

                if pages_emitted < max_pages:
                    pages_emitted += 1
                    await out_channel.put(page)
                    if pages_emitted >= max_pages:
                        stop_event.set()

                for link in extractor.links:
                    link_host = urlparse(link).netloc
                    if allowed_domains and link_host not in allowed_domains:
                        continue
                    if link not in seen and not stop_event.is_set():
                        seen.add(link)
                        await queue.put(link)

            queue.task_done()

    async def supervisor() -> None:
        async with asyncio.TaskGroup() as tg:
            for _ in range(concurrency):
                tg.create_task(worker())
        await out_channel.put(None)

    supervisor_task = asyncio.create_task(supervisor())

    try:
        while True:
            item = await out_channel.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        await supervisor_task
        if owns_client:
            await http_client.aclose()
