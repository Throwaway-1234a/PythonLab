import asyncio

import httpx

from findex.crawler.crawl import canonicalize_url, crawl
from findex.crawler.fetch import fetch
from findex.crawler.robots import RobotsCache


def test_canonicalize_url() -> None:
    raw = "HTTPS://Example.COM:443/Path/To/?b=2&a=1#section"
    expected = "https://example.com/Path/To?a=1&b=2"
    assert canonicalize_url(raw) == expected


def test_fetch_retry_after() -> None:
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200, content=b"Success", headers={"content-type": "text/html"}
        )

    transport = httpx.MockTransport(mock_handler)

    async def _test() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(client, "https://mock.local/rate-limit")
            assert res is not None
            assert res.status_code == 200
            assert attempts == 2

    asyncio.run(_test())


def test_fetch_client_error_no_retry() -> None:
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    async def _test() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(client, "https://mock.local/not-found")
            assert res is None
            assert attempts == 1

    asyncio.run(_test())


def test_robots_cache_disallow() -> None:
    def mock_handler(request: httpx.Request) -> httpx.Response:
        robots_txt = "User-agent: *\nDisallow: /private/\n"
        return httpx.Response(200, text=robots_txt)

    transport = httpx.MockTransport(mock_handler)

    async def _test() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            robots = RobotsCache(client)
            assert await robots.is_allowed("https://example.com/public")
            assert not await robots.is_allowed("https://example.com/private/secret")

    asyncio.run(_test())


def test_crawl_queue_and_max_pages() -> None:
    html_home = (
        '<html><head><title>Home</title></head><body><a href="/p1">P1</a></body></html>'
    )
    html_p1 = (
        '<html><head><title>P1</title></head><body><a href="/p2">P2</a></body></html>'
    )
    html_p2 = "<html><head><title>P2</title></head><body>Content</body></html>"

    def mock_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in ("/", ""):
            return httpx.Response(
                200, text=html_home, headers={"content-type": "text/html"}
            )
        elif path == "/p1":
            return httpx.Response(
                200, text=html_p1, headers={"content-type": "text/html"}
            )
        elif path == "/p2":
            return httpx.Response(
                200, text=html_p2, headers={"content-type": "text/html"}
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    async def _test() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            pages = []
            async for p in crawl(
                seeds=["https://example.com/"],
                max_pages=2,
                concurrency=2,
                allowed_domains={"example.com"},
                client=client,
            ):
                pages.append(p)
            assert len(pages) == 2
            assert pages[0].title == "Home"
            assert pages[1].title == "P1"

    asyncio.run(_test())
