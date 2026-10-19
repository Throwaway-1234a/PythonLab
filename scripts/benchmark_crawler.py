import asyncio
import resource
import time

import httpx

from findex.crawler.crawl import crawl

TOTAL_PAGES = 200


def get_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def create_mock_site(num_pages: int = TOTAL_PAGES) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")

        idx = int(path[1:]) if path[1:].isdigit() else 0
        next_links = "".join(
            f'<a href="/{idx + i}">P{idx + i}</a>'
            for i in range(1, 4)
            if idx + i <= num_pages
        )
        body = f"<html><head><title>Page {idx}</title></head><body><h1>Content {idx}</h1>{next_links}</body></html>"
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


async def run_benchmark(concurrency: int) -> tuple[float, float, int, float]:
    transport = create_mock_site(TOTAL_PAGES)
    errors = 0
    pages = 0

    async with httpx.AsyncClient(transport=transport) as client:
        t0 = time.perf_counter()
        async for _ in crawl(
            seeds=["https://mock.site/0"],
            max_pages=TOTAL_PAGES,
            concurrency=concurrency,
            per_host=concurrency,
            per_host_delay=0.0,
            allowed_domains={"mock.site"},
            client=client,
        ):
            pages += 1
        wall = time.perf_counter() - t0

    return wall, pages / max(wall, 0.001), errors, get_peak_rss_mb()


def main() -> None:
    print("Benchmarking Crawler Concurrency (200 Pages, Simulated 30ms Latency):\n")
    print(
        f"| {'Concurrency':12} | {'Pages':7} | {'Wall (s)':10} | {'Pages/s':10} | {'Errors':8} | {'Peak RSS':10} |"
    )
    print(
        f"|:{'-' * 12}-|-{'-' * 7}:|-{'-' * 10}:|-{'-' * 10}:|-{'-' * 8}:|-{'-' * 10}:|"
    )

    for c in [1, 5, 20]:
        wall, rate, errors, rss = asyncio.run(run_benchmark(c))
        label = f"{c} (sequential)" if c == 1 else str(c)
        print(
            f"| {label:12} | {TOTAL_PAGES:7d} | {wall:9.2f}s | {rate:9.1f} | {errors:8d} | {rss:8.1f} MB |"
        )


if __name__ == "__main__":
    main()
