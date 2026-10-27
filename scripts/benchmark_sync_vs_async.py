import asyncio
import time
from pathlib import Path

import httpx
from fastapi import FastAPI

from findex.index import Index

app_test = FastAPI()
idx = Index.load("index.bin") if Path("index.bin").exists() else Index()


@app_test.get("/sync-search")
def sync_search(q: str = "whale"):
    return idx.search(q, k=10)


@app_test.get("/async-search")
async def async_search(q: str = "whale"):
    return idx.search(q, k=10)


async def test_concurrent(route: str, concurrency: int = 20) -> float:
    transport = httpx.ASGITransport(app=app_test)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.perf_counter()
        async with asyncio.TaskGroup() as tg:
            for _ in range(concurrency):
                tg.create_task(client.get(route))
        return time.perf_counter() - t0


def main() -> None:
    print("Benchmarking 20 Concurrent Requests (def vs. async def):")
    t_sync = asyncio.run(test_concurrent("/sync-search", 20))
    t_async = asyncio.run(test_concurrent("/async-search", 20))
    print(f"  Sync (def) under thread pool:       {t_sync:.4f} s")
    print(f"  Async (async def) blocking loop:    {t_async:.4f} s")


if __name__ == "__main__":
    main()
