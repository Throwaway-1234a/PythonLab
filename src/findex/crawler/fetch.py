import asyncio
import logging
import random
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger("findex.crawler")

USER_AGENT = "findex-crawler/0.1 (+https://github.com/Throwaway-1234a/PythonLab)"


@dataclass(slots=True)
class FetchResult:
    url: str
    status_code: int
    content: bytes
    elapsed: float
    headers: dict[str, str]


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> FetchResult | None:
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, headers=headers, follow_redirects=True)
                elapsed = time.perf_counter() - t0

                if resp.status_code == 200:
                    return FetchResult(
                        url=str(resp.url),
                        status_code=resp.status_code,
                        content=resp.content,
                        elapsed=elapsed,
                        headers=dict(resp.headers),
                    )

                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    if attempt == max_retries:
                        log.warning(
                            "Fetch failed after %d retries [%d]: %s",
                            max_retries,
                            resp.status_code,
                            url,
                        )
                        return None

                    retry_after_raw = resp.headers.get("retry-after")
                    if retry_after_raw and retry_after_raw.isdigit():
                        delay = float(retry_after_raw)
                        log.info(
                            "Honoring Retry-After: waiting %.2f s for %s", delay, url
                        )
                    else:
                        delay = (0.5 * (2**attempt)) + random.uniform(0.0, 0.25)

                    await asyncio.sleep(delay)
                    continue

                if 400 <= resp.status_code < 500:
                    log.debug("Client error [%d], skipping: %s", resp.status_code, url)
                    return None

        except (TimeoutError, httpx.TimeoutException) as err:
            if attempt == max_retries:
                log.warning(
                    "Timeout fetching %s after %d retries: %s", url, max_retries, err
                )
                return None
            delay = (0.5 * (2**attempt)) + random.uniform(0.0, 0.25)
            await asyncio.sleep(delay)

        except httpx.NetworkError as err:
            if attempt == max_retries:
                log.warning("Network error on %s: %s", url, err)
                return None
            delay = (0.5 * (2**attempt)) + random.uniform(0.0, 0.25)
            await asyncio.sleep(delay)

    return None
