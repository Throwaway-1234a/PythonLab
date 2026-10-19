import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT

log = logging.getLogger("findex.crawler")


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self._parsers: dict[str, RobotFileParser] = {}

    async def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc
        if not host:
            return False

        if host not in self._parsers:
            robots_url = f"{parsed.scheme}://{host}/robots.txt"
            parser = RobotFileParser()
            try:
                resp = await self.client.get(
                    robots_url, headers={"User-Agent": USER_AGENT}, timeout=5.0
                )
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    parser.allow_all = True  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                parser.allow_all = True  # type: ignore[attr-defined]
            self._parsers[host] = parser

        parser = self._parsers[host]
        if getattr(parser, "allow_all", False):
            return True
        return parser.can_fetch(USER_AGENT, url)
