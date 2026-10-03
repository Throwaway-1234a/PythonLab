import functools
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from findex.index import Index

log = logging.getLogger("findex")


def timed[F: Callable[..., Any]](fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.info("%s executed in %.3f ms", fn.__qualname__, elapsed_ms)

    return cast(F, wrapper)


@contextmanager
def open_index(path: Path) -> Generator["Index", None, None]:
    from findex.index import Index

    idx = Index.load(path)
    try:
        yield idx
    finally:
        pass
