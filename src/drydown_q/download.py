"""File download shared by the scripts in ``downloads/`` (transfer only, no science).

``fetch`` streams a URL to ``<dest>.part`` and renames it to ``dest`` once the
byte count matches the server's Content-Length. A transfer that breaks off
resumes from the partial file with an HTTP Range request; time-outs, dropped
connections, "too many requests" and server errors are retried after a wait.
Any other HTTP error (e.g. 404, file not on the server) is raised at once.
"""

import logging
import re
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path

log = logging.getLogger(__name__)


def _retryable(error: urllib.error.HTTPError) -> bool:
    return (error.code == HTTPStatus.TOO_MANY_REQUESTS
            or error.code >= HTTPStatus.INTERNAL_SERVER_ERROR)


def list_directory(url: str, timeout_s: float) -> list[str]:
    """File names linked from a web server's directory index page (``href="name"``),
    or the entries of a local folder for a ``file://`` URL."""
    if url.startswith("file://"):
        return sorted(p.name for p in Path(urllib.request.url2pathname(url[len("file://"):])).iterdir())
    with urllib.request.urlopen(url.rstrip("/") + "/", timeout=timeout_s) as response:
        page = response.read().decode("utf-8", errors="replace")
    return sorted({Path(href).name for href in re.findall(r'href="([^"?#]+)"', page)})


def fetch(url: str, dest: Path, timeout_s: float, retries: int, retry_wait_s: float,
          chunk_bytes: int) -> int:
    """Download ``url`` to ``dest``; return its size in bytes.

    A failed attempt is followed by another after ``retry_wait_s`` seconds,
    resuming from the bytes already received; the download is given up after
    ``retries`` consecutive failed attempts that added no bytes (an attempt
    that made progress resets the count, since a slow server may stall many
    times during one large file). ``timeout_s`` is the socket time-out. Raises
    urllib.error.HTTPError for a non-retryable HTTP status and OSError when
    the retries are used up.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    problem, stalled, attempt = None, 0, 0
    while stalled <= retries:
        if attempt:
            log.warning("    %s: %s (%.0f MiB so far); attempt %d without progress of %d, next in %g s",
                        dest.name, problem, (part.stat().st_size if part.exists() else 0) / (1 << 20),
                        stalled, retries, retry_wait_s)
            time.sleep(retry_wait_s)
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        stalled += 1                          # reset below if this attempt adds bytes
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                resumed = getattr(response, "status", None) == HTTPStatus.PARTIAL_CONTENT
                length = response.headers.get("Content-Length")
                expected = (have if resumed else 0) + int(length) if length is not None else None
                with open(part, "ab" if resumed else "wb") as f:
                    while chunk := response.read(chunk_bytes):
                        f.write(chunk)
        except urllib.error.HTTPError as error:
            if error.code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE:
                part.unlink(missing_ok=True)      # partial file no longer matches the server's file
            elif not _retryable(error):
                raise
            problem = f"HTTP {error.code}"
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            problem = f"{type(error).__name__}: {error}"
            continue
        size = part.stat().st_size
        if expected is not None and size != expected:
            problem = f"got {size} of {expected} bytes"
            continue
        part.rename(dest)
        return size
    raise OSError(f"{url}: gave up after {retries} attempts in a row without progress ({problem})")
