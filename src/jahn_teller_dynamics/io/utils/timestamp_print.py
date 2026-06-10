"""Timestamp-prefixed printing for CLI / batch logs."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, TextIO


def print_ts(
    *objects: Any,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """
    Like built-in :func:`print`, but prefix each line with a local wall-clock timestamp
    (ISO-8601 seconds, e.g. ``2026-05-06T14:32:01``).

    Args:
        objects: Values to print (``str`` conversion, joined with ``sep``).
        sep: Joiner between values.
        end: Suffix after the message.
        file: Stream (default: ``sys.stdout``).
        flush: Pass through to the underlying print.
    """
    stream = sys.stdout if file is None else file
    stamp = datetime.now().isoformat(timespec="seconds")
    body = sep.join(str(o) for o in objects)
    print(f"[{stamp}] {body}", end=end, file=stream, flush=flush)
