from __future__ import annotations

import inspect
import os
from types import FrameType

# Files under this directory are internal to wsidata. The trailing os.sep keeps
# sibling distributions from matching the prefix.
_PKG_DIR = os.path.dirname(__file__) + os.sep


def find_stack_level() -> int:
    """Return the ``stacklevel`` of the first caller outside of wsidata.

    Pass it to :func:`warnings.warn` or :func:`logging.warning` so the message is
    attributed to the user's call site instead of an internal frame.
    """
    # inspect.stack() is slow, walk f_back instead.
    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    try:
        n = 0
        while frame is not None and frame.f_code.co_filename.startswith(_PKG_DIR):
            frame = frame.f_back
            n += 1
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    # n is 0 only when currentframe() is unavailable (non-CPython implementations).
    # stacklevel=0 makes logging blame logging/__init__.py, so never return it.
    return max(n, 1)
