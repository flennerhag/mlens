"""ML-ENSEMBLE

Collection of simpler fixes occassionally necessary for backwards
compatibility.
"""

import sys
from contextlib import contextmanager


@contextmanager
def redirect(target, file=sys.stderr):
    """Replication of ``redirect_stderr`` or ``redirect_stdout``.

    :source: http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    """
    original, file = file, target

    try:
        yield target
    finally:
        file = original
