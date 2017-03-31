"""ML-ENSEMBLE

Collection of simpler fixes occasionally necessary for backwards
compatibility.
"""

import os
import stat
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


def onerror(func, path, exc_info):
    """Error handler for ``shutil.rmtree`` to bypass Windows access error.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``

    author : Michael Foord
    copyright : 2004
    licence : BSD

    source: http://www.voidspace.org.uk/downloads/pathutils.py
    """
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise OSError(exc_info)
