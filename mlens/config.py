"""
Global configurations.
"""
from __future__ import print_function

import os
import sys
import numpy
import shutil
import tempfile
import warnings
import sysconfig
import subprocess
from multiprocessing import current_process

###############################################################################
# Variables

DTYPE = getattr(numpy, os.environ.get('MLENS_DTYPE', 'float32'))
TMPDIR = os.environ.get('MLENS_TMPDIR', tempfile.gettempdir())
PREFIX = os.environ.get('MLENS_PREFIX', ".mlens_tmp_cache_")
BACKEND = os.environ.get('MLENS_BACKEND', 'threading')
START_METHOD = os.environ.get('MLENS_START_METHOD', '')

_PY_VERSION = float(sysconfig._PY_VERSION_SHORT)


###############################################################################
# Configuration calls


def set_tmpdir(tmp):
    """Set the root directory for temporary caches during estimation.

    Parameters
    ----------
    tmp : str
        directory path
    """
    global TMPDIR
    TMPDIR = tmp


def set_prefix(prefix):
    """Set the prefix assigned to temporary directories during estimation.

    Parameters
    ----------
    prefix : str
        cache file name prefix
    """
    global PREFIX
    PREFIX = prefix


def set_dtype(dtype):
    """Set the  dtype to use during estimation.

    Parameters
    ----------
    dtype : object
        numpy dtype
    """
    global DTYPE
    DTYPE = dtype


def set_backend(backend):
    """Set the parallel backend to use during estimation.

    Parameters
    ----------
    backend : str
        backend type, one of 'multiprocessing', 'threading', 'sequential'
    """
    global BACKEND
    BACKEND = backend


def set_start_method(method):
    """Set the method for starting multiprocess worker pool.

    Parameters
    ----------
    method : str
        Methods available: 'fork', 'spawn', 'forkserver'.
    """
    global START_METHOD
    START_METHOD = method
    os.environ['JOBLIB_START_METHOD'] = START_METHOD


def __get_default_start_method(method):
    """Determine default backend."""
    # Check for environmental variables
    win = sys.platform.startswith('win') or sys.platform.startswith('cygwin')
    if method == '':
        method = 'fork' if not win else 'spawn'
    return method

###############################################################################
# Handlers


def clear_cache(tmp):
    """ Check that cache directory is empty.

    Checks that a specified directory do not contain any directories with
    the ML-Ensemble temporary cache signature. Attempts to remove any found
    directories.

    Parameters
    ----------
    tmp : str
        the directory to check for residual caches in.
    """
    global PREFIX
    residuals = [i for i in os.walk(tmp)
                 if os.path.split(i[0])[-1].startswith(PREFIX)]

    n = len(residuals)
    if n > 0:
        print("[MLENS] Found %i residual cache(s):" % n, file=sys.stderr)

        size = 0
        for i, res in enumerate(residuals):
            s = os.path.getsize(res[0])
            size += s

            print("        %i (%i): %s" % (i + 1, s, res[0]), file=sys.stderr)

        print("        Total size: %i\n[MLENS] Removing..." % size,
              end=" ", file=sys.stderr)

        for res in residuals:
            try:
                shutil.rmtree(res[0])
            except OSError:
                try:
                    subprocess.Popen('rmdir /S /Q %s' % res[0],
                                     shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                except OSError:
                    warnings.warn("Failed to delete cache at %s." % res[0])
        print("done.", file=sys.stderr)

###############################################################################
# Set up


def print_settings():
    """Print package settings on system."""
    if BACKEND == 'threading':
        msg = "[MLENS] backend: %s"
        arg = BACKEND,
    else:
        msg = "[MLENS] backend: %s | start method: %s"
        arg = (BACKEND, START_METHOD)

    print(msg % arg, file=sys.stderr)


if current_process().name == 'MainProcess':
    START_METHOD = __get_default_start_method(START_METHOD)
    set_start_method(START_METHOD)

    print_settings()

    clear_cache(TMPDIR)
