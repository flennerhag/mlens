"""ML-Ensemble

:author: Sebastian Flennerhag
:license: MIT
:copyright: 2017-2018

Global backend configurations.

Variables

1. ``DTYPE``: data type of prediction arrays. Must be a numpy dtype.
   Default is ``float32``.

2. ``TMPDIR``: path to directory where temprorary caches will be hosted.
   Default is to use system ``tmp`` structure.

3. ``PREFIX``: cache prefix. Default is ``'.mlens_tmp_cache_'``

4. ``BACKEND``: global default backend. Default is ``'threading'``

5. ``START_METHOD``: global start method (if ``backend='multiprocessing'``)
   Default is ``'fork'``

6. ``VERBOSE``: verbose import. Set to ``Y`` for verbose. Needs to be
   set before import (i.e. ``export MLENS_VERBOSE=0``).

7. ``IVALS``: load exception handling interval. Default is ``(0.01, 120)``.

Environmental variables can be set by ::

    export MLENS_[VARIABLE]=VALUE

For changing defaults during a session, use
``set_[variable]`` and ``get_[variable]``, where ``[variable]`` is replaced
with the lower case name of the environmental variable to change.

Changing global configurations in-session is experimental: Please report any
unexpected behavior.
"""
# pylint: disable=protected-access
# pylint: disable=global-statement
# pylint: disable=not-callable

from __future__ import print_function

import os
import sys
import shutil
import tempfile
import warnings
import sysconfig
import subprocess
from multiprocessing import current_process

import numpy

###############################################################################
# Variables

_DTYPE = getattr(numpy, os.environ.get('MLENS_DTYPE', 'float32'))
_TMPDIR = os.environ.get('MLENS_TMPDIR', tempfile.gettempdir())
_PREFIX = os.environ.get('MLENS_PREFIX', ".mlens_tmp_cache_")
_BACKEND = os.environ.get('MLENS_BACKEND', 'threading')
_START_METHOD = os.environ.get('MLENS_START_METHOD', '')
_VERBOSE = os.environ.get('MLENS_VERBOSE', 'Y')

_IVALS = os.environ.get('MLENS_IVALS', '0.01_120').split('_')
_IVALS = (float(_IVALS[0]), float(_IVALS[1]))

_PY_VERSION = float(sysconfig._PY_VERSION_SHORT)


###############################################################################
# dispatcjh configs

def get_ivals():
    """Return _IVALS"""
    return _IVALS


def get_dtype():
    """Return dtype"""
    return _DTYPE


def get_prefix():
    """Return cache prefix"""
    return _PREFIX


def get_backend():
    """Return backend"""
    return _BACKEND


def get_start_method():
    """Return start method"""
    return _START_METHOD


def get_tmpdir():
    """Return start method"""
    return _TMPDIR

###############################################################################
# Configuration calls


def set_tmpdir(tmp):
    """Set the root directory for temporary caches during estimation.

    Parameters
    ----------
    tmp : str
        directory path
    """
    global _TMPDIR
    _TMPDIR = tmp


def set_prefix(prefix):
    """Set the prefix assigned to temporary directories during estimation.

    Parameters
    ----------
    prefix : str
        cache file name prefix
    """
    global _PREFIX
    _PREFIX = prefix


def set_dtype(dtype):
    """Set the  dtype to use during estimation.

    Parameters
    ----------
    dtype : object
        numpy dtype
    """
    global _DTYPE
    _DTYPE = dtype


def set_backend(backend):
    """Set the parallel backend to use during estimation.

    Parameters
    ----------
    backend : str
        backend type, one of 'multiprocessing', 'threading', 'sequential'
    """
    global _BACKEND
    _BACKEND = backend


def set_start_method(method):
    """Set the method for starting multiprocess worker pool.

    Parameters
    ----------
    method : str
        Methods available: 'fork', 'spawn', 'forkserver'.
    """
    global _START_METHOD
    _START_METHOD = method
    os.environ['JOBLIB_START_METHOD'] = _START_METHOD


def set_ivals(interval, limit):
    """Set the parallel backend to use during estimation.

    Parameters
    ----------
    interval : int
        number of seconds between each check

    limit : int
        number of seconds to wait.
    """
    global _IVALS
    _IVALS = (interval, limit)


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
    global _PREFIX
    residuals = [i for i in os.walk(tmp)
                 if os.path.split(i[0])[-1].startswith(_PREFIX)]

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
    if _VERBOSE != 'Y':
        return
    if _BACKEND == 'threading':
        msg = "[MLENS] backend: %s"
        arg = _BACKEND,
    else:
        msg = "[MLENS] backend: %s | start method: %s"
        arg = (_BACKEND, _START_METHOD)

    print(msg % arg, file=sys.stderr)


if current_process().name == 'MainProcess':
    _START_METHOD = __get_default_start_method(_START_METHOD)
    set_start_method(_START_METHOD)

    print_settings()

    clear_cache(_TMPDIR)
