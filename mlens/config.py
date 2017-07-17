"""
Global configurations.
"""
from __future__ import print_function

import os
import sys
import numpy
import tempfile
import sysconfig


###############################################################################
# Variables

DTYPE = getattr(numpy, os.environ.get('MLENS_DTYPE', 'float32'))
TMPDIR = os.environ.get('MLENS_TMPDIR', tempfile.gettempdir())
BACKEND = os.environ.get('MLENS_BACKEND', 'multiprocessing')
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
    if method == '':
        # Else set default depending on platform and system
        new_python = _PY_VERSION >= 3.4
        win = \
            sys.platform.startswith('win') or sys.platform.startswith('cygwin')

        if new_python:
            # Use forkserver for unix and spawn for windows
            # Travis currently stalling on OSX, use 'spawn' until investigated
#            method = 'forkserver' if not win else 'spawn'
            method = 'spawn'
        else:
            # Use fork (multiprocessing default)
            method = 'fork'
    return method


###############################################################################
# Set up

START_METHOD = __get_default_start_method(START_METHOD)
set_start_method(START_METHOD)
