"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT
"""

from __future__ import division, print_function, with_statement

import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    from time import perf_counter as time
except ImportError:
    # Fall back on time for older versions
    from time import time


###############################################################################
def pickle_save(obj, name):
    """Utility function for pickling an object"""
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(name):
    """Utility function for loading pickled object"""
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


###############################################################################
def safe_print(*objects, **kwargs):
    """Safe print function for backwards compatibility."""
    # Get stream
    file = kwargs.pop('file', sys.stdout)
    if isinstance(file, str):
        file = getattr(sys, file)

    # Get flush
    flush = kwargs.pop('flush', False)

    # Print
    print(*objects, file=file, **kwargs)

    # Need to flush outside print function for python2 compatibility
    if flush:
        file.flush()


def print_time(t0, message='', **kwargs):
    """Utility function for printing time"""
    if len(message) > 0:
        message += ' | '

    m, s = divmod(time() - t0, 60)
    h, m = divmod(m, 60)

    safe_print(message + '%02d:%02d:%02d\n' % (h, m, s), **kwargs)
