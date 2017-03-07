"""ML-ENSEMBLE

author: Sebastian Flennerhag
licence: MIT
"""

from __future__ import division, print_function, with_statement

from time import time
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle


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
def safe_print(*objects, sep='', end='\n', file=sys.stdout, flush=False):
    """Safe print function for backwards compatibility."""
    if isinstance(file, str):
        file = getattr(sys, file)
    
    print(*objects, sep=sep, end=end, file=file)
    if flush:
        # Need manual flush for Python 2
        file.flush()
    
    
def print_time(t0, message='', **kwargs):
    """Utility function for printing time"""
    if len(message) > 0:
        message += ' | '

    if 'file' in kwargs and isinstance(kwargs['file'], str):
            kwargs['file'] = getattr(sys, kwargs['file'])

    m, s = divmod(time() - t0, 60)
    h, m = divmod(m, 60)
   
    safe_print(message + '%02d:%02d:%02d\n' % (h, m, s), **kwargs)
