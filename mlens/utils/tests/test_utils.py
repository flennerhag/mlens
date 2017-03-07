"""ML-ENSEMBLE

author: Sebastian Flennerhag
licence: MIT
"""

from __future__ import division, print_function

from mlens.utils import utils
from time import time, sleep

# An object to pickle
d = {'entry1': 'test', 'entry2': 'also_test'}


def test_print_msg():
    """Check that printing timed messages looks as they should."""

    class Logger(object):
        """Temporary class redirect print messages to a python object."""

        def __init__(self):
            self.log = []

        def write(self, msg):
            """Write a printed message to log"""
            self.log.append(msg)

    logger = Logger()

    # Initiate a time interval
    t0 = time()
    sleep(1)

    # Record recorded print_time message
    utils.print_time(t0, message='test', file=logger)

    assert logger.log[0] == 'test | 00:00:01\n'


def test_pickle():
    """Check that pickling a standard object works."""
    utils.pickle_save(d, 'd')
    test = utils.pickle_load('d')

    assert isinstance(d, dict)
    assert test['entry1'] == 'test'
    assert test['entry2'] == 'also_test'
