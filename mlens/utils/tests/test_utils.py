"""ML-ENSEMBLE

author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT
"""

from __future__ import division

import os
import numpy as np
import sysconfig
import subprocess
from mlens import config
from mlens.utils import utils
from mlens.utils.exceptions import ParallelProcessingError

from time import sleep
try:
    from time import perf_counter as time
except ImportError:
    from time import time

try:
    from contextlib import redirect_stdout, redirect_stderr
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stdout
    redirect_stderr = redirect_stdout

try:
    import psutil
except ImportError:
    psutil = None

__version__ = sysconfig.get_python_version()
# An object to pickle
d = {'entry1': 'test', 'entry2': 'also_test'}


class Logger(object):
    """Temporary class redirect print messages to a python object."""

    def __init__(self):
        self.log = []

    def write(self, msg):
        """Write a printed message to log"""
        self.log.append(msg)


def test_print_time():
    """[Utils] print_time: messages looks as expected."""

    logger = Logger()

    # Initiate a time interval
    t0 = time()
    sleep(1.3)

    # Record recorded print_time message
    utils.print_time(t0, message='test', file=logger)

    assert logger.log[0][:15] == 'test | 00:00:01'


def test_safe_print():
    """[Utils] safe_print: prints correctly."""
    l = Logger()
    utils.safe_print('test', file=l)
    assert l.log[0] == 'test'


def test_safe_print_string():
    """[Utils] safe_print: accepts flush and stream name as string."""

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        utils.safe_print('test', flush=True, file="stdout")


def test_recorder():
    """[Utils] _recorder: test subprocess recording function."""
    if psutil is not None and not __version__.startswith('2.'):
        l = Logger()
        pid = os.getpid()
        with redirect_stdout(l):
            utils._recorder(pid, 0.2, 0.1)

        entries = ''.join(l.log).split('\n')
        if entries[-1] == '':
            entries = entries[:-1]

        assert len(entries) == 2
        assert len(entries[0].split(',')) == 3


def test_cm():
    """[Utils] CMLog: test logging."""
    if psutil is not None and not __version__.startswith('2.') :
        cm = utils.CMLog(verbose=True)

        with open(os.devnull, 'w') as f, redirect_stdout(f):
            cm.monitor(0.3)

            while not hasattr(cm, 'cpu'):
                sleep(0.3)
                cm.collect()

        assert len(cm.cpu) == 3
        assert len(cm.rss) == 3
        assert len(cm.vms) == 3

        # Check that it overwrites
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            cm.monitor(0.2)

            while not hasattr(cm, 'cpu'):
                sleep(0.2)
                cm.collect()

        assert len(cm.cpu) == 2
        assert len(cm.rss) == 2
        assert len(cm.vms) == 2


def test_cm_exception():
    """[Utils] CMLog: test collecting un-monitored returns None."""
    if psutil is not None and not __version__.startswith('2.'):
        cm = utils.CMLog(verbose=False)
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            out = cm.collect()
        assert out is None


def test_pickle():
    """[Utils] Check that pickling a standard object works."""
    utils.pickle_save(d, 'd')
    test = utils.pickle_load('d')

    subprocess.check_call(['rm', 'd.pkl'])

    assert isinstance(d, dict)
    assert test['entry1'] == 'test'
    assert test['entry2'] == 'also_test'


def test_load():
    """[Utils] Check that load handles exceptions gracefully"""

    config.set_ivals(0.1, 0.1)

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        np.testing.assert_raises(
            ParallelProcessingError,
            utils.load, os.path.join(os.getcwd(), 'nonexist'))
