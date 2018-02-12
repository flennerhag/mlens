"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT
"""

from __future__ import division, print_function, with_statement

import os
import sys
import warnings

import subprocess
from numpy import array

from ..config import get_ivals
from .exceptions import ParallelProcessingError, ParallelProcessingWarning

try:
    import psutil
except ImportError:
    psutil = None

try:
    import cPickle as pickle
except ImportError:
    import pickle

from time import sleep
try:
    # Try get performance counter
    from time import perf_counter as time
except ImportError:
    # Fall back on wall clock
    from time import time


###############################################################################
def pickled(name):
    """Filetype enforcer"""
    if not name.endswith('.pkl'):
        name = '.'.join([name, 'pkl'])
    return name


def pickle_save(obj, name):
    """Utility function for pickling an object"""
    with open(pickled(name), 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(name):
    """Utility function for loading pickled object"""
    with open(pickled(name), 'rb') as f:
        return pickle.load(f)


def load(file, enforce_filetype=True):
    """Utility exception handler for loading file"""
    s, lim = get_ivals()
    if enforce_filetype:
        file = pickled(file)
    try:
        return pickle_load(file)
    except (EOFError, OSError, IOError) as exc:
        msg = str(exc)
        warnings.warn(
            "Could not load transformer at %s. Will check every %.1f seconds "
            "for %i seconds before aborting. " % (file, s, lim),
            ParallelProcessingWarning)

        ts = time()
        while not os.path.exists(file):
            sleep(s)
            if time() - ts > lim:
                raise ParallelProcessingError(
                    "Could not load transformer at %s\nDetails:\n%r" %
                    (dir, msg))

        return pickle_load(file)


###############################################################################
def clone_attribute(iterable, attribute):
    """clone parameters"""
    return [(j.name, j.estimator)
            for i in iterable
            for j in getattr(i, attribute)]


def kwarg_parser(func, kwargs):
    """Utility function for parsing keyword arguments"""
    func_kwargs = dict()
    args = func.__code__.co_varnames
    for arg in args:
        if arg in kwargs:
            func_kwargs[arg] = kwargs.pop(arg)
    return func_kwargs, kwargs


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

    safe_print(message + '%02d:%02d:%02d' % (h, m, s), **kwargs)


class CMLog(object):

    """CPU and Memory logger.

    Class for starting a monitor job of CPU and memory utilization in the
    background in a Python script. The ``monitor`` class records the
    ``cpu_percent``, ``rss`` and ``vms`` as collected by the
    psutil_ library for the parent process' pid.

    CPU usage and memory utilization are stored as attributes in numpy arrays.

    .. _psutil: https://pypi.python.org/pypi/psutil

    Examples
    --------
    >>> from time import sleep
    >>> from mlens.utils.utils import CMLog
    >>> cm = CMLog(verbose=True)
    >>> cm.monitor(2, 0.5)
    >>> _ = [i for i in range(10000000)]
    >>>
    >>> # Collecting before completion triggers a message but no error
    >>> cm._collect()
    >>>
    >>> sleep(2)
    >>> cm._collect()
    >>> print('CPU usage:')
    >>> cm.cpu
    [CMLog] Monitoring for 2 seconds with checks every 0.5 seconds.
    [CMLog] Job not finished. Cannot _collect yet.
    [CMLog] Collecting... done. Read 4 lines in 0.000 seconds.
    CPU usage:
    array([ 50. ,  22.4,   6. ,  11.9])

    Raises
    ------
    ImportError :
        Depends on psutil. If not installed, raises ImportError on
        instantiation.

    Parameters
    ----------
    verbose : bool
        whether to notify of job start.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.pid = os.getpid()

        if psutil is None:
            raise ImportError("psutil not installed. Install psutil, for "
                              "example through pip (pip install psutil) "
                              "before initializing CMLog.")

    def monitor(self, stop=None, ival=0.1, kill=True):
        """Start monitoring CPU and memory usage.

        Parameters
        ----------
        stop : float or None (default = None)
            seconds to monitor for. If None, monitors until ``_collect`` is
            called.

        ival : float (default=0.1)
            interval of monitoring.

        kill : bool (default = True)
            whether to kill the monitoring job if ``_collect`` is called before
            timeout (``stop``). If set to False, calling ``_collect`` will
            cause the instance to wait until the job completes.
        """
        if stop is None and not kill:
            raise ValueError("If no time limit is set 'kill' must be enabled.")

        self._t0 = time()
        self._stop = stop
        self._kill = kill

        # Delete previous job data to avoid confusion
        try:
            del self.cpu
            del self.rss
            del self.vms
        except AttributeError:
            pass

        if self.verbose:
            if self._stop is not None:
                safe_print("[CMLog] Monitoring for {} seconds with checks "
                           "every {} seconds.".format(stop, ival))
            else:
                safe_print("[CMLog] Monitoring until collection with checks "
                           "every {} seconds.".format(ival))

        # Initialize subprocess
        self._out = subprocess.Popen(
            [sys.executable, '-c',
             'from mlens.utils.utils import _recorder; '
             '_recorder({}, {}, {})'.format(
                 self.pid, stop, float(ival))], stdout=subprocess.PIPE)
        return

    def collect(self):
        """Collect monitored data.

        Once a monitor job finishes, call ``_collect`` to read the CPU and
        memory usage into python objects in the current process. If called
        before the job finishes, _collect issues a print statement to try
        again later, but no warning or error is raised.
        """
        if not hasattr(self, '_stop'):
            safe_print('No monitoring job initiated: nothing to _collect.')
            return

        if self._stop is None:
            # If no timer, kill process.
            self._out.kill()

            if self.verbose:
                safe_print("[CMLog] Collecting...", end=" ",  flush=True)

        # Check if job is not completed and if so, check whether to kill
        elif time() - self._t0 < self._stop:
            if self._kill:
                if self.verbose:
                    safe_print("[CMLog] Job not finished - killing process "
                               "and collecting...", end=" ", flush=True)
                self._out.kill()

            elif self.verbose:
                # Wait until completion (this is done in the 'communicate'
                # command
                safe_print("[CMLog] Job not finished - waiting "
                           "until completion and collecting...",
                           end=" ",  flush=True)

        # If job done, we just need to _collect
        elif self.verbose:
            safe_print("[CMLog] Collecting...", end=" ",  flush=True)

        t0, i = time(), 0
        cpu, rss, vms = [], [], []

        out = self._out.communicate()
        out = out[0].decode().strip().split('\n')
        for line in out:

            c, r, v = line.split(',')

            cpu.append(float(c.strip()))
            rss.append(int(r.strip()))
            vms.append(int(v.strip()))

            i += 1

        if self.verbose:
            safe_print('done. Read {} lines in '
                       '{:.3f} seconds.'.format(i, time() - t0))

        self.cpu = array(cpu)
        self.vms = array(vms)
        self.rss = array(rss)

        # Clear job data
        del self._t0
        del self._stop
        del self._kill

        self._out.terminate()
        del self._out


def _recorder(pid, stop, ival):
    """Subprocess call function to record cpu and memory."""
    t = t0 = time()

    process = psutil.Process(pid)

    if stop is None:
        while True:
            m = process.memory_info()
            print(psutil.cpu_percent(), ',', m[0], ',', m[1])
            sleep(ival)
            t = time()
    else:
        while t - t0 < stop:
            m = process.memory_info()
            print(psutil.cpu_percent(), ',', m[0], ',', m[1])
            sleep(ival)
            t = time()
