"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT
"""

from __future__ import division, print_function, with_statement

from numpy import array
import subprocess
import sys
import os

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
    from time import perf_counter as _time
except ImportError:
    # Fall back on time for older versions
    from time import time as _time


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

    m, s = divmod(_time() - t0, 60)
    h, m = divmod(m, 60)

    safe_print(message + '%02d:%02d:%02d\n' % (h, m, s), **kwargs)


class CMLog(object):

    """CPU and Memory logger.

    Class for starting a monitor job of CPU and memory utilization in the
    background in a Python script. The ``monitor`` class records the
     ``cpu_percent``, ``rss`` and ``vms`` as collected by the
     :mod:`psutil` library for the parent process' pid.

    CPU usage and memory utlization are stored as attributes in numpy arrays.

     Notes
     -----
     CMLog uses subprocess to start a recording job for the specified amount
     of time. Once issued, the job cannot be aborted without killing the
     parent process.

     Examples
     --------
     >>> from time import sleep
     >>> from mlens.utils.utils import CMLog
     >>> cm = CMLog(verbose=True)
     >>> cm.monitor(2, 0.5)
     >>> _ = [i for i in range(10000000)]
     >>>
     >>> # Collecting before completion triggers a message but no error
     >>> cm.collect()
     >>>
     >>> sleep(2)
     >>> cm.collect()
     >>> print('CPU usage:')
     >>> cm.cpu
     [CMLog] Monitoring for 2 seconds with checks every 0.5 seconds.
     [CMLog] Job not finished. Cannot collect yet.
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

    def monitor(self, stop, ival=0.1):
        """Start monitoring CPU and memory usage.

        Parameters
        ----------
        stop : float
            seconds to monitor for

        ival : float (default=0.1)
            interval of monitoring.

        Notes
        -----
        Monitor
        """
        # Keep track of job
        self._t0 = _time()
        self._stop = stop

        # Delete previous job data to avoid confusion
        try:
            del self.cpu
            del self.rss
            del self.vms
        except AttributeError:
            pass

        if self.verbose:
            print("[CMLog] Monitoring for {} seconds with checks every {} "
                  "seconds.".format(stop, ival))
            sys.stdout.flush()

        # Initialize subprocess
        self._out = \
            subprocess.Popen([sys.executable, '-c',
                              'from mlens.utils.utils import _recorder; '
                              '_recorder(%i, %f, %f)' % (self.pid,
                                                         float(stop),
                                                         float(ival))],
                             stdout=subprocess.PIPE)
        return

    def collect(self):
        """Collect monitored data.

        Once a monitor job finishes, call ``collect`` to read the CPU and
        memory usage into python objects in the current process. If called
        before the job finishes, collect issues a print statement to try
        again later, but no warning or error is raised.
        """
        if not hasattr(self, '_stop'):
            print('No monitoring job initiated: nothing to collect.')
            return

        if _time() - self._t0 < self._stop:
            if self.verbose:
                print("[CMLog] Job not finished. Cannot collect yet.")
                sys.stdout.flush()
            return

        if self.verbose:
            print("[CMLog] Collecting...", end=" ")
            sys.stdout.flush()

        # Get logs by reading stdout stream from the subprocess call.
        # Probably not the best way to do this but it does the trick
        t0, i = _time(), 0
        cpu, rss, vms = [], [], []

        for line in self._out.stdout:

            c, r, v = str(line).split(',')

            c = float(c.split("'")[1].strip())
            r = int(r.strip())
            v = int(v.strip().split("\\")[0])

            cpu.append(c)
            rss.append(r)
            vms.append(v)

            i += 1

        if self.verbose:
            print('done. Read {} lines in '
                  '{:.3f} seconds.'.format(i, _time() - t0))

        self.cpu = array(cpu)
        self.vms = array(vms)
        self.rss = array(rss)

        # Clear job data
        del self._t0
        del self._stop
        del self._out


def _recorder(pid, stop, ival):
    """Subprocess call function to record cpu and memory."""
    t = t0 = _time()

    process = psutil.Process(pid)

    while t - t0 < stop:
        m = process.memory_info()
        print(psutil.cpu_percent(), ',', m[0], ',', m[1])
        sleep(ival)
        t = _time()
