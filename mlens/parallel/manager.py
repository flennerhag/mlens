"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Parallel processing job manager.
"""

import numpy as np

from . import Stacker, Blender
from ..utils import check_initialized
from ..utils.exceptions import ParallelProcessingWarning, \
    ParallelProcessingError

from sys import platform
import os
import tempfile
import gc
import shutil
from subprocess import check_call

from joblib import Parallel, dump, load

import warnings


ENGINES = {'stack': Stacker,
           'blend': Blender,
           }

JOBS = {'predict', 'fit', 'predict_proba'}


class Job(object):

    """Container class for holding job data.

    See Also
    --------
    :class:`ParallelProcessing`
    """

    __slots__ = ['X', 'y', 'P', 'dir', 'l']

    def __init__(self):
        self.X = None
        self.y = None
        self.P = None
        self.l = None
        self.dir = None


class ParallelProcessing(object):

    """Parallel processing engine.

    ``ParallelProcessing`` is an engine for running estimation.

    Parameters
    ----------
    layers : :class:`mlens.ensemble.base.LayerContainer`
        The ``LayerContainer`` that instantiated the processor.

    job : str
        The job type to manage. ``job`` must be in ``['predict', 'fit',
        'predict_proba']``, otherwise an error will be raised.
    """

    __slots__ = ['layers', '_initialized', '_fitted', '_job', 'job']

    def __init__(self, layers, job):
        self._check_job(job)
        self.layers = layers
        self.job = job
        self._initialized = 0
        self._fitted = 0

    def initialize(self, X, y=None, dir=None):
        """Create a job instance for estimation."""
        self._job = Job()

        self._job.dir = tempfile.mkdtemp(dir=dir)

        # Build mmaps for inputs
        for name, arr in zip(('X', 'y'), (X, y)):

            if arr is None:
                # Can happen if y is not specified (i.e. during prediction)
                continue

            if isinstance(arr, str):
                # Load file from disk. Need to dump if not memmaped already
                if not arr.split('.')[-1] in ['mmap', 'npy', 'npz']:
                    # Try loading the file assuming a csv-like format
                    arr = self._load(arr)

            if isinstance(arr, str):
                # If arr remains a string, it's pointing to an mmap file
                f = arr
            else:
                # Dump ndarray on disk
                f = os.path.join(self._job.dir, '%s.mmap' % name)
                if os.path.exists(f):
                    os.unlink(f)
                dump(arr, f)

            # Get memmap in read-only mode (we don't want to corrupt the input)
            if name is 'y' and y is not None:
                self._job.y = self._load_mmap(f)
            else:
                # Store X as the first input matrix in list of inputs matrices
                self._job.P = [self._load_mmap(f)]

        # Append pre-allocated prediction arrays in r+ to the P list
        # Each layer will be fitted on P[i] and write to P[i + 1]
        for name, lyr in self.layers.layers.items():
            f = os.path.join(self._job.dir, '%s.mmap' % name)

            # We call the indexers fit method now at initialization - if there
            # is something funky with indexing it is better to catch it now
            # than mid-estimation
            lyr.indexer.fit(self._job.P[0])

            shape = self._get_lyr_sample_size(lyr)

            self._job.P.append(np.memmap(filename=f,
                                         dtype=np.float,
                                         mode='w+',
                                         shape=shape))

        self._initialized = 1

        # Release any memory before going into process
        gc.collect()

    def _get_lyr_sample_size(self, lyr):
        """Decide what sample size to create P with based on the job type."""
        s0 = lyr.indexer.n_test_samples if self.job == 'fit' else \
            lyr.indexer.n_samples

        s1 = lyr.n_pred

        if self.job == 'predict_proba':
            self._job.l = np.unique(self._job.y).shape[0]
            s1 *= self._job.l

        return s0, s1

    @staticmethod
    def _check_job(job):
        """Check that a valid job is initialized."""
        if job not in JOBS:
            raise NotImplementedError('The job %s is not valid for the input '
                                      'for the ParallelProcessing job '
                                      'manager. Accepted jobs: %r.'
                                      % (job, JOBS))
    @staticmethod
    def _load(arr):
        """Load array from file using default settings."""
        try:
            return np.genfromtxt(arr)
        except Exception as e:
            raise IOError("Could not load X from %s, does not "
                          "appear to be valid as a ndarray. "
                          "Details:\n%r" % e)

    @staticmethod
    def _load_mmap(f):
        """Load a mmap presumably dumped by joblib, otherwise try numpy."""
        try:
            return load(f, mmap_mode='r')
        except IndexError:
            # Joblib's 'load' func fails on npy and npz: use numpy.load
            return np.load(f, mmap_mode='r')

    def process(self):
        """Fit all layers in the attached :class:`LayerContainer`."""
        check_initialized(self)

        # Use context manager to ensure same parallel job during entire process
        with Parallel(n_jobs=self.layers.n_jobs,
                      temp_folder=self._job.dir,
                      max_nbytes=None,
                      mmap_mode='r+',
                      verbose=self.layers.verbose,
                      backend=self.layers.backend) as parallel:

            for n, lyr in enumerate(self.layers.layers.values()):
                self._partial_process(n, lyr, parallel, self.job)

        self._fitted = 1

    def _get_preds(self, n=-1, dtype=np.float, order='C'):
        """Return prediction matrix.

        Parameters
        ----------
        n : int (default = -1)
            layer to retrieves as indexed from base 0 (i.e.
            predictions form layer-1 is retrieved as ``n = 0``).
            List slicing is accepted, so ``n = -1`` retrieves the final
            predictions.

        dtype : object (default = numpy.float)
            data type to return

        order : str (default = 'C')
            data order. See :class:`numpy.asarray` for details.
        """
        if not hasattr(self, '_job'):
            raise ParallelProcessingError("Processor has been terminated: "
                                          "cannot retrieve final prediction "
                                          "array as the estimation cache has "
                                          "been removed.")

        return np.asarray(self._job.P[n], dtype=dtype, order=order)

    def terminate(self):
        """Remove temporary folder and all cache data."""

        temp_folder = self._job.dir

        # Delete the job
        del self._job

        # Collect garbage
        gc.collect()

        # Reset initialized flag
        self._initialized = 0

        # Remove temporary folder
        try:
            shutil.rmtree(temp_folder)
        except OSError as e:
            # Can fail on Windows - we try to force remove it using the CLI
            warnings.warn("Failed to remove temporary directory with "
                          "standard procedure. Will try command line "
                          "interface. Details:\n%r" % e,
                          ParallelProcessingWarning)

            if "win" in platform:
                flag = check_call(['rmdir', temp_folder, '/s', '/q'])
            else:
                flag = check_call(['rm', '-rf', temp_folder])

            if flag != 0:
                raise RuntimeError("Could not remove temporary directory.")

    def _partial_process(self, n, lyr, parallel, attr):
        """Generic method for processing a :class:`layer` with ``attr``."""

        # Fire up the estimation instance
        kwd = lyr.cls_kwargs if lyr.cls_kwargs is not None else {}
        kwd['labels'] = self._job.l
        e = ENGINES[lyr.cls](lyr, **kwd)

        # Get function to process and its variables
        f = getattr(e, attr)
        fargs = f.__func__.__code__.co_varnames

        # Strip variables we don't want to set from _job directly
        args = [a for a in fargs if a not in {'parallel', 'X', 'P', 'self'}]

        # Build argument list
        kwargs = {a: getattr(self._job, a) for a in args if a in
                  self._job.__slots__}

        kwargs['parallel'] = parallel
        if 'X' in fargs:
            kwargs['X'] = self._job.P[n]
        if 'P' in fargs:
            kwargs['P'] = self._job.P[n + 1]

        f(**kwargs)

    def _process_layer(self, parallel, attr, n, name):
        """Wrapper around process if a single layer is to be processed.

        Checks is a job has been initialized, and if parallel has not been set,
        wraps call to ``_partial_process`` around a context manager.
        """
        check_initialized(self)

        if parallel is None:
            with Parallel(n_jobs=self.layers.n_jobs,
                          temp_folder=self._job.dir,
                          max_nbytes=None,
                          mmap_mode='r+') as parallel:
                self._partial_process(parallel, attr, n, name)
        else:
            # Assume parallel was already created in a context manager
            self._partial_process(parallel, attr, n, name)
