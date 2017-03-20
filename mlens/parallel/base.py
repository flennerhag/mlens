"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Parallel processing core backend.
"""

import numpy as np

from ..utils import check_initialized, check_process_attr
from ..utils.exceptions import ParallelProcessingWarning, \
    ParallelProcessingError

from sys import platform
import os
import tempfile
import gc
import shutil
from subprocess import check_call

from ..externals.joblib import Parallel, dump, load

import warnings


def _load(arr):
    """Load array from file using default settings."""
    try:
        return np.genfromtxt(arr)
    except Exception as e:
        raise IOError("Could not load X from %s, does not "
                      "appear to be valid as a ndarray. "
                      "Details:\n%r" % e)


class Job(object):

    """Container class for holding job data.

    See Also
    --------
    :class:`ParallelProcessing`
    """

    __slots__ = ['X', 'y', 'P', 'dir']

    def __init__(self):
        self.X = None
        self.y = None
        self.P = None
        self.dir = None


class ParallelProcessing(object):

    """Parallel processing engine.

    ``ParallelProcessing`` is an engine for running estimation.

    Parameters
    ----------
    layers : :class:`mlens.ensemble.base.LayerContainer`
        The ``LayerContainer`` instance the engine is attached to.
    """

    __slots__ = ['layers', '_initialized', '_fitted', '_job']

    def __init__(self, layers):
        self.layers = layers
        self._initialized = 0
        self._fitted = 0

    def initialize(self, X, y=None, dir=None):
        """Create a job instance for estimation."""
        self._job = Job()

        # Set up temporary directory
        self._job.dir = tempfile.mkdtemp(dir=dir)

        # Build mmaps for X and y (if specified)
        for name, arr in zip(('X', 'y'), (X, y)):

            if arr is None:
                # Can happen if y is not specified, as during predictions
                continue

            if isinstance(arr, str):
                # Load file from disk. Need to dump if not memmaped for us
                if not arr.split('.')[-1] in ['mmap', 'npy', 'npz']:
                    # Try loading the file assuming a csv-like format
                    arr = _load(arr)

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
            try:
                setattr(self._job, name, load(f, mmap_mode='r'))
            except IndexError:
                # Joblib's 'load' func fails on npy and npz: we use numpy here
                setattr(self._job, name, np.load(f, mmap_mode='r'))

        # Pre-allocate prediction arrays in r+ (to be updated across workers)
        self._job.P = \
            [np.memmap(filename=os.path.join(self._job.dir,
                                             '%s.mmap' % 'layer-%i' % n),
                       shape=(self._job.X.shape[0],
                              self.layers.struct['layer-%i' % n]['n_pred']),
                       dtype=np.float,
                       mode='w+')
             for n in range(1, self.layers.n_layers + 1)]

        self._initialized = 1

        # Release any memory before going into process
        gc.collect()

    def process(self, attr):
        """Fit all layers in the attached :class:`LayerContainer`."""
        # Pre-checks before starting job
        check_initialized(self)
        check_process_attr(self.layers, attr)

        # Use context manager to ensure same parallel job is passed along
        with Parallel(n_jobs=self.layers.n_jobs,
                      temp_folder=self._job.dir,
                      max_nbytes=None,
                      mmap_mode='r+',
                      verbose=self.layers.verbose,
                      backend=self.layers.backend) as parallel:

            for n, name in enumerate(self.layers.layers):
                self._partial_process(parallel, attr, n, name)

        self._fitted = 1

    def _get_preds(self, n=-1, dtype=np.float, order='C'):
        """Return prediction matrix.

        Parameters
        ----------
        n : int (default = -1)
            layer to retrieves as indexed from base 0 (i.e.
            predictions form layer-1 is retrieved as ``n = 0``).
            List slicing is accepted, so ``n = -1`` retreives the final
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

    def _partial_process(self, parallel, attr, n, name):
        """Generic method for processing a :class:`layer` with ``attr``."""

        # Determine whether training data is X or a previous P matrix
        if n == 0:
            # First layer, we use X
            X = self._job.X
        else:
            # Set X to previous layer's prediction matrix P
            X = self._job.P[n - 1]

        # Get function to process and its variables
        f = getattr(self.layers.layers[name], attr)
        vars = f.__func__.__code__.co_varnames

        # Strip variables we don't want to set from _job
        args = [a for a in vars if a not in {'parallel', 'X', 'P', 'self'}]

        # Build argument list
        kwargs = {a: getattr(self._job, a) for a in args if a in
                  self._job.__slots__}

        kwargs['parallel'] = parallel
        if 'X' in vars:
            kwargs['X'] = X
        if 'P' in vars:
            kwargs['P'] = self._job.P[n]

        for attr in ['lim', 'sec']:
            if attr in vars:
                kwargs[attr] = getattr(self.layers, '__%s__' % attr)

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
