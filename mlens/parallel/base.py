"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Parallel processing core backend.
"""

import numpy as np

from collections import OrderedDict

from ..utils import check_initialized, check_process_attr
from ..utils.exceptions import ParallelProcessingWarning, \
    ParallelProcessingError

import os
import tempfile
import gc
import shutil
from subprocess import check_call

from joblib import Parallel, dump, load

import warnings


def _load(arr):
    """Load array from file using default settings."""
    try:
        return np.genfromtxt(arr)
    except Exception as e:
        raise IOError("Could not load X from %s, does not "
                      "appear to be valid as a ndarray. "
                      "Details:\n%r" % arr)


class ParallelProcessing(object):

    """Parallel processing engine.

    ``ParallelProcessing`` is an engine for running estimation.

    Parameters
    ----------
    layers : :class:`mlens.ensemble.base.LayerContainer`
        The ``LayerContainer`` instance the engine is attached to.
    """

    def __init__(self, layers):
        self.layers = layers
        self._initialized = 0
        self._fitted = 0

    def initialize(self, X, y=None, dir=None):
        """Create a job instance for estimation."""
        self._job = dict()

        # Set up temporary directory
        self._job['temp_folder'] = tempfile.mkdtemp(dir=dir)

        # Build mmaps for X and y (if specified)
        self._job['paths'] = dict()
        for name, arr in zip(('X', 'y'), (X, y)):

            if arr is None:
                continue

            if isinstance(arr, str):
                # If a memmapable file was specified, we're done
                s = arr.split('.')[-1]
                if s in ['mmap', 'npy', 'npz']:
                    self._job['paths'][name] = arr
                    continue
                else:
                    # Try loading the file assuming a csv-like format
                    arr = _load(arr)

            # Dump ndarray on disk and get memmap
            f = os.path.join(self._job['temp_folder'], '%s.mmap' % name)

            if os.path.exists(f):
                os.unlink(f)

            self._job['paths'][name] = f
            dump(arr, f)

        # Replace arrays in memory with memmaps
        for key, val in self._job['paths'].items():
            try:
                self._job[key] = load(val, mmap_mode='r')
            except IndexError:
                # Joblib's 'load' func fails on npy and npz: we use numpy here
                self._job[key] = np.load(val, mmap_mode='r')

        # Pre-allocate prediction arrays and store memmaps in a prediction dict
        self._job['P'] = OrderedDict()

        for n in range(self.layers.n_layers):
            # Do all layers first
            name = 'layer-%i' % (n + 1)

            self._job['paths'][name] = \
                os.path.join(self._job['temp_folder'], '%s.mmap' % name)

            n_pred = self.layers._layer_data['layer-%i' % (n + 1)]['n_pred']

            self._job['P'][name] = \
                np.memmap(filename=self._job['paths'][name],
                          shape=(self._job['X'].shape[0], n_pred),
                          dtype=np.float,
                          mode='w+')

        self._initialized = 1

    def process(self, attr):
        """Fit all layers in the attached :class:`LayerContainer`."""
        # Pre-checks before starting job
        check_initialized(self)
        check_process_attr(self.layers, attr)

        # Use context manager to ensure same parallel job is passed along
        with Parallel(n_jobs=self.layers.n_jobs,
                      temp_folder=self._job['temp_folder'],
                      max_nbytes=None,
                      mmap_mode='r',
                      verbose=self.layers.verbose) as parallel:

            for layer_name in self.layers.layers:
                self._partial_process(parallel, attr, layer_name)

        self._fitted = 1

    def _get_final_preds(self, order='C'):
        """Return final prediction matrix."""
        if not hasattr(self, '_job'):
            raise ParallelProcessingError("Processor has been terminated: "
                                          "cannot retrieve final prediction "
                                          "array as the estimation cache has "
                                          "been removed.")

        final_layer = 'layer-%i' % self.layers.n_layers

        return np.asarray(self._job['P'][final_layer],
                          dtype=np.float, order=order)

    def terminate(self):
        """Remove temporary folder and all cache data."""
        temp_folder = self._job['temp_folder']

        # Release memory
        self._job.clear()
        self.__delattr__('_job')

        # Remove temporary folder
        try:
            shutil.rmtree(temp_folder)
            flag = 0
        except OSError:
            # Can fail on Windows - we try to force remove it using the CLI
            flag = check_call(['rmdir', temp_folder, '/s', '/q'])

            if flag != 0:
                warnings.warn("Could not use 'rmdir' to remove tmp dir "
                              "[Errno: %i]. Will try 'rm'." % flag)
                raise RuntimeError
        except Exception:
            # Try Unix
            flag = check_call(['rm', '-rf', temp_folder])
            if flag != 0:
                warnings.warn("Could not use 'rm' to remove tmp dir "
                              "[Errno: %i]." % flag)
                raise RuntimeError
        else:
            if flag != 0:
                msg = ("Could not remove temporary estimation cache at "
                       "%s.\nFolder will remain in memory until machine "
                       "reboots.")
                warnings.warn(msg % temp_folder, ParallelProcessingWarning)
        finally:
            # Always collect garbage
            gc.collect()

            self._initialized = 0

    def _partial_process(self, parallel, attr, layer_name):
        """Generic method for processing a :class:`layer` with ``attr``."""

        # Determine whether training data is X or a previous P matrix
        splitted_name = layer_name.split('-')
        base = splitted_name[0]
        n = int(splitted_name[-1])

        if n == 1:
            # First layer, we use X
            X = self._job['X']
        else:
            # Set X to previous layer's prediction matrix P
            prev_layer = '%s-%i' % (base, int(n - 1))
            X = self._job['P'][prev_layer]

        # Get function to process and its variables
        f = getattr(self.layers.layers[layer_name], attr)
        varnames = f.__func__.__code__.co_varnames

        # Strip variables we don't want to set from _job
        args = [a for a in varnames if a not in {'parallel', 'X', 'P', 'self'}]

        # Build argument list
        kwargs = {a: self._job[a] for a in args if a in self._job}

        kwargs['parallel'] = parallel
        if 'X' in varnames:
            kwargs['X'] = X
        if 'P' in varnames:
            kwargs['P'] = self._job['P'][layer_name]

        for attr in ['lim', 'sec']:
            if attr in varnames:
                kwargs[attr] = getattr(self.layers, '__%s__' % attr)

        f(**kwargs)

    def _process_layer(self, parallel, attr, layer_name):
        """Wrapper around process if a single layer is to be processed.

        Checks is a job has been initialized, and if parallel has not been set,
        wraps call to ``_partial_process`` around a context manager.
        """
        check_initialized(self)

        if parallel is None:
            with Parallel(n_jobs=self.layers.n_jobs,
                          temp_folder=self._job['temp_folder'],
                          max_nbytes=None,
                          mmap_mode='r') as parallel:
                self._partial_process(parallel, attr, layer_name)
        else:
            # Assume parallel was already created in a context manager
            self._partial_process(parallel, attr, layer_name)
