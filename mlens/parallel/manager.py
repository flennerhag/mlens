"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Parallel processing job managers.
"""
import gc
import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
from scipy.sparse import issparse, hstack

from . import Blender, Evaluation, SingleRun, Stacker, SubStacker
from .. import config
from ..externals.joblib import Parallel, dump, load
from ..utils import check_initialized
from ..utils.exceptions import (ParallelProcessingError,
                                ParallelProcessingWarning)


ENGINES = {'full': SingleRun,
           'stack': Stacker,
           'blend': Blender,
           'subset': SubStacker,
           'subsemble': SubStacker,
           'evaluation': Evaluation
           }

JOBS = ['predict', 'fit', 'transform', 'evaluate']


###############################################################################
def dump_array(array, name, dir):
    """Dump array for memmapping."""
    # First check if the array is on file
    if isinstance(array, str):
        # Load file from disk. Need to dump if not memmaped already
        if not array.split('.')[-1] in ['mmap', 'npy', 'npz']:
            # Try loading the file assuming a csv-like format
            array = _load(array)

    if isinstance(array, str):
        # If arr remains a string, it's pointing to an mmap file
        f = array
    else:
        # Dump ndarray on disk
        f = os.path.join(dir, '%s.mmap' % name)
        if os.path.exists(f):
            os.unlink(f)
        dump(array, f)
    return f


def _load(arr):
    """Load array from file using default settings."""
    if arr.split('.')[-1] in ['npy', 'npz']:
        return np.load(arr)
    else:
        try:
            return np.genfromtxt(arr)
        except Exception as e:
            raise IOError("Could not load X from %s, does not "
                          "appear to be a valid ndarray. "
                          "Details:\n%r" % (arr, e))


def _load_mmap(f):
    """Load a mmap presumably dumped by joblib, otherwise try numpy."""
    try:
        return load(f, mmap_mode='r')
    except (IndexError, KeyError):
        # Joblib's 'load' func fails on npy and npz: use numpy.load
        return np.load(f, mmap_mode='r')


def _check_job(job):
    """Check that a valid job is initialized."""
    if job not in JOBS:
        raise NotImplementedError('The job %s is not valid input '
                                  'for the ParallelProcessing job '
                                  'manager. Accepted jobs: %r.'
                                  % (job, list(JOBS)))


###############################################################################
class Job(object):

    """Container class for holding job data.

    See Also
    --------
    :class:`ParallelProcessing`, :class:`ParallelEvaluation`
    """

    __slots__ = ['y', 'predict_in', 'predict_out', 'dir', 'job', 'tmp']

    def __init__(self, job):
        _check_job(job)
        self.job = job
        self.y = None
        self.predict_in = None
        self.predict_out = None
        self.tmp = None
        self.dir = None

    def update(self):
        """Shift output array to input array."""
        # Enforce csr on spare matrices
        if issparse(self.predict_out) and not \
                self.predict_out.__class__.__name__.startswith('csr'):
            self.predict_out = self.predict_out.tocsr()

        self.predict_in = self.predict_out


###############################################################################
class BaseProcessor(object):

    """Parallel processing base class.

    Base class for parallel processing engines.
    """
    __slots__ = ['caller', '__initialized__', '__threading__', 'job']

    def __init__(self, caller):
        self.job = None
        self.caller = caller
        self.__initialized__ = 0
        self.__threading__ = self.caller.backend == 'threading'

    def initialize(self, job, X, y=None, dir=None):
        """Create a job instance for estimation."""
        self.job = Job(job)

        if dir is None:
            dir = config.TMPDIR
        try:
            self.job.tmp = \
                tempfile.TemporaryDirectory(prefix=config.PREFIX, dir=dir)
            self.job.dir = self.job.tmp.name
        except AttributeError:
            # Fails on python 2
            self.job.dir = tempfile.mkdtemp(prefix=config.PREFIX, dir=dir)

        # --- Prepare inputs
        for name, arr in zip(('X', 'y'), (X, y)):
            if arr is None:
                continue

            # Dump data in cache
            if self.__threading__:
                # No need to memmap
                f = None
                if isinstance(arr, str):
                    arr = _load(arr)
            else:
                f = dump_array(arr, name, self.job.dir)

            # Store data for processing
            if name is 'y' and y is not None:
                self.job.y = arr if self.__threading__ else _load_mmap(f)
            else:
                self.job.predict_in = arr \
                    if self.__threading__ else _load_mmap(f)

        self.__initialized__ = 1
        gc.collect()

    def terminate(self):
        """Remove temporary folder and all cache data."""
        # Delete all contents from cache
        try:
            self.job.tmp.cleanup()

        except (AttributeError, OSError):
            # Fall back on shutil for python 2, can also fail on windows
            try:
                shutil.rmtree(self.job.dir)

            except OSError:
                # Can fail on windows, need to use the shell
                try:
                    subprocess.Popen('rmdir /S /Q %s' % self.job.dir,
                                     shell=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                except OSError:
                    warnings.warn("Failed to delete cache at %s."
                                  "If created with default settings, will be "
                                  "removed on reboot. For immediate "
                                  "removal, manual removal is required." %
                                  self.job.dir, ParallelProcessingWarning)

        finally:
            # Always release process memory
            del self.job

            gc.collect()

            if not len(gc.garbage) == 0:
                warnings.warn("Clearing process memory failed, "
                              "uncollected:\n%r." % gc.garbage,
                              ParallelProcessingWarning)

            self.__initialized__ = 0


class ParallelProcessing(BaseProcessor):

    """Parallel processing engine.

    Engine for running ensemble estimation.

    Parameters
    ----------
    layers : :class:`mlens.ensemble.base.LayerContainer`
        The ``LayerContainer`` that instantiated the processor.
    """
    def __init__(self, caller):
        super(ParallelProcessing, self).__init__(caller)

    def process(self):
        """Fit all layers in the attached :class:`LayerContainer`."""
        check_initialized(self)

        # Process each layer sequentially with the same worker pool
        with Parallel(n_jobs=self.caller.n_jobs,
                      temp_folder=self.job.dir,
                      max_nbytes=None,
                      mmap_mode='w+',
                      verbose=self.caller.verbose,
                      backend=self.caller.backend) as parallel:

            for name, lyr in self.caller.layers.items():

                # Process layer
                self._partial_process(name, lyr, parallel)

                # Update input array with output array
                self.job.update()

    def _partial_process(self, name, lyr, parallel):
        """Generate prediction matrix for a given :class:`layer`."""
        lyr.indexer.fit(self.job.predict_in, self.job.y, self.job.job)
        self._gen_prediction_array(name, lyr, self.__threading__)

        # Run estimation to populate prediction matrix
        kwd = lyr.cls_kwargs if lyr.cls_kwargs else {}
        engine = ENGINES[lyr.cls](self.job, lyr, **kwd)
        engine(parallel)

        # Propagate features from input to output
        if lyr.propagate_features is not None:
            self._propagate_features(lyr)

    def _propagate_features(self, lyr):
        """Propagate features from input array to output array."""
        p_out, p_in = self.job.predict_out, self.job.predict_in

        # Check for loss of obs between layers (i.e. blend)
        n_in, n_out = p_in.shape[0], p_out.shape[0]
        r = int(n_in - n_out)

        # Propagate features as the n first features of the outgoing array
        if not issparse(p_in):
            # Simple item setting
            p_out[:, :lyr.n_feature_prop] = p_in[r:, lyr.propagate_features]
        else:
            # Need to populate propagated features using scipy sparse hstack
            self.job.predict_out = hstack([p_in[r:, lyr.propagate_features],
                                           p_out[:, lyr.n_feature_prop:]]
                                          ).tolil()

    def _gen_prediction_array(self, name, lyr, threading):
        """Generate prediction array either in-memory or persist to disk."""
        shape = self._get_lyr_sample_size(lyr)
        if threading:
            self.job.predict_out = np.empty(shape, dtype=lyr.dtype)
        else:
            f = os.path.join(self.job.dir, '%s.mmap' % name)
            try:
                self.job.predict_out = np.memmap(filename=f,
                                                 dtype=lyr.dtype,
                                                 mode='w+',
                                                 shape=shape)
            except Exception as exc:
                raise OSError("Cannot create prediction matrix of shape ("
                              "%i, %i), size %i MBs, for %s.\n Details:\n%r" %
                              (shape[0], shape[1],
                               8 * shape[0] * shape[1] / (1024 ** 2),
                               name, exc))

    def _get_lyr_sample_size(self, lyr):
        """Decide what sample size to create P with based on the job type."""
        s0 = lyr.indexer.n_test_samples if self.job.job != 'predict' else \
            lyr.indexer.n_samples

        # Number of prediction columns depends on:
        # 1. number of estimators in layer
        # 2. if ``predict_proba``, number of classes in training set
        # 3. number of subsets (default is one for all data)
        # 4. number of features to propagate
        # Note that 1., 3. and 4. are params but 2. is data dependent
        s1 = lyr.n_pred

        if lyr.proba:
            if self.job.job == 'fit':
                lyr.classes_ = np.unique(self.job.y).shape[0]

            s1 *= lyr.classes_

        if lyr.propagate_features is not None:
            s1 += lyr.n_feature_prop

        return s0, s1

    def get_preds(self, dtype=None, order='C'):
        """Return prediction matrix.

        Parameters
        ----------
        dtype : numpy dtype object, optional
            data type to return

        order : str (default = 'C')
            data order. See :class:`numpy.asarray` for details.
        """
        if not hasattr(self, 'job'):
            raise ParallelProcessingError("Processor has been terminated: "
                                          "cannot retrieve final prediction "
                                          "array from cache.")
        if dtype is None:
            dtype = self.caller.layers[self.caller.layer_names[-1]].dtype

        if issparse(self.job.predict_out):
            return self.job.predict_out
        else:
            return np.asarray(self.job.predict_out, dtype=dtype, order=order)


###############################################################################
class ParallelEvaluation(BaseProcessor):

    """Parallel cross-validation engine.

    Parameters
    ----------
    caller : :class:`Evaluator`
        The ``Evaluator`` that instantiated the processor.
    """

    def __init__(self, caller):
        super(ParallelEvaluation, self).__init__(caller)

    def process(self, attr):
        """Fit all layers in the attached :class:`LayerContainer`."""
        check_initialized(self)

        # Use context manager to ensure same parallel job during entire process
        with Parallel(n_jobs=self.caller.n_jobs,
                      temp_folder=self.job.dir,
                      max_nbytes=None,
                      mmap_mode='w+',
                      verbose=self.caller.verbose,
                      backend=self.caller.backend) as parallel:

            f = ENGINES['evaluation'](self.caller)

            getattr(f, attr)(parallel,
                             self.job.predict_in,
                             self.job.y,
                             self.job.dir)
