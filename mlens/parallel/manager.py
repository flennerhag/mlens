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

JOBS = ['predict', 'fit', 'transform']


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


###############################################################################
class Job(object):

    """Container class for holding job data.

    See Also
    --------
    :class:`ParallelProcessing`, :class:`ParallelEvaluation`
    """

    __slots__ = ['y', 'P', 'dir', 'l', 'j', 'tmp']

    def __init__(self, job):
        self.j = job
        self.y = None
        self.P = None
        self.l = None
        self.tmp = None
        self.dir = None


###############################################################################
class ParallelProcessing(object):

    """Parallel processing engine.

    Engine for running ensemble estimation.

    Parameters
    ----------
    layers : :class:`mlens.ensemble.base.LayerContainer`
        The ``LayerContainer`` that instantiated the processor.
    """

    __slots__ = ['layers', '__initialized__', '__fitted__', 'job']

    def __init__(self, layers):
        self.layers = layers
        self.__initialized__ = 0

    def initialize(self, job, X, y=None, dir=None):
        """Create a job instance for estimation."""
        self._check_job(job)
        self.job = Job(job)

        threading = self.layers.backend == 'threading'

        if dir is None:
            dir = config.TMPDIR
        try:
            # Fails on python 2
            self.job.tmp = \
                tempfile.TemporaryDirectory(prefix='mlens_', dir=dir)
            self.job.dir = self.job.tmp.name
        except Exception:
            self.job.dir = tempfile.mkdtemp(prefix='mlens_', dir=dir)

        # --- Prepare inputs
        for name, arr in zip(('X', 'y'), (X, y)):

            if arr is None:
                # e.g predict X
                continue

            # For threading, don't memmap
            if threading:
                f = None
                if isinstance(arr, str):
                    arr = _load(arr)
            else:
                f = dump_array(arr, name, self.job.dir)

            # Store data for processing
            if name is 'y' and y is not None:
                self.job.y = arr if threading else _load_mmap(f)
            else:
                # Store X as the first input matrix in list of inputs matrices
                self.job.P = [arr if threading else _load_mmap(f)]

        # Append pre-allocated prediction arrays in r+ to the P list
        # Each layer will be fitted on P[i] and write to P[i + 1]
        for n, (name, lyr) in enumerate(self.layers.layers.items()):

            # Pre-check indexer
            lyr.indexer.fit(self.job.P[n], y, self.job.j)

            self._gen_prediction_array(lyr, name, threading)

        self.__initialized__ = 1
        gc.collect()

    def _propagate_features(self, lyr):
        """Propagate features from input array to output array."""
        P_out, P_in = self.job.P[-1], self.job.P[-2]

        # Check for loss of obs between layers (i.e. blend)
        n_in, n_out = P_in.shape[0], P_out.shape[0]
        r = int(n_in - n_out)

        # Propagate features as the n first features of the outgoing array
        P_out[:, :lyr.n_feature_prop] = P_in[r:, lyr.propagate_features]

    def _gen_prediction_array(self, lyr, name, threading):
        """Generate prediction array either in-memory or persist to disk."""
        shape = self._get_lyr_sample_size(lyr)
        if threading:
            self.job.P.append(np.empty(shape, dtype=lyr.dtype))
        else:
            f = os.path.join(self.job.dir, '%s.mmap' % name)
            try:
                self.job.P.append(np.memmap(filename=f,
                                            dtype=lyr.dtype,
                                            mode='w+',
                                            shape=shape))
            except Exception as exc:
                raise OSError("Cannot create prediction matrix of shape ("
                              "%i, %i), size %i MBs, for %s.\n Details:\n%r" %
                              (shape[0], shape[1],
                               8 * shape[0] * shape[1] / (1024 ** 2),
                               name, exc))

        # If asked, propagate features
        if lyr.propagate_features is not None:
            self._propagate_features(lyr)

    def _get_lyr_sample_size(self, lyr):
        """Decide what sample size to create P with based on the job type."""
        s0 = lyr.indexer.n_test_samples if self.job.j != 'predict' else \
            lyr.indexer.n_samples

        # Number of prediction columns depends on:
        # 1. number of estimators in layer
        # 2. if ``predict_proba``, number of classes in training set
        # 3. number of subsets (default is one for all data)
        # 4. number of features to propagate
        # Note that 1., 3. and 4. are params but 2. is data dependent
        s1 = lyr.n_pred

        if lyr.proba:
            if self.job.j == 'fit':
                lyr.classes_ = self.job.l = np.unique(self.job.y).shape[0]

            s1 *= lyr.classes_

        if lyr.propagate_features is not None:
            s1 += lyr.n_feature_prop

        return s0, s1

    @staticmethod
    def _check_job(job):
        """Check that a valid job is initialized."""
        if job not in JOBS:
            raise NotImplementedError('The job %s is not valid input '
                                      'for the ParallelProcessing job '
                                      'manager. Accepted jobs: %r.'
                                      % (job, list(JOBS)))

    def process(self):
        """Fit all layers in the attached :class:`LayerContainer`."""
        check_initialized(self)

        # Use context manager to ensure same parallel job during entire process
        with Parallel(n_jobs=self.layers.n_jobs,
                      temp_folder=self.job.dir,
                      max_nbytes=None,
                      mmap_mode='w+',
                      verbose=self.layers.verbose,
                      backend=self.layers.backend) as parallel:

            for n, lyr in enumerate(self.layers.layers.values()):
                self._partial_process(n, lyr, parallel)

    def get_preds(self, n=-1, dtype=None, order='C'):
        """Return prediction matrix.

        Parameters
        ----------
        n : int (default = -1)
            layer to retrieves as indexed from base 0 (i.e.
            predictions form layer-1 is retrieved as ``n = 0``).
            List slicing is accepted, so ``n = -1`` retrieves the final
            predictions.

        dtype : numpy dtype object, optional
            data type to return

        order : str (default = 'C')
            data order. See :class:`numpy.asarray` for details.
        """
        if not hasattr(self, 'job'):
            raise ParallelProcessingError("Processor has been terminated: "
                                          "cannot retrieve final prediction "
                                          "array as the estimation cache has "
                                          "been removed.")
        if dtype is None:
            dtype = self.layers.layers[self.layers.layer_names[n]].dtype

        return np.asarray(self.job.P[n], dtype=dtype, order=order)

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
                              "uncollectable : %r." % gc.garbage,
                              ParallelProcessingWarning)

            self.__initialized__ = 0

    def _partial_process(self, n, lyr, parallel):
        """Generic method for processing a :class:`layer` with ``attr``."""
        # Fire up the estimation instance
        kwd = lyr.cls_kwargs if lyr.cls_kwargs is not None else {}
        e = ENGINES[lyr.cls](self.job, lyr, n, **kwd)
        e.run(parallel)


###############################################################################
class ParallelEvaluation(object):

    """Parallel cross-validation engine.

    Parameters
    ----------
    evaluator : :class:`Evaluator`
        The ``Evaluator`` that instantiated the processor.
    """

    __slots__ = ['evaluator', '__initialized__', 'job']

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.__initialized__ = 0

    def initialize(self, X, y=None, dir=config.TMPDIR):
        """Create cache and memmap X and y."""
        self.job = Job('evaluate')

        try:
            # Fails on python 2
            self.job.tmp = \
                tempfile.TemporaryDirectory(prefix='mlens_', dir=dir)
            self.job.dir = self.job.tmp.name
        except Exception:
            self.job.dir = tempfile.mkdtemp(prefix='mlens_', dir=dir)

        # Build mmaps for inputs
        for name, arr in zip(('X', 'y'), (X, y)):

            if isinstance(arr, str):
                # Load file from disk. Need to dump if not memmaped already
                if not arr.split('.')[-1] in ['mmap', 'npy', 'npz']:
                    # Try loading the file assuming a csv-like format
                    arr = _load(arr)

            if isinstance(arr, str):
                # If arr remains a string, it's pointing to an mmap file
                f = arr
            else:
                # Dump ndarray on disk
                f = os.path.join(self.job.dir, '%s.mmap' % name)
                if os.path.exists(f):
                    os.unlink(f)
                dump(arr, f)

            # Get memmap in read-only mode (we don't want to corrupt the input)
            if name is 'y':
                self.job.y = _load_mmap(f)
            else:
                self.job.P = _load_mmap(f)

        self.__initialized__ = 1

        # Release any memory before going into process
        gc.collect()

    def process(self, attr):
        """Fit all layers in the attached :class:`LayerContainer`."""
        check_initialized(self)

        # Use context manager to ensure same parallel job during entire process
        with Parallel(n_jobs=self.evaluator.n_jobs,
                      temp_folder=self.job.dir,
                      max_nbytes=None,
                      mmap_mode='w+',
                      verbose=self.evaluator.verbose,
                      backend=self.evaluator.backend) as parallel:

            f = ENGINES['evaluation'](self.evaluator)

            getattr(f, attr)(parallel, self.job.P, self.job.y,
                             self.job.dir)

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
                              "uncollectable : %r." % gc.garbage,
                              ParallelProcessingWarning)

            self.__initialized__ = 0
