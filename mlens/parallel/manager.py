"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Parallel processing job managers.
"""
import numpy as np

from . import Stacker, Blender, SingleRun, SubStacker, Evaluation
from ..utils import check_initialized
from ..utils.exceptions import ParallelProcessingError, \
    ParallelProcessingWarning

import os
import gc
import shutil
import tempfile
import warnings
import subprocess
from joblib import Parallel, dump, load


ENGINES = {'full': SingleRun,
           'stack': Stacker,
           'blend': Blender,
           'subset': SubStacker,
           'evaluation': Evaluation
           }

JOBS = ['predict', 'fit', 'transform']


###############################################################################
def _load(arr):
    """Load array from file using default settings."""
    try:
        return np.genfromtxt(arr)
    except Exception as e:
        raise IOError("Could not load X from %s, does not "
                      "appear to be a valid ndarray. "
                      "Details:\n%r" % e)


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

    __slots__ = ['layers', '_initialized', '_fitted', '_job']

    def __init__(self, layers):
        self.layers = layers
        self._initialized = 0
        self._fitted = 0

    def initialize(self, job, X, y=None, dir=None):
        """Create a job instance for estimation."""
        self._check_job(job)
        self._job = Job(job)

        try:
            # Fails on python 2
            self._job.tmp = tempfile.TemporaryDirectory(prefix='mlens_',
                                                        dir=dir)
            self._job.dir = self._job.tmp.name
        except Exception:
            self._job.dir = tempfile.mkdtemp(prefix='mlens_', dir=dir)

        # Build mmaps for inputs
        for name, arr in zip(('X', 'y'), (X, y)):

            if arr is None:
                # Can happen if y is not specified (i.e. during prediction)
                continue

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
                f = os.path.join(self._job.dir, '%s.mmap' % name)
                if os.path.exists(f):
                    os.unlink(f)
                dump(arr, f)

            # Get memmap in read-only mode (we don't want to corrupt the input)
            if name is 'y' and y is not None:
                self._job.y = _load_mmap(f)
            else:
                # Store X as the first input matrix in list of inputs matrices
                self._job.P = [_load_mmap(f)]

        # Append pre-allocated prediction arrays in r+ to the P list
        # Each layer will be fitted on P[i] and write to P[i + 1]
        for n, (name, lyr) in enumerate(self.layers.layers.items()):

            f = os.path.join(self._job.dir, '%s.mmap' % name)

            # We call the indexers fit method now at initialization - if there
            # is something funky with indexing it is better to catch it now
            # than mid-estimation
            lyr.indexer.fit(self._job.P[n])

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
        # Sample size is full for prediction, for fitting
        # it can be less if predictions are not generated for full train set
        s0 = lyr.indexer.n_test_samples if self._job.j != 'predict' else \
            lyr.indexer.n_samples

        # Number of prediction columns depends on:
        # 1. number of estimators in layer
        # 2. if predict_proba, number of classes in training set
        # 3. number of subsets (default is one for all data)
        # Note that 1. and 2. are encoded in n_pred (see Layer) but 2.
        # depends on the data and thus has to be handled by the manager.
        s1 = lyr.n_pred

        if lyr.proba:
            if self._job.j == 'fit':
                lyr.classes_ = self._job.l = np.unique(self._job.y).shape[0]

            s1 *= lyr.classes_

        return s0, s1

    @staticmethod
    def _check_job(job):
        """Check that a valid job is initialized."""
        if job not in JOBS:
            raise NotImplementedError('The job %s is not valid for the input '
                                      'for the ParallelProcessing job '
                                      'manager. Accepted jobs: %r.'
                                      % (job, list(JOBS)))

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
                self._partial_process(n, lyr, parallel)

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
        # Delete all contents from cache
        try:
            self._job.tmp.cleanup()

        except AttributeError:
            # Fall back on shutil
            try:
                shutil.rmtree(self._job.dir)

            except OSError:
                # This can fail on Windows
                try:
                    subprocess.Popen('rmdir /S /Q %s' % self._job.dir,
                                     shell=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                except OSError:
                    warnings.warn("Failed to delete cache at %s."
                                  "If created with default settings, will be "
                                  "removed on reboot. For immediate "
                                  "removal, manual removal is required." %
                                  self._job.dir, ParallelProcessingWarning)

        finally:
            # Always release process memory
            del self._job

            gc.collect()

            if not len(gc.garbage) == 0:
                warnings.warn("Clearing process memory failed, "
                              "uncollectable : %r." % gc.garbage,
                              ParallelProcessingWarning)

            self._initialized = 0

    def _partial_process(self, n, lyr, parallel):
        """Generic method for processing a :class:`layer` with ``attr``."""
        # Fire up the estimation instance
        kwd = lyr.cls_kwargs if lyr.cls_kwargs is not None else {}
        e = ENGINES[lyr.cls](lyr, **kwd)

        # Get function to process and its variables
        f = getattr(e, self._job.j)
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


###############################################################################
class ParallelEvaluation(object):

    """Parallel cross-validation engine.

    Parameters
    ----------
    evaluator : :class:`Evaluator`
        The ``Evaluator`` that instantiated the processor.
    """

    __slots__ = ['evaluator', '_initialized', '_job']

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self._initialized = 0

    def initialize(self, X, y=None, dir=None):
        """Create cache and memmap X and y."""
        self._job = Job('evaluate')

        try:
            # Fails on python 2
            self._job.tmp = tempfile.TemporaryDirectory(prefix='mlens_',
                                                        dir=dir)
            self._job.dir = self._job.tmp.name
        except Exception:
            self._job.dir = tempfile.mkdtemp(prefix='mlens_', dir=dir)

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
                f = os.path.join(self._job.dir, '%s.mmap' % name)
                if os.path.exists(f):
                    os.unlink(f)
                dump(arr, f)

            # Get memmap in read-only mode (we don't want to corrupt the input)
            if name is 'y':
                self._job.y = _load_mmap(f)
            else:
                self._job.P = _load_mmap(f)

        self._initialized = 1

        # Release any memory before going into process
        gc.collect()

    def process(self, attr):
        """Fit all layers in the attached :class:`LayerContainer`."""
        check_initialized(self)

        # Use context manager to ensure same parallel job during entire process
        with Parallel(n_jobs=self.evaluator.n_jobs,
                      temp_folder=self._job.dir,
                      max_nbytes=None,
                      mmap_mode='r+',
                      verbose=self.evaluator.verbose,
                      backend=self.evaluator.backend) as parallel:

            f = ENGINES['evaluation'](self.evaluator)

            getattr(f, attr)(parallel, self._job.P, self._job.y,
                             self._job.dir)

    def terminate(self):
        """Remove temporary folder and all cache data."""
        # Delete all contents from cache
        try:
            self._job.tmp.cleanup()

        except AttributeError:
            # Fall back on shutil for python 2
            try:
                shutil.rmtree(self._job.dir)
            except OSError:
                # This can fail on Windows
                warnings.warn("Failed to delete cache at %s. Will be removed "
                              "on reboot. Consider manual removal (rmdir)" %
                              self._job.dir, ParallelProcessingWarning)

        finally:
            # Always release process memory
            del self._job

            gc.collect()

            if not len(gc.garbage) == 0:
                warnings.warn("Clearing process memory failed, "
                              "uncollectable : %r." % gc.garbage,
                              ParallelProcessingWarning)

            self._initialized = 0
