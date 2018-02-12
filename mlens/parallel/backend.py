"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Parallel processing backend classes. Manages memory-mapping of data, estimation
caching and job scheduling.
"""
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=useless-super-delegation

from __future__ import with_statement, division

import gc
import os
import shutil
import subprocess
import tempfile
import warnings

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import issparse, hstack

from .. import config
from ..externals.joblib import Parallel, dump, load
from ..utils import check_initialized
from ..utils.exceptions import (ParallelProcessingError,
                                ParallelProcessingWarning)
from ..externals.sklearn.validation import check_random_state


###############################################################################
def _dtype(a, b=None):
    """Utility for getting a dtype"""
    return getattr(a, 'dtype', getattr(b, 'dtype', None))


def dump_array(array, name, path):
    """Dump array for memmapping.

    Parameters
    ----------
    array : array-like
        Array to be persisted

    name : str
        Name of file

    path : str
        Path to cache.

    Returns
    -------
    f: array-like
        memory-mapped array.
    """
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
        f = os.path.join(path, '%s.mmap' % name)
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


def _set_path(job, path, threading):
    """Build path as a cache or list depending on whether using threading"""
    if path:
        if not isinstance(path, str) and not threading:
            raise ValueError("Path must be a str with backend=multiprocessing."
                             " Got %r" % path.__class__)
        elif not isinstance(path, (str, dict)):
            raise ValueError("Invalid path format. Should be one of "
                             "str, dict. Got %r" % path.__class__)
        job.dir = path
        return job

    if threading:
        # No need to pickle
        job.dir = dict()
        return job

    # Else, need a directory
    path = config.get_tmpdir()
    try:
        job.tmp = tempfile.TemporaryDirectory(
            prefix=config.get_prefix(), dir=path)
        job.dir = job.tmp.name
    except AttributeError:
        # Fails on python 2
        job.dir = tempfile.mkdtemp(prefix=config.get_prefix(), dir=path)
    return job


###############################################################################
class Job(object):

    """Container class for holding and managing job data.

    :class:`Job` is intended as a on-the-fly job handler that keeps track
    of input data, predictions, and manages estimation caches.

    .. versionchanged:: 0.2.0

    See Also
    --------
    :class:`ParallelProcessing`, :class:`ParallelEvaluation`

    Parameters
    ----------
    job : str
        Type of job to run. One of ``'fit'``, ``'transform'``, ``'predict'``.

    stack : bool
        Whether to stack outputs when calls to
        :func:`~mlens.parallel.backend.Job.update` are made. This will make
        the ``predict_out`` array become ``predict_in``.

    split : bool
        Whether to create a new sub-cache when the
        :attr:`~mlens.parallel.backend.Job.args` property is called.

    dir : str, dict, optional
        estimation cache. Pass dictionary for use with multiprocessing or a
        string pointing to the disk directory to create the cache in

    tmp : obj, optional
        a Tempfile object for temporary directories

    targets : array-like of shape [n_in_samples,], optional
        input targets

    predict_in : array-like of shape [n_in_samples, n_in_features], optional
        input data

    predict_out : array_like of shape [n_out_samples, n_out_features], optional
        prediction output array
    """

    __slots__ = ['targets', 'predict_in', 'predict_out', 'dir', 'job', 'tmp',
                 '_n_dir', 'kwargs', 'stack', 'split']

    def __init__(self, job, stack, split, dir=None, tmp=None, predict_in=None,
                 targets=None, predict_out=None):
        self.job = job
        self.stack = stack
        self.split = split

        self.targets = targets
        self.predict_in = predict_in
        self.predict_out = predict_out
        self.tmp = tmp
        self.dir = dir
        self._n_dir = 0

    def clear(self):
        """Clear output data for new task"""
        self.predict_out = None

    def update(self):
        """Updated output array and shift to input if stacked.

        If stacking is en force, the output array will replace the input
        array, and used as input for subsequent jobs. Sparse matrices are
        force-converted to ``csr`` format.
        """
        if self.predict_out is None:
            return
        if (issparse(self.predict_out) and not
                self.predict_out.__class__.__name__.startswith('csr')):
            # Enforce csr on spare matrices
            self.predict_out = self.predict_out.tocsr()

        if self.stack:
            self.predict_in = self.predict_out
            self.rebase()

    def rebase(self):
        """Rebase output labels to input indexing.

        Some indexers that only generate predictions for subsets of the
        training data require the targets to be rebased. Since indexers
        operate in a strictly sequential manner, rebase simply drop the first
        ``n`` observations in the target vector until number of observations
        remaining coincide.

        .. seealso::
            :class:`~mlens.index.blend.BlendIndex`

        """
        if self.targets is not None and (
                    self.targets.shape[0] > self.predict_in.shape[0]):
            # This is legal if X is a prediction matrix generated by predicting
            # only a subset of the original training set.
            # Since indexing is strictly monotonic, we can simply discard
            # the first observations in y to get the corresponding labels.
            rebase = self.targets.shape[0] - self.predict_in.shape[0]
            self.targets = self.targets[rebase:]

    def shuffle(self, random_state):
        """Shuffle inputs.

        Permutes the indexing of ``predict_in`` and ``y`` arrays.

        Parameters
        ----------
        random_state : int, obj
            Random seed number or generator to use.
        """
        r = check_random_state(random_state)
        idx = r.permutation(self.targets.shape[0])
        self.predict_in = self.predict_in[idx]
        self.targets = self.targets[idx]

    def subdir(self):
        """Return a cache subdirectory

        If ``split`` is en force, a new sub-cache will be created in the
        main cache. Otherwise the same sub-cache as used in previous call
        will be returned.

        .. versionadded:: 0.2.0

        Returns
        -------
        cache : str, list
            Either a string pointing to a cache persisted to disk, or an
            in-memory cache in the form of a list.
        """
        path_name = "task_%s" % str(self._n_dir)
        if self.split:
            # Increment sub-cache counter
            self._n_dir += 1

        if isinstance(self.dir, str):
            path = os.path.join(self.dir, path_name)
            cache_exists = os.path.exists(path)
            # Persist cache to disk
            if cache_exists and self.split:
                raise ParallelProcessingError(
                    "Subdirectory %s exist. Clear cache." % path_name)
            elif not cache_exists:
                os.mkdir(path)
            return path

        # Keep in memory
        if path_name in self.dir and self.split:
            raise ParallelProcessingError(
                "Subdirectory %s exist. Clear cache." % path_name)
        elif path_name not in self.dir:
            self.dir[path_name] = list()
        return self.dir[path_name]

    def args(self, **kwargs):
        """Produce args dict

        .. versionadded:: 0.2.0

        Returns the arguments dictionary passed to a task of a parallel
        processing manager. Output dictionary has the following form::

            out = {'auxiliary':
                       {'X': self.predict_in, 'P': self.predict_out},
                   'main':
                       {'X': self.predict_in, 'P': self.predict_out},
                   'dir':
                       self.subdir(),
                   'job':
                        self.job
                    }

        Parameters
        ----------
        **kwargs : optional
            Optional keyword arguments to pass to the task.

        Returns
        -------
        args : dict
            Arguments dictionary

        """
        aux_feed = {'X': self.predict_in, 'P': None}
        main_feed = {'X': self.predict_in, 'P': self.predict_out}

        if self.job in ['fit', 'evaluate']:
            main_feed['y'] = self.targets
            aux_feed['y'] = self.targets

        out = dict()
        if kwargs:
            out.update(kwargs)

        out = {'auxiliary': aux_feed,
               'main': main_feed,
               'dir': self.subdir(),
               'job': self.job}

        return out


###############################################################################
class BaseProcessor(object):

    """Parallel processing base class.

    Base class for parallel processing engines.

    Parameters
    ----------
    backend: str, optional
        Type of backend. One of ``'threading'``, ``'multiprocessing'``,
        ``'sequential'``.

    n_jobs : int, optional
        Degree of concurrency.

    verbose: bool, int, optional
        Level of verbosity of the
        :class:`~mlens.externals.joblib.parallel.Parallel` instance.
    """

    __meta_class__ = ABCMeta

    __slots__ = ['caller', '__initialized__', '__threading__', 'job',
                 'n_jobs', 'backend', 'verbose']

    @abstractmethod
    def __init__(self, backend=None, n_jobs=None, verbose=None):
        self.job = None
        self.__initialized__ = 0

        self.backend = config.get_backend() if not backend else backend
        self.n_jobs = -1 if not n_jobs else n_jobs
        self.verbose = False if not verbose else verbose
        self.__threading__ = self.backend == 'threading'

    def __enter__(self):
        return self

    def initialize(self, job, X, y, path,
                   warm_start=False, return_preds=False, **kwargs):
        """Initialize processing engine.

        Set up the job parameters before an estimation call. Calling
        :func:`~mlens.parallel.backend.BaseProcessor.clear`
        undoes initialization.

        Parameters
        ----------
        job : str
            type of job to complete with each task. One of ``'fit'``,
            ``'predict'`` and ``'transform'``.

        X : array-like of shape [n_samples, n_features]
            Input data

        y : array-like of shape [n_samples,], optional.
            targets. Required for fit, should not be passed to predict or
            transform jobs.

        path : str or dict, optional
            Custom estimation cache. Pass a string to force use of persistent
            cache on disk. Pass a ``dict`` for in-memory cache (requires
            ``backend != 'multiprocessing'``.

        return_preds : bool or list, optional
            whether to return prediction ouput. If ``True``, final prediction
            is returned. Alternatively, pass a list of task names for which
            output should be returned.

        warm_start : bool, optional
            whether to re-use previous input data initialization. Useful if
            repeated jobs are made on the same input arrays.

        **kwargs  : optional
            optional keyword arguments to pass onto the task's call method.

        Returns
        -------
        out : dict
            An output parameter dictionary to pass to pass to an estimation
            method. Either ``None`` (no output), or
            ``{'final':True}`` for only final prediction, or
            ``{'final': False, 'return_names': return_preds}`` if a list of
            task-specific output was passed.
        """
        if not warm_start:
            self._initialize(job=job, X=X, y=y, path=path, **kwargs)

        if return_preds is True:
            return {'return_final': True}
        if return_preds is False:
            return {}
        return {'return_final': False, 'return_names': return_preds}

    def _initialize(self, job, X, y=None, path=None, **kwargs):
        """Create a job instance for estimation.

        See :func:`~mlens.parallel.backend.BaseProcess.initialize` for
        further details.
        """
        job = Job(job, **kwargs)
        job = _set_path(job, path, self.__threading__)

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
                f = dump_array(arr, name, job.dir)

            # Store data for processing
            if name == 'y' and arr is not None:
                job.targets = arr if self.__threading__ else _load_mmap(f)
            elif name == 'X':
                job.predict_in = arr \
                    if self.__threading__ else _load_mmap(f)

        self.job = job
        self.__initialized__ = 1
        gc.collect()
        return self

    def __exit__(self, *args):
        self.clear()

    def clear(self):
        """Destroy cache and reset instance job parameters."""
        # Detach Job instance
        job = self.job
        self.job = None
        self.__initialized__ = 0

        if job:
            path = job.dir
            path_handle = job.tmp

            # Release shared memory references
            del job
            gc.collect()

            # Destroy cache
            try:
                # If the cache has been persisted to disk, remove it
                if isinstance(path, str):
                    path_handle.cleanup()
            except (AttributeError, OSError):
                # Python 2 has no handler, can also fail on windows
                # Use explicit shutil process, or fall back on subprocess
                try:
                    shutil.rmtree(path)
                except OSError:
                    # Can fail on windows, need to use the shell
                    try:
                        subprocess.Popen(
                            'rmdir /S /Q %s' % path, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).kill()
                    except OSError:
                        warnings.warn(
                            "Failed to delete cache at %s."
                            "If created with default settings, will be "
                            "removed on reboot. For immediate "
                            "removal, manual removal is required." %
                            path, ParallelProcessingWarning)
            finally:
                del path, path_handle
                gc.collect()
                if gc.garbage:
                    warnings.warn(
                        "Clearing cache failed, uncollected:\n%r" %
                        gc.garbage, ParallelProcessingWarning)


class ParallelProcessing(BaseProcessor):

    """Parallel processing engine.

    Engine for running computational graph.

    :class:`ParallelProcessing` is a manager for executing a sequence of tasks
    in a given caller, where each task is run sequentially, but assumed to be
    parallelized internally. The main responsibility of
    :class:`ParallelProcessing` is to handle memory-mapping, estimation
    cache updates, input and output array updates and output collection.

    Parameters
    ----------
    caller :  obj
        the caller of the job. Either a Layer or a meta layer class
        such as Sequential.

    *args : optional
        Optional arguments to :class:`~mlens.parallel.backend.BaseProcessor`

    **kwargs : optional
        Optional keyword arguments to
        :class:`~mlens.parallel.backend.BaseProcessor`.
    """
    def __init__(self, *args, **kwargs):
        super(ParallelProcessing, self).__init__(*args, **kwargs)

    def map(self, caller, job, X, y=None, path=None,
            return_preds=False, wart_start=False, split=False, **kwargs):
        """Parallel task mapping.

        Run independent tasks in caller in parallel.

        Warning
        -------
        By default, the :~mlens.parallel.backend.ParallelProcessing.map` runs
        on a shallow cache, where all tasks share the same cache. As such, the
        user must ensure that each task has a unique name, or cache retrieval
        will be corrupted. To commit a seperate sub-cache to each task, set
        ``split=True``.

        Parameters
        ----------
        caller : iterable
            Iterable that generates accepted task instances. Caller should be
            a child of the :class:`~mlens.parallel.base.BaseBackend` class,
            and tasks need to implement an appropriate call method.

        job : str
            type of job to complete with each task. One of ``'fit'``,
            ``'predict'`` and ``'transform'``.

        X : array-like of shape [n_samples, n_features]
            Input data

        y : array-like of shape [n_samples,], optional.
            targets. Required for fit, should not be passed to predict or
            transform jobs.

        path : str or dict, optional
            Custom estimation cache. Pass a string to force use of persistent
            cache on disk. Pass a ``dict`` for in-memory cache (requires
            ``backend != 'multiprocessing'``.

        return_preds : bool or list, optional
            whether to return prediction ouput. If ``True``, final prediction
            is returned. Alternatively, pass a list of task names for which
            output should be returned.

        warm_start : bool, optional
            whether to re-use previous input data initialization. Useful if
            repeated jobs are made on the same input arrays.

        split : bool, default = False
            whether to commit a separate sub-cache to each task.

        **kwargs : optional
            optional keyword arguments to pass onto each task.

        Returns
        -------
        out: array-like, list, optional
            Prediction array(s).
        """
        out = self.initialize(
            job=job, X=X, y=y, path=path, warm_start=wart_start,
            return_preds=return_preds, split=split, stack=False)
        return self.process(caller=caller, out=out, **kwargs)

    def stack(self, caller, job, X, y=None, path=None, return_preds=False,
              warm_start=False, split=True, **kwargs):
        """Stacked parallel task mapping.

        Run stacked tasks in caller in parallel.

        This method runs a stack of tasks as a stack, where the output of
        each task is the input to the next.

        Warning
        -------
        By default, the :func:`~mlens.parallel.backend.ParallelProcessing.stack`
        method runs on a deep cache, where each tasks has a separate cache.
        As such, the user must ensure that tasks don't depend on data cached
        by previous tasks. To run all tasks in a single sub-cache, set
        ``split=False``.

        Parameters
        ----------
        caller : iterable
            Iterable that generates accepted task instances. Caller should be
            a child of the :class:`~mlens.parallel.base.BaseBackend` class,
            and tasks need to implement an appropriate call method.

        job : str
            type of job to complete with each task. One of ``'fit'``,
            ``'predict'`` and ``'transform'``.

        X : array-like of shape [n_samples, n_features]
            Input data

        y : array-like of shape [n_samples,], optional.
            targets. Required for fit, should not be passed to predict or
            transform jobs.

        path : str or dict, optional
            Custom estimation cache. Pass a string to force use of persistent
            cache on disk. Pass a ``dict`` for in-memory cache (requires
            ``backend != 'multiprocessing'``.

        return_preds : bool or list, optional
            whether to return prediction output. If ``True``, final prediction
            is returned. Alternatively, pass a list of task names for which
            output should be returned.

        warm_start : bool, optional
            whether to re-use previous input data initialization. Useful if
            repeated jobs are made on the same input arrays.

        split : bool, default = True
            whether to commit a separate sub-cache to each task.

        **kwargs : optional
            optional keyword arguments to pass onto each task.

        Returns
        -------
        out: array-like, list, optional
            Prediction array(s).
        """
        out = self.initialize(
            job=job, X=X, y=y, path=path, warm_start=warm_start,
            return_preds=return_preds, split=split, stack=True)
        return self.process(caller=caller, out=out, **kwargs)

    def process(self, caller, out, **kwargs):
        """Process job.

        Main method for processing a caller. Requires the instance to be
        setup by a prior call to
        :func:`~mlens.parallel.backend.BaseProcessor.initialize`.

        .. seealso::
            :func:`~mlens.parallel.backend.ParallelProcessing.map`,
            :func:`~mlens.parallel.backend.ParallelProcessing.stack`

        Parameters
        ----------
        caller : iterable
            Iterable that generates accepted task instances. Caller should be
            a child of the :class:`~mlens.parallel.base.BaseBackend` class,
            and tasks need to implement an appropriate call method.

        out : dict
            A dictionary with output parameters. Pass an empty dict for no
            output. See
            :func:`~mlens.parallel.backend.BaseProcessor.initialize` for more
            details.

        Returns
        -------
        out: array-like, list, optional
            Prediction array(s).
        """
        check_initialized(self)

        return_names = out.pop('return_names', [])
        return_final = out.pop('return_final', False)
        out = list() if return_names else None

        tf = self.job.dir if not isinstance(self.job.dir, list) else None
        with Parallel(n_jobs=self.n_jobs, temp_folder=tf, max_nbytes=None,
                      mmap_mode='w+', verbose=self.verbose,
                      backend=self.backend) as parallel:

            for task in caller:
                self.job.clear()

                self._partial_process(task, parallel, **kwargs)

                if task.name in return_names:
                    out.append(self.get_preds(dtype=_dtype(task)))

                self.job.update()

        if return_final:
            out = self.get_preds(dtype=_dtype(task))
        return out

    def _partial_process(self, task, parallel, **kwargs):
        """Process given task"""
        if self.job.job == 'fit' and getattr(task, 'shuffle', False):
            self.job.shuffle(getattr(task, 'random_state', None))

        task.setup(self.job.predict_in, self.job.targets, self.job.job)

        if not task.__no_output__:
            self._gen_prediction_array(task, self.job.job, self.__threading__)

        task(self.job.args(**kwargs), parallel=parallel)

        if not task.__no_output__ and getattr(task, 'n_feature_prop', 0):
            self._propagate_features(task)

    def _propagate_features(self, task):
        """Propagate features from input array to output array."""
        p_out, p_in = self.job.predict_out, self.job.predict_in

        # Check for loss of obs between layers (i.e. with blendindex)
        n_in, n_out = p_in.shape[0], p_out.shape[0]
        r = int(n_in - n_out)

        if not issparse(p_in):
            # Simple item setting
            p_out[:, :task.n_feature_prop] = p_in[r:, task.propagate_features]
        else:
            # Need to populate propagated features using scipy sparse hstack
            self.job.predict_out = hstack(
                [p_in[r:, task.propagate_features],
                 p_out[:, task.n_feature_prop:]]
            ).tolil()

    def _gen_prediction_array(self, task, job, threading):
        """Generate prediction array either in-memory or persist to disk."""
        shape = task.shape(job)
        if threading:
            self.job.predict_out = np.empty(shape, dtype=_dtype(task))
        else:
            f = os.path.join(self.job.dir, '%s_out_array.mmap' % task.name)
            try:
                self.job.predict_out = np.memmap(
                    filename=f, dtype=_dtype(task), mode='w+', shape=shape)
            except Exception as exc:
                raise OSError(
                    "Cannot create prediction matrix of shape ("
                    "%i, %i), size %i MBs, for %s.\n Details:\n%r" %
                    (shape[0], shape[1], 8 * shape[0] * shape[1] / (1024 ** 2),
                     task.name, exc))

    def get_preds(self, dtype=None, order='C'):
        """Return prediction matrix.

        Parameters
        ----------
        dtype : numpy dtype object, optional
            data type to return

        order : str (default = 'C')
            data order. See :class:`numpy.asarray` for details.

        Returns
        -------
        P: array-like
            Prediction array
        """
        if not hasattr(self, 'job'):
            raise ParallelProcessingError(
                "Processor has been terminated:\ncannot retrieve final "
                "prediction array from cache.")
        if dtype is None:
            dtype = config.get_dtype()

        if issparse(self.job.predict_out):
            return self.job.predict_out
        return np.asarray(self.job.predict_out, dtype=dtype, order=order)


###############################################################################
class ParallelEvaluation(BaseProcessor):

    """Parallel cross-validation engine.

    Minimal parallel processing engine. Similar to :class:`ParallelProcessing`,
    but offers less features, only fits the *callers* indexer, and excepts
    no task output.
    """

    def __init__(self, *args, **kwargs):
        super(ParallelEvaluation, self).__init__(*args, **kwargs)

    def process(self, caller, case, X, y, path=None, **kwargs):
        """Process caller.

        Parameters
        ----------
        caller: iterable
            Iterable for evaluation job.s Expected caller is a
            :class:`Evaluator` instance.

        case: str
            evaluation case to run on the evaluator. One of
            ``'preprocess'`` and ``'evaluate'``.

        X: array-like of shape [n_samples, n_features]
            Input data

        y: array-like of shape [n_samples,], optional.
            targets. Required for fit, should not be passed to predict or
            transform jobs.

        path: str or dict, optional
            Custom estimation cache. Pass a string to force use of persistent
            cache on disk. Pass a ``dict`` for in-memory cache (requires
            ``backend != 'multiprocessing'``.
        """
        self._initialize(
            job='fit', X=X, y=y, path=path, split=False, stack=False)
        check_initialized(self)

        # Use context manager to ensure same parallel job during entire process
        tf = self.job.dir if not isinstance(self.job.dir, list) else None
        with Parallel(n_jobs=self.n_jobs, temp_folder=tf, max_nbytes=None,
                      mmap_mode='w+', verbose=self.verbose,
                      backend=self.backend) as parallel:

            caller.indexer.fit(self.job.predict_in, self.job.targets, self.job.job)
            caller(parallel, self.job.args(**kwargs), case)
