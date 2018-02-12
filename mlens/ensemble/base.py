"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Base classes for ensemble layer management.
"""
# pylint: disable=protected-access
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes


from __future__ import division, print_function, with_statement

from abc import ABCMeta, abstractmethod
import warnings

from .. import config
from ..parallel import Layer, ParallelProcessing, make_group
from ..parallel.base import BaseStacker
from ..externals.sklearn.validation import check_random_state
from ..utils import (check_ensemble_build, check_inputs, print_time,
                     safe_print, IdTrain, format_name)
from ..utils.exceptions import (
    LayerSpecificationWarning, NotFittedError, NotInitializedError)
from ..metrics import Data
from ..externals.sklearn.base import BaseEstimator, clone
try:
    # Try get performance counter
    from time import perf_counter as time
except ImportError:
    # Fall back on wall clock
    from time import time


GLOBAL_SEQUENTIAL_NAME = list()


def check_kwargs(kwargs, forbidden):
    """Pop unwanted arguments and issue warning"""
    for f in forbidden:
        s = kwargs.pop(f, None)
        if s is not None:
            warnings.warn(
                "Layer-specific parameter '%s' contradicts"
                "ensemble-wide settings. Ignoring." % f,
                LayerSpecificationWarning)


def print_job(lc, start_message):
    """Print job details.

    Parameters
    ----------
    lc : :class:`Sequential`
        The LayerContainer instance running the job.

    start_message : str
        Initial message.
    """
    f = "stdout" if lc.verbose < 10 else "stderr"
    if lc.verbose:
        safe_print("\n%s %d layers" % (start_message, len(lc.stack)),
                   file=f, flush=True)
        if lc.verbose >= 5:
            safe_print("""[INFO] n_jobs = %i
[INFO] backend = %r
[INFO] start_method = %r
[INFO] cache = %r
""" % (lc.n_jobs, lc.backend, config.get_start_method(), config.get_tmpdir()),
                       file=f, flush=True)

    t0 = time()
    return f, t0


###############################################################################
class Sequential(BaseStacker):

    r"""Container class for a stack of sequentially processed estimators.

    The Sequential class stories all layers as an ordered dictionary
    and modifies possesses a ``get_params`` method to appear as an estimator
    in the Scikit-learn API. This allows correct cloning and parameter
    updating.


    Parameters
    ----------
    stack: list, optional (default = None)
        list of estimators (i.e. layers) to build instance with.

    n_jobs : int (default = -1)
        Degree of concurrency. Set ``n_jobs = -1`` for maximal parallelism and
        ``n_jobs=1`` for sequential processing.

    backend : str, (default="threading")
        the joblib backend to use (i.e. "multiprocessing" or "threading").

    raise_on_exception : bool (default = False)
        raise error on soft exceptions. Otherwise issue warning.

    verbose : int or bool (default = False)
        level of verbosity.

            - ``verbose = 0`` silent (same as ``verbose = False``)
            - ``verbose = 1`` messages at start and finish
              (same as ``verbose = True``)
            - ``verbose = 2`` messages for each layer
            - etc

        If ``verbose >= 10`` prints to ``sys.stderr``, else ``sys.stdout``.
    """

    def __init__(self, name=None, verbose=False, stack=None, **kwargs):
        if stack and not isinstance(stack, list):
            if stack.__class__.__name__.lower() == 'layer':
                stack = [stack]
            else:
                raise ValueError(
                    "Expect stack to be a Layer or a list of Layers. "
                    "Got %r" % stack)

        name = format_name(name, 'sequential', GLOBAL_SEQUENTIAL_NAME)
        super(Sequential, self).__init__(
            stack=stack, name=name, verbose=verbose, **kwargs)

    def __iter__(self):
        """Generator for stacked layers"""
        for layer in self.stack:
            yield layer

    def fit(self, X, y=None, **kwargs):
        r"""Fit instance.

        Iterative fits each layer in the stack on the output of
        the subsequent layer. First layer is fitted on input data.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for fitting and predicting.

        y : array-like of shape = [n_samples, ]
            training labels.

        **kwargs : optional
            optional arguments to processor
       """
        if not self.__stack__:
            raise NotInitializedError("No elements in stack to fit.")

        f, t0 = print_job(self, "Fitting")

        with ParallelProcessing(self.backend, self.n_jobs,
                                max(self.verbose - 4, 0)) as manager:
            out = manager.stack(self, 'fit', X, y, **kwargs)

        if self.verbose:
            print_time(t0, "{:<35}".format("Fit complete"), file=f)

        if out is None:
            return self
        return out

    def fit_transform(self, X, y=None, **kwargs):
        r"""Fit instance and return cross-validated predictions.

        Equivalent to ``Sequential().fit(X, y, return_preds=True)``

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for fitting and predicting.

        y : array-like of shape = [n_samples, ]
            training labels.

        **kwargs : optional
            optional arguments to processor
        """
        return self.fit(X, y, return_preds=True, **kwargs)

    def predict(self, X, **kwargs):
        r"""Predict.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        **kwargs : optional
            optional keyword arguments.

        Returns
        -------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            predictions from final layer.
        """
        if not self.__fitted__:
            NotFittedError("Instance not fitted.")

        f, t0 = print_job(self, "Predicting")

        out = self._predict(X, 'predict', **kwargs)

        if self.verbose:
            print_time(t0, "{:<35}".format("Predict complete"),
                       file=f, flush=True)
        return out

    def transform(self, X, **kwargs):
        """Predict using sub-learners as is done during the ``fit`` call.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        *args : optional
            optional arguments.

        **kwargs : optional
            optional keyword arguments.

        Returns
        -------
        X_pred : array-like of shape = [n_test_samples, n_fitted_estimators]
            predictions from ``fit`` call to final layer.
        """
        if not self.__fitted__:
            NotFittedError("Instance not fitted.")

        f, t0 = print_job(self, "Transforming")

        out = self._predict(X, 'transform', **kwargs)

        if self.verbose:
            print_time(t0, "{:<35}".format("Transform complete"),
                       file=f, flush=True)

        return out

    def _predict(self, X, job, **kwargs):
        r"""Generic for processing a predict job through all layers.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        job : str
            type of prediction. Should be 'predict' or 'transform'.

        Returns
        -------
        X_pred : array-like
            predictions from final layer. Either predictions from ``fit`` call
            or new predictions on X using base learners fitted on all training
            data.
        """
        r = kwargs.pop('return_preds', True)
        with ParallelProcessing(self.backend, self.n_jobs,
                                max(self.verbose - 4, 0)) as manager:
            out = manager.stack(self, job, X, return_preds=r, **kwargs)

        if not isinstance(out, list):
            out = [out]
        out = [p.squeeze() for p in out]
        if len(out) == 1:
            out = out[0]
        return out

    @property
    def data(self):
        """Ensemble data"""
        out = list()
        for layer in self.stack:
            d = layer.raw_data
            if not d:
                continue
            out.extend([('%s/%s' % (layer.name, k), v) for k, v in d])
        return Data(out)


###############################################################################
class BaseEnsemble(BaseEstimator):

    """BaseEnsemble class.

    Core ensemble class methods used to add ensemble layers and manipulate
    parameters.

    Parameters
    ----------
    model_selection: bool (default=False)
        Whether to use the ensemble in model selection mode. If ``True``,
        this will alter the ``transform`` method. When calling ``transform``
        on new data, the ensemble will call ``predict``, while calling
        ``transform`` with the training data reproduces predictions from the
        ``fit`` call. Hence the ensemble can be used as a pure transformer
        in a preprocessing pipeline passed to the :class:`Evaluator`, as
        training folds are faithfully reproduced as during a ``fit``call and
        test folds are transformed with the ``predict`` method.

    samples_size: int (default=20)
        size of training set sample
        (``[min(sample_size, X.size[0]), min(X.size[1], sample_size)]``

    shuffle: bool (default=False)
        whether to shuffle input data during fit calls

    random_state: bool (default=False)
        random seed.

    scorer: obj, optional
        scorer function

    verbose: bool, optional
        verbosity

    array_check: int (default=2)
        severity of array checks. ``2`` mimicks Scikit-learn, ``1`` warns,
        ``0`` disables.

    samples_size: int (default=20)
        size of training set sample
        (``[min(sample_size, X.size[0]), min(X.size[1], sample_size)]``
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(
            self, shuffle=False, random_state=None, scorer=None, verbose=False,
            layers=None, array_check=2, model_selection=False, sample_size=20,
            **kwargs):
        self.shuffle = shuffle
        self.random_state = random_state
        self.scorer = scorer
        self.array_check = array_check
        self._model_selection = model_selection
        self._verbose = verbose
        self.layers = layers if layers else list()

        self.sample_size = sample_size
        self.model_selection = model_selection

        self._backend = Sequential(verbose=verbose, **kwargs)
        self.raise_on_exception = self._backend.raise_on_exception
        if layers:
            layers_ = clone(layers)
            self._backend.push(*layers_)

    def add(self, estimators, indexer, preprocessing=None, **kwargs):
        """Method for adding a layer.

        Parameters
        -----------
        estimators: dict of lists or list of estimators, or `:class:`Layer`.
            Pre-made layer or estimators to construct layer with.
            If ``preprocessing`` is ``None`` or ``list``, ``estimators`` should
            be a ``list``. The list can either contain estimator instances,
            named tuples of estimator instances, or a combination of both. ::

                option_1 = [estimator_1, estimator_2]
                option_2 = [("est-1", estimator_1), ("est-2", estimator_2)]
                option_3 = [estimator_1, ("est-2", estimator_2)]

            If different preprocessing pipelines are desired, a dictionary
            that maps estimators to preprocessing pipelines must be passed.
            The names of the estimator dictionary must correspond to the
            names of the estimator dictionary. ::

                preprocessing_cases = {"case-1": [trans_1, trans_2].
                                       "case-2": [alt_trans_1, alt_trans_2]}

                estimators = {"case-1": [est_a, est_b].
                              "case-2": [est_c, est_d]}

            The lists for each dictionary entry can be any of ``option_1``,
            ``option_2`` and ``option_3``.

        indexer : instance or None (default = None)
            Indexer instance to use. Defaults to the layer class
            indexer with default settings. See :mod:`mlens.base` for details.

        preprocessing: dict of lists or list, optional (default = None)
            preprocessing pipelines for given layer. If
            the same preprocessing applies to all estimators, ``preprocessing``
            should be a list of transformer instances. The list can contain the
            instances directly, named tuples of transformers,
            or a combination of both. ::

                option_1 = [transformer_1, transformer_2]
                option_2 = [("trans-1", transformer_1),
                            ("trans-2", transformer_2)]
                option_3 = [transformer_1, ("trans-2", transformer_2)]

            If different preprocessing pipelines are desired, a dictionary
            that maps preprocessing pipelines must be passed. The names of the
            preprocessing dictionary must correspond to the names of the
            estimator dictionary. ::

                preprocessing_cases = {"case-1": [trans_1, trans_2].
                                       "case-2": [alt_trans_1, alt_trans_2]}

                estimators = {"case-1": [est_a, est_b].
                              "case-2": [est_c, est_d]}

            The lists for each dictionary entry can be any of ``option_1``,
            ``option_2`` and ``option_3``.

        **kwargs : optional
            keyword arguments to be passed onto the layer at instantiation.

        Returns
        ----------
        self : instance
            Modified instance.
        """
        lyr = self._build_layer(estimators, indexer, preprocessing, **kwargs)

        self.layers.append(clone(lyr))
        setattr(self, lyr.name.replace('-', '_'), lyr)

        self._backend.push(lyr)
        return self

    def replace(self, idx, estimators, indexer, preprocessing=None, **kwargs):
        """Replace a layer.

        Replace a layer in the stack with a new layer.
        See :func:`add` for full parameter documentation.

        Parameters
        -----------
        idx: int
            Position in stack of layer to replace. Indexing is 0-based.

        estimators: dict of lists or list of estimators, or `:class:`Layer`.
            Pre-made layer or estimators to construct layer with.

        indexer : instance or None (default = None)
            Indexer instance to use. Defaults to the layer class
            indexer with default settings. See :mod:`mlens.base` for details.

        preprocessing: dict of lists or list, optional (default = None)
            preprocessing pipelines for given layer.

        **kwargs : optional
            keyword arguments to be passed onto the layer at instantiation.

        Returns
        ----------
        self : instance
            Modified instance
        """
        lyr = self._build_layer(estimators, indexer, preprocessing, **kwargs)

        self.layers[idx] = clone(lyr)
        setattr(self, lyr.name.replace('-', '_'), lyr)

        self._backend.replace(idx, lyr)
        return self

    def remove(self, idx):
        """Remove a layer from stack

        Remove a layer at a given position from stack.

        Parameters
        ----------
        idx: int
            Position in stack. Indexing is 0-based.

        Returns
        -------
        self: instance
            Modified instance
        """
        name = self.layers[idx].name

        self.layers.pop(idx)
        delattr(self, name.replace('-', '_'))

        self._backend.pop(idx)
        return self

    def fit(self, X, y=None, **kwargs):
        """Fit ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like of shape = [n_samples, ] or None (default = None)
            output vector to trained estimators on.

        Returns
        -------
        self : instance
            class instance with fitted estimators.
        """
        if not check_ensemble_build(self._backend):
            # No layers instantiated, but raise_on_exception is False
            return self

        X, y = check_inputs(X, y, self.array_check)

        if self.model_selection:
            self._id_train.fit(X)

        out = self._backend.fit(X, y, **kwargs)
        if out is not self._backend:
            # fit_transform
            return out
        else:
            return self

    def transform(self, X, y=None, **kwargs):
        """Transform with fitted ensemble.

        Replicates cross-validated prediction process from training.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like, shape[n_samples, ]
            targets. Needs to be passed as input in model selection mode as
            some indexers will reduce the size of the input array (X) and
            y must be adjusted accordingly.

        Returns
        -------
        pred : array-like or tuple, shape=[n_samples, n_features]
            predictions for provided input array. If in model selection mode,
            return a tuple ``(X_trans, y_trans)`` where ``y_trans`` is either
            ``y``, or a trunctated version to match the samples in ``X_trans``.
        """
        if not check_ensemble_build(self._backend):
            # No layers instantiated, but raise_on_exception is False
            return

        X, y = check_inputs(X, y, check_level=self.array_check)

        if self.model_selection:
            if y is None:
                raise TypeError(
                    "In model selection mode, y is a required argument.")

            # Need to modify the transform method to account for blending
            # cutting X in size, so y needs to be cut too
            if not self._id_train.is_train(X):
                return self.predict(X, **kwargs), y

            # Asked to reproduce predictions during fit, here we need to
            # account for that in model selection mode,
            # blend ensemble will cut X in observation size so need to adjust y
            X = self._backend.transform(X, **kwargs)
            if X.shape[0] != y.shape[0]:
                r = y.shape[0] - X.shape[0]
                y = y[r:]
            return X, y

        return self._backend.transform(X, **kwargs)

    def fit_transform(self, X, y, **kwargs):
        r"""Fit ensemble and return cross-validated predictions.

        Equivalent to ``ensemble.fit(X, y).transform(X)``, but more efficient.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for fitting and predicting.

        y : array-like of shape = [n_samples, ]
            training labels.

        **kwargs : optional
            optional arguments to processor

        Returns
        -------
        pred : array-like or tuple, shape=[n_samples, n_features]
            predictions for provided input array. If in model selection mode,
            return a tuple ``(X_trans, y_trans)`` where ``y_trans`` is either
            ``y``, or a trunctated version to match the samples in ``X_trans``.
        """
        kwargs.pop('return_preds', None)
        return self.fit(X, y, return_preds=True)

    def predict(self, X, **kwargs):
        """Predict with fitted ensemble.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        Returns
        -------
        pred : array-like or tuple, shape=[n_samples, n_features]
            predictions for provided input array.
        """
        if not check_ensemble_build(self._backend):
            # No layers instantiated, but raise_on_exception is False
            return
        X, _ = check_inputs(X, check_level=self.array_check)
        return self._backend.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        """Predict class probabilities with fitted ensemble.

        Compatibility method for Scikit-learn. This method checks that the
        final layer has ``proba=True``, then calls the regular ``predict``
        method.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        Returns
        -------
        pred : array-like or tuple, shape=[n_samples, n_features]
            predictions for provided input array.
        """
        kwargs.pop('proba', None)
        return self.predict(X, proba=True, **kwargs)

    def _build_layer(self, estimators, indexer, preprocessing, **kwargs):
        """Build a layer from estimators and preprocessing pipelines"""
        # --- check args ---

        # Arguments that cannot be very between layers
        check_kwargs(kwargs, ['backend', 'n_jobs'])

        # Pop layer kwargs and override Sequential args
        verbose = kwargs.pop('verbose', max(self._backend.verbose - 1, 0))
        dtype = kwargs.pop('dtype', self._backend.dtype)
        propagate = kwargs.pop('propagate_features', None)
        shuffle = kwargs.pop('shuffle', self.shuffle)
        random_state = kwargs.pop('random_state', self.random_state)
        rs = kwargs.pop('raise_on_exception', self.raise_on_exception)
        if random_state:
            random_state = check_random_state(random_state).randint(0, 10000)

        # Set learner kwargs
        kwargs['verbose'] = max(verbose - 1, 0)
        kwargs['scorer'] = kwargs.pop('scorer', self.scorer)

        # Check estimator and preprocessing formatting
        group = make_group(indexer, estimators, preprocessing, kwargs)

        # --- layer ---
        name = "layer-%i" % (len(self._backend.stack) + 1)  # Start count at 1
        lyr = Layer(
            name=name, dtype=dtype, shuffle=shuffle,
            random_state=random_state, verbose=verbose,
            raise_on_exception=rs, propagate_features=propagate)
        lyr.push(group)
        return lyr

    @property
    def model_selection(self):
        """Turn model selection mode"""
        return self._model_selection

    @model_selection.setter
    def model_selection(self, model_selection):
        """Turn model selection on or off"""
        self._model_selection = model_selection
        if self._model_selection:
            self._id_train = IdTrain(self.sample_size)
        else:
            self._id_train = None

    @property
    def data(self):
        """Fit data"""
        return self._backend.data

    @property
    def verbose(self):
        """Level of printed messages"""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """Set level of printed messages"""
        self._verbose = value
        self._backend.verbose = value
