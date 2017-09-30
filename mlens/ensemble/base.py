"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
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
from ..parallel import Layer, ParallelProcessing
from ..externals.sklearn.base import BaseEstimator
from ..externals.sklearn.validation import check_random_state
from ..utils import (check_ensemble_build,
                     check_layers,
                     check_inputs,
                     print_time,
                     safe_print)
from ..utils.exceptions import LayerSpecificationWarning
from ..metrics import Data
try:
    # Try get performance counter
    from time import perf_counter as time
except ImportError:
    # Fall back on wall clock
    from time import time


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
        safe_print("\n%s %d layers" % (start_message, lc.n_layers),
                   file=f, flush=True)
        if lc.verbose >= 5:
            safe_print("""[INFO] n_jobs = %i
[INFO] backend = %r
[INFO] start_method = %r
[INFO] cache = %r
""" % (lc.n_jobs, lc.backend, config.START_METHOD, config.TMPDIR),
                       file=f, flush=True)

    t0 = time()
    return f, t0


###############################################################################
class Sequential(BaseEstimator):

    r"""Container class for layers.

    The Sequential class stories all layers as an ordered dictionary
    and modifies possesses a ``get_params`` method to appear as an estimator
    in the Scikit-learn API. This allows correct cloning and parameter
    updating.


    Parameters
    ----------
    layers : list, optional (default = None)
        list of layers to instantiate with.

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

    def __init__(self,
                 layers=None,
                 n_jobs=-1,
                 backend=None,
                 raise_on_exception=False,
                 dtype=None,
                 verbose=False):

        # True params
        self.n_jobs = n_jobs
        self.backend = backend if backend is not None else config.BACKEND
        self.raise_on_exception = raise_on_exception
        self._verbose = verbose

        # Set up layer
        self.dtype = dtype if dtype else config.DTYPE
        self._init_layers(layers)

    def __call__(self, *layers):
        """Add layers to instance

        Parameters
        ----------
        *layers : iterable
            list of layers to add to sequential
        """
        check_layers(layers)
        for lyr in layers:
            if lyr.name in [lr.name for lr in self.layers]:
                raise ValueError("Layer name exists in stack. "
                                 "Rename layers before attempting to push.")

            self.n_layers += 1
            self.layers.append(lyr)
            self._get_layer_data(lyr)

            attr = lyr.name.replace('-', '_').replace(' ', '').strip()
            setattr(self, attr, lyr)

        return self

    def __iter__(self):
        """Generator for layers"""
        for layer in self.layers:
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
        f, t0 = print_job(self, "Fitting")

        with ParallelProcessing(self) \
                as manager:
            out = manager.process('fit', X, y, **kwargs)

        if self.verbose:
            print_time(t0, "{:<35}".format("Fit complete"), file=f, flush=True)

        if out is None:
            return self
        return out

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
        return_preds = kwargs.pop('return_preds', True)
        with ParallelProcessing(self) as manager:
            preds = manager.process(
                job, X, return_preds=return_preds, **kwargs)

        return preds

    def _init_layers(self, layers):
        """Initialize layers"""
        if layers is None:
            layers = list()
        elif layers.__class__.__name__.lower() == 'layer':
            layers = [layers]

        self.layers = layers
        self.n_layers = len(self.layers)

        self.summary = dict()
        for layer in self.layers:
            self._get_layer_data(layer)

    def _get_layer_data(self, layer,
                        attr=('cls', 'n_prep', 'n_pred', 'n_est', 'cases')):
        """Utility for storing aggregate data about an added layer."""
        self.summary[layer.name] = {k: getattr(layer, k, None) for k in attr}

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            whether to return nested parameters.
        """
        out = super(Sequential, self).get_params(deep=deep)

        if not deep:
            return out

        for layer in self.layers:
            out[layer.name] = layer
            for key, val in layer.get_params(deep=True).items():
                out['%s__%s' % (layer.name, key)] = val
        return out

    @property
    def verbose(self):
        """Adjust the level of verbosity."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
        for layer in self.layers:
            layer.verbose = verbose

    @property
    def data(self):
        """Ensemble data"""
        out = list()
        for layer in self.layers:
            d = layer.raw_data
            if not d:
                continue
            out.extend([('%s  %s' % (layer.name, k), v) for k, v in d])
        return Data(out)


###############################################################################
class BaseEnsemble(Sequential):

    """BaseEnsemble class.

    Core ensemble class methods used to add ensemble layers and manipulate
    parameters.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 shuffle=False,
                 random_state=None,
                 scorer=None,
                 raise_on_exception=True,
                 verbose=False,
                 dtype=None,
                 n_jobs=-1,
                 layers=None,
                 array_check=2,
                 backend=None
                 ):
        super(BaseEnsemble, self).__init__(
            layers=layers,
            n_jobs=n_jobs,
            backend=backend,
            raise_on_exception=raise_on_exception,
            dtype=dtype,
            verbose=verbose)
        self.shuffle = shuffle
        self.random_state = random_state
        self.scorer = scorer
        self.array_check = array_check

    def _add(self,
             estimators,
             indexer,
             preprocessing=None,
             **kwargs):
        """Method for adding a layer.

        Parameters
        -----------
        estimators: dict of lists or list
            estimators constituting the layer. If ``preprocessing`` is
            ``None`` or ``list``, ``estimators`` should be a ``list``.
            The list can either contain estimator instances,
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
        self : instance, optional
            if ``in_place = True``, returns ``self`` with the layer
            instantiated.
        """
        if estimators.__class__.__name__.lower() == 'layer':
            return self(estimators)

        # Override Sequential arguments if layer-specific args are found
        verbose = kwargs.pop('verbose', self.verbose)
        raise_ = kwargs.pop('raise_on_exception', self.raise_on_exception)
        dtype = kwargs.pop('dtype', self.dtype)
        scorer = kwargs.pop('scorer', self.scorer)

        # Arguments that cannot be very between layers in a given fit -
        # use Sequential params
        check_kwargs(kwargs, ['backend', 'n_jobs'])

        # Check shuffle
        shuffle = kwargs.pop('shuffle', self.shuffle)
        random_state = kwargs.pop('random_state', self.random_state)
        if random_state:
            random_state = check_random_state(random_state).randint(0, 10000)

        # Instantiate layer
        name = "layer-%i" % (self.n_layers + 1)  # Start count at 1
        lyr = Layer(estimators=estimators,
                    indexer=indexer,
                    name=name,
                    preprocessing=preprocessing,
                    dtype=dtype,
                    scorer=scorer,
                    shuffle=shuffle,
                    random_state=random_state,
                    verbose=max(verbose - 1, 0),
                    raise_on_exception=raise_,
                    **kwargs)

        # Add layer to stack
        return self(lyr)

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
        if not check_ensemble_build(self):
            # No layers instantiated, but raise_on_exception is False
            return self

        X, y = check_inputs(X, y, self.array_check)

        return super(BaseEnsemble, self).fit(X, y, **kwargs)

    def transform(self, X, **kwargs):
        """Transform with fitted ensemble.

        Replicates cross-validated prediction process from training.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        Returns
        -------
        y_pred : array-like, shape=[n_samples, n_features]
            predictions for provided input array.
        """
        if not check_ensemble_build(self):
            # No layers instantiated, but raise_on_exception is False
            return
        X, _ = check_inputs(X, check_level=self.array_check)
        return super(BaseEnsemble, self).transform(X, **kwargs)

    def predict(self, X, **kwargs):
        """Predict with fitted ensemble.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        Returns
        -------
        y_pred : array-like, shape=[n_samples, ]
            predictions for provided input array.
        """
        if not check_ensemble_build(self):
            # No layers instantiated, but raise_on_exception is False
            return
        X, _ = check_inputs(X, check_level=self.array_check)
        y = super(BaseEnsemble, self).predict(X, **kwargs)

        if y.shape[1] == 1:
            # The meta estimator is treated as a layer and thus a prediction
            # matrix with shape [n_samples, 1] is created. Ravel before return
            y = y.ravel()
        return y

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
        y_pred : array-like, shape=[n_samples, n_classes]
            predicted class membership probabilities for provided input array.
        """
        lyr = self.layers.layers[-1]
        if not getattr(lyr, 'proba', False):
            raise ValueError("Cannot use 'predict_proba' if final layer"
                             "does not have 'proba=True'.")
        return self.predict(X, **kwargs)
