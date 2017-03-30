"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base classes for ensemble layer management.
"""

from __future__ import division, print_function

import gc
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict


from ..base import INDEXERS
from ..parallel import ParallelProcessing
from ..externals.sklearn.base import BaseEstimator
from ..externals.sklearn.validation import check_random_state
from ..utils import assert_correct_format, check_ensemble_build, \
    check_inputs, check_instances, print_time, safe_print
try:
    # Try get performance counter
    from time import perf_counter as time
except:
    # Fall back on wall clock
    from time import time


class LayerContainer(BaseEstimator):

    """Container class for layers.

    The LayerContainer class stories all layers as an ordered dictionary
    and modifies possesses a ``get_params`` method to appear as an estimator
    in the Scikit-learn API. This allows correct cloning and parameter
    updating.

    Parameters
    ----------
    layers : OrderedDict, None (default = None)
        An ordered dictionary of Layer instances. To initiate a new
        ``LayerContainer`` instance, set ``layers = None``.

    n_jobs : int (default = -1)
        Number of CPUs to use. Set ``n_jobs = -1`` for all available CPUs, and
        ``n_jobs = -2`` for all available CPUs except one, e.tc..

    raise_on_exception : bool (default = False)
        raise error on soft exceptions. Otherwise issue warning.

    verbose : int or bool (default = False)
        level of verbosity.

            - ``verbose = 0`` silent (same as ``verbose = False``)
            - ``verbose = 1`` messages at start and finish \
            (same as ``verbose = True``)
            - ``verbose = 2`` messages for each layer

        If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
        For verbosity in the layers themselves, use ``fit_params``.

    """

    def __init__(self,
                 layers=None,
                 n_jobs=-1,
                 backend='multiprocessing',
                 raise_on_exception=False,
                 verbose=False):

        # True params
        self.n_jobs = n_jobs
        self.backend = backend
        self.raise_on_exception = raise_on_exception
        self.verbose = verbose

        # Set up layer
        self._init_layers(layers)

    def add(self, estimators, cls, indexer=None, preprocessing=None, **kwargs):
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

        cls : str
            Type of layer, as defined by the estimation class to instantiate
            when processing a layer. See :mod:`mlens.ensemble` for available
            classes.

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
        # Check verbosity
        if kwargs is None:
            kwargs = {'verbose': self.verbose}
        elif 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose

        # Instantiate layer
        self.n_layers += 1
        name = "layer-%i" % self.n_layers

        lyr = Layer(estimators=estimators,
                    cls=cls,
                    indexer=indexer,
                    preprocessing=preprocessing,
                    raise_on_exception=self.raise_on_exception,
                    name=name,
                    **kwargs)

        # Attach to ordered dictionary
        self.layers[name] = lyr

        # Summarize
        self.summary[name] = self._get_layer_data(name)

        return self

    def initialize(self, X, y, job, dir=None):
        """Initialize a :class:`ParallelProcessing` engine.

        This engine will be stored as an attribute of the instance and any
        data generated by the engine will be kept alive until the ``terminate``
        method has been called.
        """
        self._processor = ParallelProcessing(self, job)
        self._processor.initialize(X, y, dir)

    def terminate(self):
        """Terminate an initialized :class:`ParallelProcessing` engine."""
        if not hasattr(self, '_processor'):
            if self.raise_on_exception:
                raise AttributeError("No initialized processor to terminate.")
            else:
                warnings.warn("No initialized processor to terminate.")
        self._processor.terminate()

        del self._processor
        gc.collect()

    def fit(self, X=None, y=None, return_preds=None, **process_kwargs):
        r"""Fit instance by calling ``predict_proba`` in the first layer.

        Similar to ``fit``, but will call the ``predict_proba`` method on
        estimators. Thus, each the ``n_test_samples * n_labels``
        prediction matrix of each estimator will be stacked and used as input
        in the subsequent layer.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for fitting and predicting.

        y : array-like of shape = [n_samples, ]
            training labels.

        return_preds : int or None (default = -1)
            How to handle the final prediction matrix. If ``return_preds=None``
            no predictions are returned. Else, an integer corresponding to the
            layer count should be passed with 0-indexing. Thus, for predictions
            from ``layer-1``, set ``return_preds=0``. If ``return_preds=-1``
            predictions from the ultimate layer is returned. Similarly,
            ``return_preds=-2`` returns the penultimate layer´s predictions.

        **process_kwargs : optional
            optional arguments to initialize processor with.

        Returns
        -----------
        out : dict
            dictionary of output data (possibly empty) generated
            through fitting. Keys correspond to layer names and values to
            the output generated by calling the layer's ``fit_function``. ::

                out = {'layer-i-estimator-j': some_data,
                       ...
                       'layer-s-estimator-q': some_data}

        X : array-like, optional
            predictions from final layer's ``fit_proba`` call.
        """
        _init = hasattr(self, '_processor')

        if not _init and (X is None):
            raise ValueError("No training data specified.")

        if self.verbose:
            pout = "stdout" if self.verbose >= 3 else "stderr"
            safe_print("Processing layers (%d)" % self.n_layers,
                       file=pout, flush=True)
            t0 = time()

        # Initialize cache
        if not _init:
            processor = ParallelProcessing(self)
            processor.initialize('fit', X, y, **process_kwargs)
        else:
            processor = self._processor

        # Fit ensemble
        try:
            processor.process()

            if self.verbose:
                print_time(t0, "Fit complete", file=pout, flush=True)

            # Generate output
            out = self._post_process(processor, return_preds)

        finally:
            # Always terminate processor unless explicitly initialized
            # before
            if not _init:
                processor.terminate()

        return out

    def predict(self, X=None, **kwargs):
        r"""Generic method for predicting through all layers in the container.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like, None (default = None)
            pass through for Scikit-learn compatibility.

        Returns
        -------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            predictions from final layer.
        """
        _init = hasattr(self, '_processor')

        if not _init and (X is None):
            raise ValueError("No training data specified.")

        if self.verbose:
            pout = "stdout" if self.verbose >= 3 else "stderr"
            safe_print("Processing layers (%d)" % self.n_layers,
                       file=pout, flush=True)
            t0 = time()

        # Initialize cache
        if not _init:
            processor = ParallelProcessing(self)
            processor.initialize('predict', X, **kwargs)
        else:
            processor = self._processor

        # Predict with ensemble
        try:
            processor.process()

            preds = processor._get_preds()

            if self.verbose:
                print_time(t0, "Prediction complete", file=pout, flush=True)

        finally:
            # Always terminate job manager unless user explicitly initialized
            if not _init:
                processor.terminate()

        return preds

    def _post_process(self, processor, return_preds):
        """Aggregate output from processing layers and collect final preds."""

        out = {}
        for layer_name, layer in self.layers.items():
            if layer.cls != 'full':
                # Layers of class 'full' make no cv predictions since they are
                # fitted on all data.
                out[layer_name] = getattr(layer, 'scores_', None)

        if return_preds is not None:
            return out, processor._get_preds(return_preds)
        else:
            return out

    def _init_layers(self, layers):
        """Return a clean ordered dictionary or copy the passed dictionary."""
        if layers is None:
            layers = OrderedDict()
            layers.clear()

        self.layers = layers
        self.n_layers = len(self.layers)

        self.summary = dict()
        for name in self.layers:
            self.summary[name] = self._get_layer_data(name)

    def _get_layer_data(self, name,
                        attr=('cls', 'n_prep', 'n_pred', 'n_est', 'cases')):
        """Utility for storing aggregate data about an added layer."""
        return {k: getattr(self.layers[name], k, None) for k in attr}

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the layers separately as individual
            parameters. If False, will return the collapsed dictionary.

        Returns
        -----------
        params : dict
            mapping of parameter names mapped to their values.
        """
        if not deep:
            return super(LayerContainer, self).get_params()

        out = {}
        for layer_name, layer in self.layers.items():
            # Add each layer
            out[layer_name] = layer

            for key, val in layer.get_params(deep=True).items():
                # Add the parameters (instances) of each layer
                out["%s__%s" % (layer_name, key)] = val

        return out


class Layer(BaseEstimator):

    r"""Layer of preprocessing pipes and estimators.

    Layer is an internal class that holds a layer and its associated data
    including an estimation procedure. It behaves as an estimator from an
    Scikit-learn API point of view.

    Parameters
    ----------
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

    cls : str
        type of layers. Should be the name of an accepted estimator class.

    indexer : instance, optional
        Indexer instance to use. Defaults to the layer class indexer
        instantiated with default settings. Required arguments depend on the
        indexer. See :mod:`mlens.base` for details.

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

    proba : bool (default = False)
        whether to call `predict_proba` on the layer´s estimators when
        predicting.

    partitions : int (default = 1)
        Number of subset-specific fits to generate from the learner library.

    raise_on_exception : bool (default = False)
        whether to raise an error on soft exceptions, else issue warning.

    verbose : int or bool (default = False)
        level of verbosity.

            - ``verbose = 0`` silent (same as ``verbose = False``)
            - ``verbose = 1`` messages at start and finish \
            (same as ``verbose = True``)
            - ``verbose = 2`` messages for each layer

        If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
        For verbosity in the layers themselves, use ``fit_params``.

    cls_kwargs : dict or None
        optional arguments to pass to the layer type class.

    Attributes
    ----------
    estimators\_ : OrderedDict, list
        container for fitted estimators, possibly mapped to preprocessing
        cases and / or folds.

    preprocessing\_ : OrderedDict, list
        container for fitted preprocessing pipelines, possibly mapped to
        preprocessing cases and / or folds.
    """

    def __init__(self,
                 estimators,
                 cls,
                 indexer=None,
                 preprocessing=None,
                 proba=False,
                 partitions=1,
                 scorer=None,
                 raise_on_exception=False,
                 name=None,
                 verbose=False,
                 cls_kwargs=None):

        assert_correct_format(estimators, preprocessing)

        self.estimators = check_instances(estimators)
        self.cls = \
            cls.strip().lower() if not cls.islower() or ' ' in cls else cls
        self.indexer = indexer if indexer is not None else INDEXERS[cls]()
        self.preprocessing = check_instances(preprocessing)
        self.cls_kwargs = cls_kwargs
        self.proba = proba
        self.partitions = partitions
        self.scorer = scorer
        self.raise_on_exception = raise_on_exception
        self.name = name
        self.verbose = verbose

        self._store_layer_data()

    def _store_layer_data(self):
        """Utility for storing aggregate attributes about the layer."""
        ests = self.estimators
        prep = self.preprocessing

        # Store layer data
        if isinstance(ests, list):
            # No preprocessing cases. Check if there is one uniform pipeline.
            self.n_prep = 0 if prep is None or len(prep) == 0 else 1
            self.n_pred = len(ests)
            self.n_est = len(ests)
            self.cases = [None]
        else:
            # Need to number of predictions by moving through each
            # case and counting estimators.
            self.n_prep = len(prep)
            self.cases = sorted(prep)

            n_pred = 0
            for case in self.cases:
                n_est = len(ests[case])
                setattr(self, '%s_n_est' % case, n_est)
                n_pred += n_est

            self.n_pred = n_pred

        if self.cls is 'subset':
            self.n_pred *= self.indexer.n_partitions

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean (default = True)
            If ``True``, will return the layers separately as individual
            parameters. If ``False``, will return the collapsed dictionary.

        Returns
        -------
        params : dict
            mapping of parameter names mapped to their values.
        """
        if not deep:
            return super(Layer, self).get_params()

        out = {}
        for step in [self.preprocessing, self.estimators]:
            if isinstance(step, dict):
                # Mapped preprocessing: need to control for case membership
                for case, instances in step.items():
                    for instance_name, instance in instances:
                        out["%s-%s" % (case, instance_name)] = instance
            else:
                # Simple named list of estimators / transformers
                for instance_name, instance in step:
                    out[instance_name] = instance
        return out


class BaseEnsemble(BaseEstimator):

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
                 n_jobs=-1,
                 layers=None,
                 array_check=2):

        self.shuffle = shuffle
        self.random_state = random_state
        self.scorer = scorer
        self.raise_on_exception = raise_on_exception
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.layers = layers
        self.array_check = array_check

    @abstractmethod
    def add(self, estimators, preprocessing=None, **kwargs):
        """Interface for adding a layer."""
        pass

    def _add(self,
             estimators,
             cls,
             indexer,
             preprocessing=None,
             **kwargs):
        """Auxiliary method to be called by ``add``.

        Checks if the ensemble's :class:`LayerContainer` is instantiated and
        if not, creates one.

        See Also
        --------
        :class:`LayerContainer`

        Returns
        -------
        self :
            instance with instantiated layer attached.
        """
        if getattr(self, 'layers', None) is None:
            raise_on_exception = getattr(self, 'raise_on_exception', True)
            n_jobs = getattr(self, 'n_jobs', -1)
            self.layers = LayerContainer(n_jobs=n_jobs,
                                         raise_on_exception=raise_on_exception)

        self.layers.add(estimators=estimators,
                        cls=cls,
                        indexer=indexer,
                        preprocessing=preprocessing,
                        scorer=self.scorer,
                        **kwargs)
        return self

    def fit(self, X, y=None):
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
            # No layers instantiated. Return vacuous fit.
            return self

        X, y = check_inputs(X, y, self.array_check)

        if self.shuffle:
            r = check_random_state(self.random_state)
            r.shuffle(X)
            r.shuffle(y)

        self.scores_ = self.layers.fit(X, y)

        return self

    def predict(self, X):
        """Predict with fitted ensemble.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like, None (default = None)
            pass through for scikit-learn compatibility.

        Returns
        -------
        y_pred : array-like, shape=[n_samples, ]
            predictions for provided input array.
        """
        if not check_ensemble_build(self):
            # No layers instantiated, but raise_on_exception is False
            return

        X, _ = check_inputs(X, check_level=self.array_check)

        if self.shuffle:
            r = check_random_state(self.random_state)
            r.shuffle(X)

        y = self.layers.predict(X)

        if y.shape[1] == 1:
            # The meta estimator is treated as a layer and thus a prediction
            # matrix with shape [n_samples, 1] is created. Ravel before return
            y = y.ravel()

        return y
