"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base classes for ensemble layer management.
"""

from __future__ import division, print_function

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from .. import config
from ..base import INDEXERS
from ..parallel import ParallelProcessing
from ..externals.sklearn.base import BaseEstimator
from ..externals.sklearn.validation import check_random_state
from ..utils import assert_correct_format, check_ensemble_build, \
    check_inputs, check_instances, print_time, safe_print
try:
    # Try get performance counter
    from time import perf_counter as time
except ImportError:
    # Fall back on wall clock
    from time import time


class LayerContainer(BaseEstimator):

    r"""Container class for layers.

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

        If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
        For verbosity in the layers themselves, use ``fit_params``.

    """

    def __init__(self,
                 layers=None,
                 n_jobs=-1,
                 backend=None,
                 raise_on_exception=False,
                 verbose=False):

        # True params
        self.n_jobs = n_jobs
        self.backend = backend if backend is not None else config.BACKEND
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
        self.layer_names.append(name)

        # Summarize
        self.summary[name] = self._get_layer_data(name)

        return self

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
            predictions from the ultimate layer is returned.

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
        if self.verbose:
            pout = "stdout" if self.verbose >= 3 else "stderr"
            safe_print("Fitting layers (%d)" % self.n_layers,
                       file=pout, flush=True, end="\n\n")
            t0 = time()

        # Initialize cache
        processor = ParallelProcessing(self)
        processor.initialize('fit', X, y, **process_kwargs)

        # Fit ensemble
        try:
            processor.process()

            if self.verbose:
                print_time(t0, "Fit complete", file=pout, flush=True)

            # Generate output
            out = self._post_process(processor, return_preds)

        finally:
            # Always terminate processor
            processor.terminate()

        return out

    def predict(self, X=None, *args, **kwargs):
        r"""Generic method for predicting through all layers in the container.

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
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            predictions from final layer.
        """
        return self._predict(X, 'predict', *args, **kwargs)

    def transform(self, X=None, *args, **kwargs):
        """Generic method for reproducing predictions of the ``fit`` call.

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
        return self._predict(X, 'transform', *args, **kwargs)

    def _predict(self, X, job, *args, **kwargs):
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
        if self.verbose:
            pout = "stdout" if self.verbose >= 3 else "stderr"
            safe_print("Processing layers (%d)" % self.n_layers,
                       file=pout, flush=True, end="\n\n")
            t0 = time()

        # Initialize cache
        processor = ParallelProcessing(self)
        processor.initialize(job, X, *args, **kwargs)

        # Predict with ensemble
        try:
            processor.process()

            preds = processor.get_preds()

            if self.verbose:
                print_time(t0, "Done", file=pout, flush=True)

        finally:
            # Always terminate job manager unless user explicitly initialized
            processor.terminate()

        return preds

    def _post_process(self, processor, return_preds):
        """Aggregate output from processing layers and _collect final preds."""
        out = {'score_mean': {}, 'score_std': {}}
        for layer_name, layer in self.layers.items():
            if layer.cls != 'full':
                # Layers of class 'full' make no cv predictions
                layer_scores = getattr(layer, 'scores_', None)

                if layer_scores is not None:
                    for est_name, s in layer_scores.items():
                        out['score_mean'][(layer_name, est_name)] = s[0]
                        out['score_std'][(layer_name, est_name)] = s[1]

        if len(out['score_mean']) == 0:
            out = None

        if return_preds is not None:
            if isinstance(return_preds, bool):
                # Safeguard against boolean argument
                return_preds = -1
            return out, processor.get_preds(return_preds)
        else:
            return out

    def _init_layers(self, layers):
        """Return a clean ordered dictionary or copy the passed dictionary."""
        if layers is None:
            layers = OrderedDict()
            layers.clear()

        self.layers = layers
        self.layer_names = list()
        self.n_layers = len(self.layers)

        self.summary = dict()
        for name in self.layers:
            self.summary[name] = self._get_layer_data(name)
            self.layer_names.append(name)

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
        whether to call `predict_proba` on the estimators in the layer when
        predicting.

    partitions : int (default = 1)
        Number of subset-specific fits to generate from the learner library.

    propagate_features : list, optional
        Features to propagate from the input array to the output array.
        Carries input features to the output of the layer, useful for
        propagating original data through several stacked layers. Propagated
        features are stored in the left-most columns.

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

    dtype : numpy dtype class, default = :class:`numpy.float32`
        dtype format of prediction array.

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
                 propagate_features=None,
                 scorer=None,
                 raise_on_exception=False,
                 name=None,
                 dtype=None,
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
        self._predict_attr = 'predict' if not proba else 'predict_proba'
        self.partitions = partitions
        self.propagate_features = propagate_features
        self.scorer = scorer
        self.raise_on_exception = raise_on_exception
        self.name = name
        self.dtype = dtype if dtype is not None else config.DTYPE
        self.verbose = verbose

        self._store_layer_data()

    def _store_layer_data(self):
        """Utility for storing aggregate attributes about the layer."""
        ests = self.estimators
        prep = self.preprocessing

        # Store feature propagation data
        if self.propagate_features:
            if not isinstance(self.propagate_features, list):
                raise ValueError("propagate features expected list, got %s" %
                                 self.propagate_features.__class__)
            self.n_feature_prop = len(self.propagate_features)
        else:
            self.n_feature_prop = 0

        # Store layer estimator data
        if isinstance(ests, list):
            # No preprocessing cases. Check if there is one uniform pipeline.
            self.n_prep = 0 if prep is None or len(prep) == 0 else 1
            self.n_pred = len(ests)
            self.n_est = len(ests)
            self.cases = [None]
        else:
            # Get the number of predictions by moving through each
            # case and count estimators.
            self.n_prep = len(prep)
            self.cases = sorted(prep)

            n_pred = 0
            for case in self.cases:
                n_est = len(ests[case])
                setattr(self, '%s_n_est' % case, n_est)
                n_pred += n_est

            self.n_pred = self.n_est = n_pred

        if self.cls is 'subset':
            self.n_pred *= self.indexer.n_partitions
            self.n_prep *= self.indexer.n_partitions

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

        out = dict()
        for step in [self.preprocessing, self.estimators]:
            if isinstance(step, dict):
                # Mapped preprocessing: need to control for case membership
                for case, instances in step.items():
                    for instance_name, instance in instances:
                        out["%s-%s" % (case, instance_name)] = instance
                        # Get instance parameters
                        for k, v in instances.get_params().items():
                            out["%s-%s__%s" % (case, instance_name, k)] = v

            else:
                # Simple named list of estimators / transformers
                for instance_name, instance in step:
                    out[instance_name] = instance
                    # Get instance parameters
                    for k, v in instance.get_params().items():
                        out["%s__%s" % (instance_name, k)] = v

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
                 array_check=2,
                 backend=None
                 ):

        self.shuffle = shuffle
        self.random_state = random_state
        self.scorer = scorer
        self.raise_on_exception = raise_on_exception
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.layers = layers
        self.array_check = array_check
        self.backend = backend if backend is not None else config.BACKEND

    def set_verbosity(self, verbose):
        """Adjust the level of verbosity."""
        self.verbose = verbose
        self.layers.verbose = verbose
        for layer in self.layers.layers.values():
            layer.verbose = verbose

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
        # Check if a Layer Container instance is initialized
        if getattr(self, 'layers', None) is None:
            self.layers = LayerContainer(
                            n_jobs=self.n_jobs,
                            raise_on_exception=self.raise_on_exception,
                            backend=self.backend,
                            verbose=self.verbose)

        # Add layer to Layer Container
        self.layers.add(estimators=estimators,
                        cls=cls,
                        indexer=indexer,
                        preprocessing=preprocessing,
                        scorer=self.scorer,
                        **kwargs)

        # Check parameter comparability
        if 'proba' in kwargs:
            scorer = getattr(self, 'scorer', None)
            if kwargs['proba'] and scorer:
                raise ValueError("Cannot score probability-based predictions."
                                 "Set either ensemble parameter 'scorer' to "
                                 "None or layer parameter 'Proba' to False.")

        # Set the layer as an attribute of the ensemble
        lyr = list(self.layers.layers)[-1]
        attr = lyr.replace('-', '_').replace(' ', '').strip()

        setattr(self, attr, self.layers.layers[lyr])

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
            idx = r.permutation(X.shape[0])
            X, y = X[idx], y[idx]

        self.scores_ = self.layers.fit(X, y)

        return self

    def predict(self, X):
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

        if self.shuffle:
            r = check_random_state(self.random_state)
            idx = r.permutation(X.shape[0])
            X = X[idx]

        y = self.layers.predict(X)

        if y.shape[1] == 1:
            # The meta estimator is treated as a layer and thus a prediction
            # matrix with shape [n_samples, 1] is created. Ravel before return
            y = y.ravel()

        return y

    def predict_proba(self, X):
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
        meta_name = list(self.layers.layers)[-1]
        lyr = self.layers.layers[meta_name]

        if not getattr(lyr, 'proba', False):
            raise ValueError("Cannot use 'predict_proba' if final layer"
                             "does not have 'proba=True'.")
        return self.predict(X)
