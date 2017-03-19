"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base classes for ensemble layer management.
"""

from __future__ import division, print_function

from collections import OrderedDict
from abc import ABCMeta, abstractmethod

from ..base import check_instances
from ..parallel import ParallelProcessing
from ..utils import (check_is_fitted, assert_correct_layer_format,
                     print_time, safe_print, check_layer_output)
from ..utils.exceptions import (LayerSpecificationWarning,
                                LayerSpecificationError,
                                NotFittedError)

from sklearn.base import BaseEstimator


import warnings
import gc
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

    n_layers : int (default = None)
        number of layers instantiated. Automatically set, normally there is no
        reason to fiddle with this parameter.

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

    __lim__ = 60   # Time limit for trying to find transformers in cache
    __sec__ = 0.1  # Time interval for checking if transformers exists in cache

    def __init__(self,
                 layers=None,
                 n_jobs=-1,
                 raise_on_exception=False,
                 verbose=False):

        # True params
        self.n_jobs = n_jobs
        self.raise_on_exception = raise_on_exception
        self.verbose = verbose

        # Set up layer
        self._init_layers(layers)

    def add(self, fit_function, predict_function, estimators,
            preprocessing=None, fit_params=None, predict_params=None,
            in_place=True, **kwargs):
        """Method for adding a layer.

        Parameters
        -----------
        fit_function : function
            Function used for fitting the layer. The ``fit_function`` must
            have the following API::

                (estimators_, preprocessing_, out) = fit_function(
                layer_instance, X, y, fit_params)

            where ``estimators_`` and ``preprocessing_`` are generic objects
            holding fitted instances. The ``LayerContainer`` class is agnostic
            to the object type, and only passes these along to the
            ``predict_function`` during a ``predict`` call. The ``out``
            variable must be a tuple of output data that at least contains
            the training data for the subsequent layer as its **first**
            element: ``out = (X, ...)``.

        predict_function : function
            Function used for generating prediction with the fitted layer.
            The predict_function must have the following API::

                X_pred = predict_function(layer_instance, X, y, predict_params)

            where ``X_pred`` is an array-like of shape [n_samples,
            n_fitted_estimators] of predictions.

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

        fit_params : dict, tuple or None (default = None)
            optional arguments passed to ``fit_function``.

        predict_params : dict, tuple or None (default = None)
            optional arguments passed to ``predict_function``.

        in_place : bool (default = True)
            whether to return the instance (i.e. ``self``).

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
        lyr = Layer(estimators=estimators,
                    preprocessing=preprocessing,
                    fit_function=fit_function,
                    fit_params=fit_params,
                    predict_function=predict_function,
                    predict_params=predict_params,
                    raise_on_exception=self.raise_on_exception,
                    **kwargs)

        self.n_layers += 1
        name = "layer-%i" % self.n_layers

        # Attached to ordered dictionary
        self.layers[name] = lyr

        self._store_layer_data(name)

        if not in_place:
            return self

    def initialize(self, X, y, dir=None):
        """Initialize a :class:`ParallelProcessing` engine.

        This engine will be stored as an attribute of the instance and any
        data generated by the engine will be kept alive until the ``terminate``
        method has been called.
        """
        self._processor = ParallelProcessing(self)
        self._processor.initialize(X, y, dir)

    def terminate(self):
        """Terminate an initialized :class:`ParallelProcessing` engine."""
        if not hasattr(self, '_processor'):
            if self.raise_on_exception:
                raise AttributeError("No initialized processor to terminate.")
            else:
                warnings.warn("No initialized processor to terminate.")
        self._processor.terminate()
        delattr(self, '_processor')
        gc.collect()

    def fit(self, X=None, y=None, return_final=True, **process_kwargs):
        r"""Generic method for fitting all layers in the container.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for fitting and predicting.

        y : array-like of shape = [n_samples, ]
            training labels.

        return_final : str or None, optional (default = None)
            How to handle the final prediction matrix. If ``return_final=None``
            the prediction matrix will not be returned. If
            ``return_final='mmap'``, a :class:`numpy.memmap` pointing to the
            final  prediction matrix is returned. If ``return_final='array'``,
            a :class:`numpy.ndarray` is returned.

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
            predictions from final layer's ``fit`` call.
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
            processor.initialize(X, y, **process_kwargs)
        else:
            processor = self._processor

        # Fit ensemble
        try:
            processor.process('fit')

            if self.verbose:
                print_time(t0, "Fit complete", file=pout, flush=True)

            # Generate output
            out = self._post_process(processor, return_final)

        finally:
            # Always terminate processor unless explicitly initialized before
            if not _init:
                processor.terminate()

        return out

    def predict(self, X=None, y=None, **process_kwargs):
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
            processor.initialize(X, y, **process_kwargs)
        else:
            processor = self._processor

        # Predict with ensemble
        try:
            processor.process('predict')

            preds = processor._get_final_preds()

            if self.verbose:
                print_time(t0, "Prediction complete", file=pout, flush=True)

        finally:
            # Always terminate processor unless explicitly initialized before
            if not _init:
                processor.terminate()

        return preds

    def _post_process(self, processor, return_final):
        """Aggregate output from processing layers and collect final preds."""

        out = {}
        for layer_name, layer in self.layers.items():
            out[layer_name] = getattr(layer, 'fit_data_', None)

        if return_final:
            return [out, processor._get_final_preds()]
        else:
            return [out]

    def _init_layers(self, layers):
        """Return a clean ordered dictionary or copy the passed dictionary."""
        if layers is None:
            layers = OrderedDict()
            layers.clear()

        self.layers = layers
        self.n_layers = len(self.layers)

        self._layer_data = dict()
        for layer_name in self.layers:
            self._store_layer_data(layer_name)

    def _store_layer_data(self, name):
        """Utility for storing aggregate data about an added layer."""
        self._layer_data[name] = self.layers[name]._layer_data

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

    Layer is an internal class that holds a layer and all associated layer
    specific methods. It behaves as an estimator from an Scikit-learn API
    point of view.

    Parameters
    ----------
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

    fit_params : dict, tuple or None (default = None)
        optional arguments passed to ``fit_function``.

    predict_params : dict, tuple or None (default = None)
        optional arguments passed to ``predict_function``.

    fit_function : function
        Function used for fitting the layer. The ``fit_function`` must
        have the following API::

            (estimators_, preprocessing_, out) = fit_function(
            layer_instance, X, y, fit_params)

        where ``estimators_`` and ``preprocessing_`` are generic objects
        holding fitted instances. The ``LayerContainer`` class is agnostic
        to the object type, and only passes these along to the
        ``predict_function`` during a ``predict`` call. The ``out``
        variable must be a tuple of output data that at least contains
        the training data for the subsequent layer as its **first**
        element: ``out = (X, ...)``.

    predict_function : function
        Function used for generating prediction with the fitted layer.
        The predict_function must have the following API::

            X_pred = predict_function(layer_instance, X, y, predict_params)

        where ``X_pred`` is an array-like of shape [n_samples,
        n_fitted_estimators] of predictions.

    fit_params : dict, tuple or None (default = None)
        optional keyword arguments passed to ``fit_function``.

    predict_params : dict, tuple or None (default = None)
        optional keyword arguments passed to ``predict_function``.

    indexer : obj, optional
        indexing object to be used for slicing data during fitting.

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
                 fit_function,
                 predict_function,
                 preprocessing=None,
                 fit_params=None,
                 predict_params=None,
                 indexer=None,
                 raise_on_exception=False,
                 verbose=False):

        assert_correct_layer_format(estimators, preprocessing)

        self.estimators = check_instances(estimators)
        self.preprocessing = check_instances(preprocessing)
        self.fit_function = fit_function
        self.predict_function = predict_function
        self.fit_params = self._format_params(fit_params)
        self.predict_params = self._format_params(predict_params)
        self.raise_on_exception = raise_on_exception
        self.indexer = indexer
        self.verbose = verbose

        self._store_layer_data()

    def fit(self, X, y, P, temp_folder, parallel):
        """Generic method for fitting and storing layer data.

        Any output data created during estimation is stored in the
        ``fit_data_`` attribute.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like of shape = [n_samples, ]
            training lables.

        P : array-like of shape = [n_prediction_samples, n_estimators]
            prediction matrix to fill with prediction.

        temp_folder : str
            path to estimation folder to use during fitting.

        parallel : inst
            a :class:`joblib.Parallel` instance to use for parallel estimation.

        """
        self.estimators_, self.preprocessing_, self.fit_data_ = \
            self.fit_function(self, X, y, P, temp_folder, parallel)

    def predict(self, X, P, parallel):
        """Generic method for predicting with fitted layer.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : None, array-like of shape = [n_samples, ]
            pass-through for Scikit-learn API.

        parallel : inst
            a :class:`joblib.Parallel` instance to use for parallel estimation.

        Returns
        -------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            predictions from fitted estimators in layer.
        """
        self._check_fitted()
        return self.predict_function(self, X, P, parallel,
                                     **self.predict_params)

    def _check_fitted(self):
        """Utility function for checking that fitted estimators exist."""
        check_is_fitted(self, "estimators_")

        # Check that there is at least one fitted estimator
        if isinstance(self.estimators_, (list, tuple, set)):
            empty = len(self.estimators_) == 0
        elif isinstance(self.estimators_, dict):
            empty = any([len(e) == 0 for e in self.estimators_.values()])
        else:
            # Cannot determine shape of estimators, skip check
            return

        if empty:
            raise NotFittedError("Cannot predict as no estimators were"
                                 "successfully fitted.")

    @staticmethod
    def _format_params(params):
        """Check that a fit or predict parameters are formatted as kwargs."""
        if params is None:
            return {}
        elif not isinstance(params, dict):

            msg = ("Wrong optional params type. 'fit_params' "
                   " and 'predict_params' must de type 'dict' "
                   "(type: %s).")

            raise LayerSpecificationError(msg % type(params))

        return params if params is not None else {}

    def _store_layer_data(self):
        """Utility for storing aggregate data about the layer."""
        self._layer_data = {}
        ests = self.estimators
        prep = self.preprocessing

        # Store layer data
        if isinstance(ests, list):
            # No preprocessing cases. Check if there is one uniform pipeline.
            self._layer_data['n_prep'] = \
                0 if prep is None or len(prep) == 0 else 1

            self._layer_data['n_pred'] = len(ests)
            self._layer_data['n_est'] = len(ests)
            self._layer_data['cases'] = [None]
        else:
            # Need to number of predictions by moving through each
            # case and counting estimators.
            self._layer_data['n_prep'] = len(prep)

            self._layer_data['cases'] = sorted(prep)

            n_pred = 0
            for case in self._layer_data['cases']:
                n_est = len(ests[case])
                self._layer_data['%s-n_est' % case] = n_est
                n_pred += n_est

            self._layer_data['n_pred'] = n_pred

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
    def add(self, estimators, preprocessing=None):
        """Interface for adding a layer."""
        pass

    @abstractmethod
    def fit(self, X, y):
        """Method for fitting all layers."""
        pass

    @abstractmethod
    def predict(self, X, y=None):
        """Method for predicting with all layers."""
        pass


    def __set_lim__(self, l):
        """Set the time limit for waiting on preprocessing to complete."""
        if not hasattr(self, 'layers'):
            raise AttributeError("No layer's attached to ensemble. Cannot"
                                 "set time limit.")
        self.layers.__lim__ = l

    def __get_lim__(self):
        """Set the time limit for waiting on preprocessing to complete."""
        if not hasattr(self, 'layers'):
            raise AttributeError("No layer's attached to ensemble. Cannot"
                                 "fetch time limit.")
        return self.layers.__lim__

    def __set_sec__(self, s):
        """Set time interval for checking if preprocessing has completed."""
        if not hasattr(self, 'layers'):
            raise AttributeError("No layer's attached to ensemble. Cannot"
                                 "set time interval.")
        self.layers.__sec__ = s

    def __get_sec__(self):
        """Set the time limit for waiting on preprocessing to complete."""
        if not hasattr(self, 'layers'):
            raise AttributeError("No layer's attached to ensemble. Cannot"
                                 "fetch time interval.")
        return self.layers.__sec__

    def _add(self,
             estimators,
             fit_function,
             predict_function,
             fit_params=None,
             predict_params=None,
             preprocessing=None,
             **kwargs):
        """Method for adding a layer.

        Parameters
        -----------
        fit_function : function
            Function used for fitting the layer. The ``fit_function`` must
            have the following API::

                (estimators_, preprocessing_, out) = fit_function(
                layer_instance, X, y, fit_params)

            where ``estimators_`` and ``preprocessing_`` are generic objects
            holding fitted instances. The ``LayerContainer`` class is agnostic
            to the object type, and only passes these along to the
            ``predict_function`` during a ``predict`` call. The ``out``
            variable must be a tuple of output data that at least contains
            the training data for the subsequent layer as its **first**
            element: ``out = (X, ...)``.

        predict_function : function
            Function used for generating prediction with the fitted layer.
            The predict_function must have the following API::

                X_pred = predict_function(layer_instance, X, y, predict_params)

            where ``X_pred`` is an array-like of shape [n_samples,
            n_fitted_estimators] of predictions.

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

             The lists for each dictionary entry can be any of 'option_1',
             'option_2' and ``option_3``.

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

             The lists for each dictionary entry can be any of 'option_1',
             'option_2' and ``option_3``.

        fit_params : dict, tuple or None (default = None)
            optional arguments passed to ``fit_function``.

        predict_params : dict, tuple or None (default = None)
            optional arguments passed to ``predict_function``.

        **kwargs : optional
            optional parameters to pass to layer at instantiation.

        Returns
        -------
        self : instance, optional
            if ``in_place = True``, returns ``self`` with the layer
            instantiated.
        """
        if getattr(self, 'layers', None) is None:
            raise_on_exception = getattr(self, 'raise_on_exception', False)
            n_jobs = getattr(self, 'n_jobs', -1)
            self.layers = LayerContainer(n_jobs=n_jobs,
                                         raise_on_exception=raise_on_exception)

        self.layers.add(estimators=estimators,
                        preprocessing=preprocessing,
                        fit_function=fit_function,
                        fit_params=fit_params,
                        predict_function=predict_function,
                        predict_params=predict_params,
                        **kwargs)
        return self

    def _fit_layers(self, X, y):
        r"""Generic method for fitting all layers in the ensemble.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for fitting and predicting.

        y : array-like of shape = [n_samples, ]
            training labels.

        Returns
        -----------
        out : dict
            dictionary of output data (possibly empty) generated
            through fitting. Keys correspond to layer names and values to
            the output generated by calling the layer"s ``fit_function``. ::

                out = {'layer-i-estimator-j': some_data,
                       ...
                       'layer-s-estimator-q': some_data}

        X : array-like (optional)
            if ``return_final = True``, returns predictions from final layer's
            ``fit`` call.
        """
        return self.layers.fit(X, y)

    def _predict_layers(self, X, y):
        r"""Generic method for predicting through all layers in the ensemble.

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
        return self.layers.predict(X, y)
