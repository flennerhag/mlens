"""

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base classes for ensemble layer management.
"""

from __future__ import division, print_function
from collections import OrderedDict

from sklearn.base import BaseEstimator
from ..base import check_instances
from ..utils import (check_is_fitted, assert_correct_layer_format,
                     print_time, safe_print, check_layer_output)

from ..utils.exceptions import (LayerSpecificationWarning,
                                LayerSpecificationError,
                                NotFittedError)

import warnings
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

    raise_on_exception : bool (default = False)
        raise error on soft exceptions. Otherwise issue warning.
    """

    def __init__(self, layers=None, n_layers=0, raise_on_exception=False):
        self.layers = self._init_layers(layers)
        self.n_layers = self._check_n_layers(n_layers)
        self.raise_on_exception = raise_on_exception

    def add(self, fit_function, predict_function, estimators,
            preprocessing=None, fit_params=None, predict_params=None,
            in_place=True):
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

        in_place: bool (default = True)
            whether to return the instance (i.e. ``self``).

        Returns
        ----------
        self : instance, optional
            if ``in_place = True``, returns ``self`` with the layer
            instantiated.
        """
        # Instantiate layer
        lyr = Layer(estimators=estimators,
                    preprocessing=preprocessing,
                    fit_function=fit_function,
                    fit_params=fit_params,
                    predict_function=predict_function,
                    predict_params=predict_params,
                    raise_on_exception=self.raise_on_exception)

        self.n_layers += 1

        # Attached to ordered dictionary
        self.layers["layer-%i" % self.n_layers] = lyr

        if not in_place:
            return self

    def fit(self, X, y, return_final, verbose):
        """Generic method for fitting all layers in the container.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for fitting and predicting.

        y : array-like of shape = [n_samples, ]
            training labels.

        return_final : bool (default = True)
            whether to return the final training input. In this case,
            the output of the ``fit`` method is a tuple ``(out, X)``.
            Otherwise, ``fit`` returns ``out``.

        verbose : int or bool (default = False)
            level of verbosity.

                - ``verbose = 0`` silent (same as ``verbose = False``)
                - ``verbose = 1`` messages at start and finish \
                (same as ``verbose = True``)
                - ``verbose = 2`` messages for each layer

            If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
            For verbosity in the layers themselves, use ``fit_params``.

        Returns
        -----------
        out : dict
            dictionary of output data (possibly empty) generated
            through fitting. Keys correspond to layer names and values to
            the output generated by calling the layer's ``fit_function``. ::

                out = {'layer-i-estimator-j': some_data,
                       ...
                       'layer-s-estimator-q': some_data}

        X : array-like (optional)
            predictions from final layer's ``fit`` call.
        """
        if verbose:
            pout = "stdout" if verbose >= 3 else "stderr"
            safe_print("Fitting all layers (%d)" % self.n_layers,
                       file=pout, flush=True)
            t0 = time()

        fit_tup = (X,)  # initiate a tuple of fit outputs for first layer
        out = {}
        for layer_name, layer in self.layers.items():

            if verbose > 1:
                safe_print("[%s] fitting" % layer_name, file=pout, flush=True)
                ti = time()

            fit_tup = layer.fit(fit_tup[0], y)
            out[layer_name] = fit_tup[1:]

            check_layer_output(layer, layer_name, self.raise_on_exception)

            if verbose > 1:
                print_time(ti, "[%s] done" % layer_name, file=pout, flush=True)

        if verbose:
            print_time(t0, "Fit complete", file=pout, flush=True)

        if return_final:
            return out, fit_tup[0]
        else:
            return out

    def predict(self, X, y, verbose):
        """Generic method for predicting through all layers in the container.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like, None (default = None)
            pass through for Scikit-learn compatibility.

        verbose : int or bool (default = False)
            level of verbosity.

                - ``verbose = 0`` silent (same as ``verbose = False``)
                - ``verbose = 1`` messages at start and finish \
                (same as ``verbose = True``)
                - ``verbose = 2`` messages for each layer

            If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
            For verbosity in the layers themselves, use ``fit_params``.

        Returns
        -------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            predictions from final layer.
        """
        if verbose:
            pout = "stdout" if verbose >= 3 else "stderr"
            safe_print("Predicting through all layers (%d)" % self.n_layers,
                       file=pout, flush=True)
            t0 = time()

        for layer_name, layer in self.layers.items():

            if verbose > 1:
                safe_print("[%s] predicting" % layer_name, file=pout,
                           flush=True)
                ti = time()

            X = layer.predict(X, y)

            if verbose > 1:
                print_time(ti, "[%s] done" % layer_name, file=pout, flush=True)

        if verbose:
            print_time(t0, "prediction complete", file=pout, flush=True)

        return X

    @staticmethod
    def _init_layers(layers):
        """Return a clean ordered dictionary or copy the passed dictionary."""
        if layers is None:
            layers = OrderedDict()
            layers.clear()
            return layers
        else:
            return layers

    def _check_n_layers(self, n_layers):
        """Check that n_layers match to dictionary length."""
        n = len(self.layers)
        if (n_layers != 0) and (n_layers != n):
            warnings.warn("Specified 'n_layers' [%d] does not correspond to "
                          "length of the 'layers' dictionary [%d]. Will "
                          "proceed with 'n_layers = len(layers)' "
                          "(%d)." % (n_layers, n, n),
                          LayerSpecificationWarning)

        return n

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the layers separately as individual
            parameters. If False, will return the collapsed dictionary.

        Returns
        -----------
        params : mapping of parameter names mapped to their values.
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

    """Layer of preprocessing pipes and estimators.

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

    raise_on_exception : bool (default = False)
        whether to raise an error on soft exceptions, else issue warning.

    Attributes
    ----------
    estimators\_ : OrderedDict, list
        container for fitted estimators, possibly mapped to preprocessing
        cases and / or folds.

    preprocessing\_ : OrderedDict, list
        container for fitted preprocessing pipelines, possibly mapped to
        preprocessing cases and / or folds.
    """

    def __init__(self, estimators, fit_function, predict_function,
                 preprocessing=None, fit_params=None, predict_params=None,
                 raise_on_exception=False):

        assert_correct_layer_format(estimators, preprocessing)

        self.estimators = check_instances(estimators)
        self.preprocessing = check_instances(preprocessing)
        self.fit_function = fit_function
        self.predict_function = predict_function
        self.fit_params = self._format_params(fit_params)
        self.predict_params = self._format_params(predict_params)
        self.raise_on_exception = raise_on_exception

    def fit(self, X, y):
        """Generic method for fitting and storing layer.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like of shape = [n_samples, ]
            training lables.

        Returns
        -------
        out : tuple
            output returned in from ``fit_function``. The first element
            **must** be training data for subsequent layer. Other entries
            are optional information passed onto the ensemble class making
            the ``fit`` call: ``out = (X_pred, ...)``.
        """
        self.estimators_, self.preprocessing_, out = \
            self.fit_function(self, X, y, **self.fit_params)
        return out

    def predict(self, X, y):
        """Generic method for predicting with fitted layer.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : None, array-like of shape = [n_samples, ]
            pass-through for Scikit-learn API.

        Returns
        -------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            predictions from fitted estimators in layer.
        """
        self._check_fitted()
        return self.predict_function(self, X, y, **self.predict_params)

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


# TODO: make the preprocessing of folds optional as it can take a lot of memory
class BaseEnsemble(BaseEstimator):

    """BaseEnsemble class.

    Core ensemble class methods used to add ensemble layers and manipulate
    parameters.
    """

    def _add(self, estimators, fit_function, predict_function,
             fit_params=None, predict_params=None, preprocessing=None):
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

        Returns
        -------
        self : instance, optional
            if ``in_place = True``, returns ``self`` with the layer
            instantiated.
        """
        if self.layers is None:
            raise_on_exception = getattr(self, 'raise_on_exception', False)
            self.layers = LayerContainer(raise_on_exception=raise_on_exception)

        self.layers.add(estimators=estimators,
                        preprocessing=preprocessing,
                        fit_function=fit_function,
                        fit_params=fit_params,
                        predict_function=predict_function,
                        predict_params=predict_params)
        return self

    def _fit_layers(self, X, y, return_final, verbose):
        """Generic method for fitting all layers in the ensemble.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for fitting and predicting.

        y : array-like of shape = [n_samples, ]
            training labels.

        return_final : bool (default = True)
            whether to return the final training input. In this case,
            the output of the ``fit`` method is a tuple ``(out, X)``.
            Otherwise, ``fit`` returns ``out``.

        verbose : int or bool (default = False)
            level of verbosity.

                - ``verbose = 0`` silent (same as ``verbose = False``)
                - ``verbose = 1`` messages at start and finish \
                (same as ``verbose = True``)
                - ``verbose = 2`` messages for each layer

            If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
            For verbosity in the layers themselves, use ``fit_params``.

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
        return self.layers.fit(X, y, return_final, verbose)

    def _predict_layers(self, X, y, verbose):
        """Generic method for predicting through all layers in the ensemble.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like, None (default = None)
            pass through for Scikit-learn compatibility.

         verbose : int or bool (default = False)
            level of verbosity.

                - ``verbose = 0`` silent (same as ``verbose = False``)
                - ``verbose = 1`` messages at start and finish \
                (same as ``verbose = True``)
                - ``verbose = 2`` messages for each layer

            If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
            For verbosity in the layers themselves, use ``fit_params``.

        Returns
        -------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            predictions from final layer.
        """
        return self.layers.predict(X, y, verbose)
