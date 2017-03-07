"""ML-ENSEMBLE

author: Sebastian Flennerhag
copyright: 2016
licence: MIT
Base ensemble class for layer API and generic 'fit' and 'predict' calls on
instances.
"""

from __future__ import division, print_function
from collections import OrderedDict as Odict

from sklearn.base import BaseEstimator
from ..base import check_instances
from ..utils import check_is_fitted, print_time
from ..utils.checks import (LayerSpecificationWarning, LayerSpecificationError,
                            assert_correct_layer_format, NotFittedError)

import warnings
import sys
from time import time


class LayerContainer(BaseEstimator):

    """Container class for layers.

    The LayerContainer class stories all layers as an ordered dictionary
     and modifies possesses a 'get_params' method to appear as an estimator
     in the Scikit-learn API. This allows correct cloning and parameter
     updating.

     ROAD MAP: this class is meant to also house connection layers and other
              intra-layer information, and act as the core layer-handler
              in all ensembles.

     Parameters
     ----------
     layers : OrderedDict (default = OrderedDict())
        An ordered dictionary of Layer instances.

     n_layers : int (default = None)
        number of layers instantiated. Automatically set, normally there is no
        reason to fiddle with this parameter.
        
    Methods
    ----------
    fit : X, y, fit_function, verbose, args, kwargs
        Generic fit method for fitting and storing all layers in the
        container The 'fit_function' must have the following API:
        
        '''
        (estimators_, preprocessing_, fitted_estimators_) = \
            fit_function(layer_instance, X, y, *args, **kwargs)
        '''
        
        where estimators_ and preprocessing_ is a generic container of fitted
        instances, such as a list or dict. These will be stored in the
        layer class and passed to a user-specified 'predict_function' in the
        layer"s 'predict' method.
                
    predict : X, y, predict_function, args, kwargs
        Generic function for generating predictions through all layers. The
        'predict_function' must have the following  API:
        
        '''
        X_preds = predict_function(layer_instance, X, y, *args, **kwargs)
        '''
        
        The predictions of each layer is passed as input to the subsequent
        layer.
        
    add : estimators, preprocessing
        Method for adding a layer to the container.
    """

    def __init__(self, layers=None, n_layers=None):
        self.layers = self._init_layers(layers)
        self.n_layers = self._check_n_layers(n_layers)

    def add(self, fit_function, predict_function, estimators,
            preprocessing=None, fit_params=None, predict_params=None,
            in_place=True):
        """Method for adding a layer.
        
        Parameters
        -----------
        fit_function : function
            The fit_function determines the type of layer and must have the
            following API:

            '''
            (estimators_, preprocessing_, fitted_estimators_, out) = \
                fit_function(layer_instance, X, y, *args, **kwargs)
            '''

            where estimators_ and preprocessing_ are generic containers of
            fitted  instances, such as lists or dicts. These will be passed
            to a user-specified 'predict_function' in each layer"s 'predict'
            method. 'out' is a tuple of output data that at least contains
            the training data for the subsequent layer as its first element.

        predict_function : function
            The predict_function must have the following API:

            '''
            X_pred = predict_function(layer_instance, X, y, *args, **kwargs)
            '''
            for each layer, its predictions will be passed as input for the
            subsequent layer.
            
            
        preprocessing: dict of lists or list, optional (default = [])
            preprocessing pipelines for given layer. If
            the same preprocessing applies to all estimators, 'preprocessing'
            can be a list of transformer instances. The list can contain the
            instances directly, or named tuples of transformers:
    
            '''
            option_1 = [transformer_1, transformer_2]
            option_2 = [("trans-1", transformer_1), ("trans-2", transformer_2)]
            '''
    
             If different preprocessing pipelines are desired, a dictionary
             that maps preprocessing pipelines must be passed. The names of the
             preprocessing dictionary must correspond to the names of the
             estimator dictionary.
    
             '''
             preprocessing_cases = {"case-1": [trans_1, trans_2].
                                    "case-2": [alt_trans_1, alt_trans_2]}
    
             estimators = {"case-1": [est_a, est_b].
                           "case-2": [est_c, est_d]}
             '''
    
             The lists for each dictionary entry can be both a list of
             transformers and a list of named tuples of transformers,
             as in 'option_1' and 'option_2' respectively.
    
        estimators: dict of lists or list
            estimators constituting the layer. If no preprocessing,
            or preprocessing applies to all estimators, a list of estimators
            can be passed. The list can either contain estimator instances,
            or named tuples of estimator instances:
    
            '''
            option_1 = [estimator_1, estimator_2]
            option_2 = [("est-1", estimator_1), ("est-2", estimator_2)]
            '''
    
             If different preprocessing pipelines are desired, a dictionary
             that maps estimators to preprocessing pipelines must be passed.
             The names of the estimator dictionary must correspond to the
             names of the estimator dictionary:
    
             '''
             preprocessing_cases = {"case-1": [trans_1, trans_2].
                                    "case-2": [alt_trans_1, alt_trans_2]}
    
             estimators = {"case-1": [est_a, est_b].
                           "case-2": [est_c, est_d]}
             '''
    
             The lists for each dictionary entry can be both a list of
             estimators and a list of named tuples of estimators,
             as in 'option_1' and 'option_2' respectively.
                          
        fit_params : dict, tuple or None (default = None)
            optional arguments passed to 'fit_function'.
            
        predict_params : dict, tuple or None (default = None)
            optional arguments passed to 'predict_function'.
             
        in_place: bool (default = True)
            whether to return the instance (i.e. 'self').
            
        Returns
        ----------
        lc : instance, None (default = None)
            if in_place = True, returns the instance with the instantiated
            layer.
        """
        # Instantiate layer
        lyr = Layer(estimators=estimators,
                    preprocessing=preprocessing,
                    fit_function=fit_function,
                    fit_params=fit_params,
                    predict_function=predict_function,
                    predict_params=predict_params)
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
            input matrix to be used for prediction.

        y : array-like of shape = [n_samples, ]
            output vector to trained estimators on.
            
        return_final : bool (default = True)
            whether to return the final training input. In this case,
            the output of the 'fit' method is a tuple '(out, X)'.
            Otherwise, 'fit' returns 'out'.

        verbose : int or bool (default = False)
            level of verbosity of 'LayerContainer' instance:
                - verbose = 0 : silent (same as verbose = False)
                - verbose = 1 : times full fitting (same as verbose = True)
                - verbose = 2 : times all layer fittings
                
            If verbose >= prints to sys.stdout, else sys.stderr.
            For verbose layers, use kwargs for the 'fit_function'

        Returns
        -----------
        out : dict
            dictionary of output data (possibly empty) generated
            through fitting. Keys correspond to layer names and values to
            the output generated by calling the layer"s 'fit_function'.

        X : array-like (optional)
            predictions from final layer"s fit call
        """
        if verbose:
            pout = "stdout" if verbose >= 3 else "stderr"
            print("Fitting all layers (%d)" % self.n_layers,
                  file=getattr(sys, pout), flush=True)
            t0 = time()

        fit_tup = (X,)  # initiate a tuple of fit outputs for first layer
        out = {}
        for layer_name, layer in self.layers.items():
            
            if verbose > 1:
                print("[%s] fitting" % layer_name,
                      file=getattr(sys, pout), flush=True)
                ti = time()

            fit_tup = layer.fit(fit_tup[0], y)
            out[layer_name] = fit_tup[1:]
            
            if verbose > 1:
                print_time(ti, "[%s] done" % layer_name,
                           file=getattr(sys, pout), flush=True)

        if verbose:
            print_time(t0, "Fit complete", file=getattr(sys, pout), flush=True)
            
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

        y : array-like of shape = [n_samples, ]
            output vector to trained estimators on.
        
        verbose : int or bool (default = False)
            level of verbosity of 'LayerContainer' instance:
            
                - verbose = 0 : silent (same as verbose = False)
                - verbose = 1 : times full prediction (same as verbose = True)
                - verbose = 2 : times all layer predictions
                
            If verbose >= prints to sys.stdout, else sys.stderr.

            For verbose layers, use kwargs for the 'predict_function'
            
        Returns
        -----------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            prediction matrix from final layer.
        """
        if verbose:
            pout = "stdout" if verbose >= 3 else "stderr"
            print("Predicting through all layers (%d)" % self.n_layers,
                  file=getattr(sys, pout), flush=True)
            t0 = time()
    
        for layer_name, layer in self.layers.items():
        
            if verbose > 1:
                print("[%s] predicting" % layer_name,
                      file=getattr(sys, pout), flush=True)
                ti = time()
        
            X = layer.predict(X, y)
        
            if verbose > 1:
                print_time(ti, "[%s] done" % layer_name,
                           file=getattr(sys, pout), flush=True)
    
        if verbose:
            print_time(t0, "prediction complete",
                       file=getattr(sys, pout), flush=True)
    
        return X
    
    @staticmethod
    def _init_layers(layers):
        """Return a clean ordered dictionary or copy the passed dictionary."""
        if layers is None:
            layers = Odict()
            layers.clear()
            return layers
        else:
            return layers
        
    def _check_n_layers(self, n_layers):
        """Check that n_layers match to dictionary length."""
        n = len(self.layers)
        if n_layers is not None and (n_layers != n):
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
    preprocessing: dict of lists or list, optional (default = [])
        preprocessing pipelines for given layer. If
        the same preprocessing applies to all estimators, 'preprocessing'
        can be a list of transformer instances. The list can contain the
        instances directly, or named tuples of transformers:

        '''
        option_1 = [transformer_1, transformer_2]
        option_2 = [("trans-1", transformer_1), ("trans-2", transformer_2)]
        '''

         If different preprocessing pipelines are desired, a dictionary that
         maps preprocessing pipelines must be passed. The names of the
         preprocessing dictionary must correspond to the names of the
         estimator dictionary.

         '''
         preprocessing_cases = {"case-1": [trans_1, trans_2].
                                "case-2": [alt_trans_1, alt_trans_2]}

         estimators = {"case-1": [est_a, est_b].
                       "case-2": [est_c, est_d]}
         '''

         The lists for each dictionary entry can be both a list of transformers
         and a list of named tuples of transformers, as in 'option_1' and
         'option_2' respectively.

    estimators: dict of lists or list
        estimators constituting the layer. If no preprocessing,
        or preprocessing applies to all estimators, a list of estimators can be
        passed. The list can either contain estimator instances, or named
        tuples of estimator instances:

        '''
        option_1 = [estimator_1, estimator_2]
        option_2 = [("est-1", estimator_1), ("est-2", estimator_2)]
        '''

         If different preprocessing pipelines are desired, a dictionary that
         maps estimators to preprocessing pipelines must be passed. The
         names of the estimator dictionary must correspond to the names of the
         estimator dictionary:

         '''
         preprocessing_cases = {"case-1": [trans_1, trans_2].
                                "case-2": [alt_trans_1, alt_trans_2]}

         estimators = {"case-1": [est_a, est_b].
                       "case-2": [est_c, est_d]}
         '''

         The lists for each dictionary entry can be both a list of estimators
         and a list of named tuples of estimators, as in 'option_1' and
         'option_2' respectively.
         
    fit_function : function
        The fit_function must have the following API:
        
        '''
        (estimators_, preprocessing_, fitted_estimators_, out) = \
            fit_function(layer_instance, X, y, *args, **kwargs)
        '''
        
        where estimators_ and preprocessing_ is a generic container of
        fitted  instances, such as a list or dict. These will be passed
        to a user-specified 'predict_function' in the 'predict' call.
        'out' is a (possibly empty) generic return object.
         
    predict_function : function
        The predict_function must have the following API:

        '''
        y_pred = predict_function(layer_instance, X, y, *args, **kwargs)
        '''
        
    fit_params : dict, tuple or None (default = None)
        optional arguments passed to 'fit_function'.
        
    predict_params : dict, tuple or None (default = None)
        optional arguments passed to 'predict_function'.
         
    Attributes
    -----------
    estimators_ : OrderedDict, list
        container for fitted estimators, possibly mapped to preprocessing
        cases and / or folds.
        
    preprocessing_ : OrderedDict, list
        container for fitted preprocessing pipelines, possibly mapped to
        preprocessing cases and / or folds.
                        
    fitted_estimators_ : list
        list of fitted estimator names.
                                   
    Methods
    ----------
    fit : X, y, fit_function, args, kwargs
        Generic fit function for storing fitted estimators and preprocessing
        pipelines. The 'fit_function' must have the following API:
        
        '''
        (estimators_, preprocessing_, fitted_estimators_) = \
            fit_function(layer_instance, X, y, *args, **kwargs)
        '''
        
        where estimators_ and preprocessing_ is a generic container of fitted
        instances, such as a list or dict. These will be passed to a user-
        specified 'predict_function' in the 'predict' call.
                
    predict : X, y, predict_function, args, kwargs
        Generic predict function for predicting using fitted layer. The
        'predict_function' must have the following  API:
        
        '''
        y_preds = predict_function(layer_instance, X, y, *args, **kwargs)
        '''
    """

    def __init__(self, estimators, fit_function, predict_function,
                 preprocessing=None, fit_params=None, predict_params=None):
        
        assert_correct_layer_format(estimators, preprocessing)
        
        self.estimators = check_instances(estimators)
        self.preprocessing = check_instances(preprocessing)
        self.fit_function = fit_function
        self.predict_function = predict_function
        self.fit_params = self._format_params(fit_params)
        self.predict_params = self._format_params(predict_params)
        
    def fit(self, X, y):
        """Generic method for fitting and storing layer.
        
        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like of shape = [n_samples, ]
            output vector to trained estimators on.
            
        Returns
        -----------
        out : tuple
            output returned in the fit_function. The first element must be
            training data for subsequent layer. Other entries are optional
            information passed onto the ensemble class making the 'fit' call.
        """
        self.estimators_, self.preprocessing_, out = \
            self.fit_function(self, X, y, **self.fit_params)
        return out
        
    def predict(self, X, y):
        """Generic method for predicting with fitted layer.

        Parameters
        -----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : None, array-like of shape = [n_samples, ]
            pass-through for Scikit-learn API.

        Returns
        -----------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            predictions from fitted estimators.
        """
        self._check_fitted()
        return self.predict_function(self, X, y, **self.predict_params)

    def _check_fitted(self):
        """Utility function for checking that fitted estimators exist."""
        check_is_fitted(self, "estimators_")
        
        # Check that there is at least one fitted estimator
        if len(self.estimators_) == 0:
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
        deep : boolean, optional
            If True, will return the layers separately as individual
            parameters. If False, will return the collapsed dictionary.

        Returns
        -------
        params : mapping of parameter names mapped to their values.
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

    Methods
    -------
    add : estimators, preprocessing (default = [])
        add a layer consisting of a mapping of estimators and preprocessing
        pipelines. See 'Layer' for documentation on format of estimators and
        preprocessing.

    get_params : None
        Method for generating mapping of parameters. Sklearn API.
    """
    def _init_layers(self, layers):
        """Set up empty LayerContainer and layer counter, or attach clone."""
        if layers is None:
            self.layers = LayerContainer()
        else:
            self.layers = layers

    def _add(self, estimators, fit_function, predict_function,
             fit_params=None, predict_params=None, preprocessing=None,):
        """Method for adding a layer.

        Parameters
        -----------
        fit_function : function
            The fit_function determines the type of layer and must have the
            following API:

            '''
            (estimators_, preprocessing_, fitted_estimators_, out) = \
                fit_function(layer_instance, X, y, *args, **kwargs)
            '''

            where estimators_ and preprocessing_ are generic containers of
            fitted  instances, such as lists or dicts. These will be passed
            to a user-specified 'predict_function' in each layer"s 'predict'
            method. 'out' is a tuple of output data that at least contains
            the training data for the subsequent layer as its first element.

        predict_function : function
            The predict_function must have the following API:

            '''
            X_pred = predict_function(layer_instance, X, y, *args, **kwargs)
            '''
            for each layer, its predictions will be passed as input for the
            subsequent layer.


        preprocessing: dict of lists or list, optional (default = [])
            preprocessing pipelines for given layer. If
            the same preprocessing applies to all estimators, 'preprocessing'
            can be a list of transformer instances. The list can contain the
            instances directly, or named tuples of transformers:

            '''
            option_1 = [transformer_1, transformer_2]
            option_2 = [("trans-1", transformer_1), ("trans-2", transformer_2)]
            '''

             If different preprocessing pipelines are desired, a dictionary
             that maps preprocessing pipelines must be passed. The names of the
             preprocessing dictionary must correspond to the names of the
             estimator dictionary.

             '''
             preprocessing_cases = {"case-1": [trans_1, trans_2].
                                    "case-2": [alt_trans_1, alt_trans_2]}

             estimators = {"case-1": [est_a, est_b].
                           "case-2": [est_c, est_d]}
             '''

             The lists for each dictionary entry can be both a list of
             transformers and a list of named tuples of transformers,
             as in 'option_1' and 'option_2' respectively.

        estimators: dict of lists or list
            estimators constituting the layer. If no preprocessing,
            or preprocessing applies to all estimators, a list of estimators
            can be passed. The list can either contain estimator instances,
            or named tuples of estimator instances:

            '''
            option_1 = [estimator_1, estimator_2]
            option_2 = [("est-1", estimator_1), ("est-2", estimator_2)]
            '''

             If different preprocessing pipelines are desired, a dictionary
             that maps estimators to preprocessing pipelines must be passed.
             The names of the estimator dictionary must correspond to the
             names of the estimator dictionary:

             '''
             preprocessing_cases = {"case-1": [trans_1, trans_2].
                                    "case-2": [alt_trans_1, alt_trans_2]}

             estimators = {"case-1": [est_a, est_b].
                           "case-2": [est_c, est_d]}
             '''

             The lists for each dictionary entry can be both a list of
             estimators and a list of named tuples of estimators,
             as in 'option_1' and 'option_2' respectively.

        fit_params : dict, tuple or None (default = None)
            optional arguments passed to 'fit_function'.

        predict_params : dict, tuple or None (default = None)
            optional arguments passed to 'predict_function'.

        Returns
        ----------
        self : instance
            The instance with the instantiated layer.
        """
        self.layers.add(estimators=estimators,
                        preprocessing=preprocessing,
                        fit_function=fit_function,
                        fit_params=fit_params,
                        predict_function=predict_function,
                        predict_params=predict_params)
        return self

    def _fit_layers(self, X, y, return_final, verbose):
        """Fit layers of an ensemble.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like, shape=[n_samples, ]
            output vector to trained estimators on.
            
        return_final : bool (default = True)
            whether to return the final training input. In this case,
            the output of the 'fit' method is a tuple '(out, X)'.
            Otherwise, 'fit' returns 'out'.

        verbose : int or bool (default = False)
            level of verbosity of 'LayerContainer' instance:
                - verbose = 0 : silent (same as verbose = False)
                - verbose = 1 : times full fitting (same as verbose = True)
                - verbose = 2 : times all layer fittings
                
            If verbose >= prints to sys.stdout, else sys.stderr.
            For verbose layers, use kwargs for the 'fit_function'

        Returns
        ----------
        out : dict
            dict of fit data as specified by each layer"s 'fit_function'.
        """
        return self.layers.fit(X, y, return_final, verbose)
        
    def _predict_layers(self, X, y, verbose):
        """Predict with fitted ensemble.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like, default=None
            pass through for pipeline compatibility.

        verbose : int or bool (default = False)
            level of verbosity of 'LayerContainer' instance:
            
                - verbose = 0 : silent (same as verbose = False)
                - verbose = 1 : times full prediction (same as verbose = True)
                - verbose = 2 : times all layer predictions
                
            If verbose >= prints to sys.stdout, else sys.stderr.

            For verbose layers, use kwargs for the 'predict_function'
            
        Returns
        -----------
        X_pred : array-like of shape = [n_samples, n_fitted_estimators]
            prediction matrix from final layer.
        """
        return self.layers.predict(X, y, verbose)
