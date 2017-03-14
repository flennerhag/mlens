"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT
"""

from __future__ import division, print_function

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler as StandardScaler_


class StandardScaler(StandardScaler_):

    """Standardize input data.

    Wrapper around Scikit-learn's ``StandardScaler`` that preserves the input
    data's original type. Specifically, if the input array is a pandas
    ``DataFrame``, the standardized data is returned as a ``DataFrame``, and if
    the input data is a NumPy ``ndarray``, the standardized data is returned as
    a ``ndarray``.

    See Also
    --------
    :class:`sklearn.preprocessing.StandardScaler`
    """

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling.

        Same as the original ``transform`` method, but preserves the
        input type.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The data used to scale along the features axis.

        y : array-like of shape = [n_samples, n_features]
            pass-through for Scikit-learn pipeline compatibility.

        copy : bool (default = None)
            whether to copy X before transforming.

        Returns
        -------
        X_scaled : array-like of shape = [n_samples, n_features]
            The scaled data.
        """
        if isinstance(X, DataFrame):
            X.loc[:, :] = super(StandardScaler, self).transform(X, y, copy)
        elif isinstance(X, Series):
            X.loc[:] = super(StandardScaler, self).transform(X, y, copy)
        else:
            X = super(StandardScaler, self).transform(X, y, copy)

        return X


class Subset(BaseEstimator, TransformerMixin):

    """Select a subset of features.

    The ``Subset`` class acts as a transformer that reduces the feature set
    to a subset specified by the user.

    Parameters
    ----------
    subset : list
        list of columns indexes to select subset with. Indexes can
        either be of type ``str`` if data accepts slicing on a list of
        strings, otherwise the list should be of type ``int``.
    """

    def __init__(self, subset=None):
        self.subset = subset

    def fit(self, X, y=None):
        """Learn what format the data is stored in.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The whose type will be inferred.

        y : array-like of shape = [n_samples, n_features]
            pass-through for Scikit-learn pipeline compatibility.
        """
        self.is_df_ = isinstance(X, (DataFrame, Series))

        if self.subset is not None:
            self.use_loc_ = any([isinstance(x, str) for x in self.subset])

        return self

    def transform(self, X, y=None, copy=False):
        """Return specified subset of X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The whose type will be inferred.

        y : array-like of shape = [n_samples, n_features]
            pass-through for Scikit-learn pipeline compatibility.

        copy : bool (default = None)
            whether to copy X before transforming.
        """
        if self.subset is None:
            return X

        else:
            Xt = X.copy() if copy else X

            if self.is_df_ and self.use_loc_:
                Xt = Xt.loc[:, self.subset]

            elif self.is_df_:
                Xt = Xt.iloc[:, self.subset]

            else:
                Xt = Xt[:, self.subset]

            return Xt
