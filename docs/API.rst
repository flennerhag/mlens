.. Main API page

API
====

ML-Ensemble estimators behaves essentially identical to `Scikit-learn`_
estimators, with one main difference: to properly instantiate an ensemble,
at least on layer, and if applicable a meta estimator, must be added to the
ensemble. Otherwise, there is no ensemble to estimate. The difference
can be summarized as follows. ::

   # sklearn API
   estimator = Estimator()
   estimator.fit(X, y)

    # mlens API
   ensemble = Ensemble().add(list_of_estimators).add_meta(estimator)
   ensemble.fit(X, y)


Ensemble estimators
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mlens.ensemble

.. autosummary::

   SuperLearner

Model Selection
^^^^^^^^^^^^^^^

.. currentmodule:: mlens.model_selection

.. autosummary::

   Evaluator
   StackingTransformer

Preprocessing
^^^^^^^^^^^^^

.. currentmodule:: mlens.preprocessing

.. autosummary::

   PredictionFeature
   StandardScaler
   Subset

Visualization
^^^^^^^^^^^^^

.. currentmodule:: mlens.visualization

.. autosummary::

   corrmat
   clustered_corrmap
   corr_X_y
   pca_plot
   pca_comp_plot
   exp_var_plot

For Developers
^^^^^^^^^^^^^^

For developers, the API of the ``LayerContainer`` and ``Layer`` class are
essential for building ensemble classes. When building a ML-Ensemble estimator,
use the :class:`mlens.ensemble.base.BaseEnsemble` as the parent class to import
the ``_add``, ``_fit`` and ``_predict`` methods, rather than calling methods of
``LayerContainer`` directly.

.. currentmodule:: mlens.ensemble.base

.. autosummary::

   LayerContainer
   Layer

.. _Scikit-learn: http://scikit-learn.org/stable/
