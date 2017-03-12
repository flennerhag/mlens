.. Main API page

API
====

ML-Ensemble uses two main APIs, one for ensemble estimators and one for model
selection. The API for ensemble estimators is as close to the API used by
Scikit-learn. The main difference is that to properly instantiate an ensemble
instance, in addition to creating a class instance, at least on layer and
if applicable a meta estimator must be attached to the ensemble. Hence, while
Scikit-learn would allow the user to fit an estimator with::

   estimator = Estimator()
   estimator.fit(X, y)

A ML-Ensemble estimator needs to first initiate layers before the ``fit``
method can be called::

   ensemble = Ensemble().add(list_of_estimators).add_meta(estimator)
   ensemble.fit(X, y)


Ensemble estimators
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mlens.ensemble

.. autosummary::

   StackingEnsemble

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
   IO_plot_comp
   IO_plot
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

