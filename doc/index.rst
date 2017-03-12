ML-Ensemble
===========

ML-Ensemble is a Python library for **parallelized ensemble learning**,
deploying an API that combines a flexibility approach to building a network
of estimators to combine and the Scikit-learn estimator API. Ensembles behave
just as any Scikit-learn estimator and are fully compatible with the
Scikit-learn library.

The only fundamental difference between the ML-Ensemble API and the
Scikit-learn API is how to instantiate an estimator. In Scikit-learn,
predictions can be achieved by::

   estimator = Estimator()
   estimator.fit(X, y)
   predictions = estimator.predict(X

In ML-Ensemble, to fully instantiate an estimator you need to specify a layer
of estimators to combine::

   estimator = Ensmeble().add(list_of_estimators).add_meta(meta_estimator)
   estimator.fit(X, y)
   predictions = estimator.predict(X

See :ref:`getting-started` and :ref:`ensemble-tutorial` for further examples.

Features
--------

Layers
^^^^^^

The core unit of an ensemble is a **layer**, much as a hidden unit in a neural
network. Each layer contains a set of preprocessing steps, estimators and
an estimation type that determines how to fit the estimators of the layer
and generate predictions.

The modular build of an ensemble allows great flexibility in building ensemble
structures, both in terms of the depth of the ensemble (number of layers)
and how each layer generates predictions. As such, it is straightforward to
build arbitrarily complex ensembles that can combine several ensemble
estimation methods (such as stacking, blending, Subsemble e.t.c.).

Differentiated preprocessing pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A key feature of ML-Ensemble is the ability to specify, for each layer, a set
of preprocessing pipelines, and map different (or the same) sets of of
estimators to each preprocessing pipeline. ::

      ensemble = StackingEnsemble()

      preprocessing = {'pipeline-1': list_of_transformers_1,
                       'pipeline-2': list_of_transformers_2}

      estimators = {'pipeline-1': list_of_estimators_1,
                    'pipeline-2': list_of_estimators_2}

      ensemble.add(estimators, preprocessing)

Dedicated Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^^

ML Ensemble implements a dedicated model selection and ensemble evaluation
suite for simplified evaluation of preprocessing pipelines and estimators in an
ensemble for faster and simpler development.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Installation

   install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   ensemble_tutorial



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   API

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Additional Information

   licence


ML Ensemble is licenced under MIT and is hosted on Github.
