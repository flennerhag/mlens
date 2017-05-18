ML-Ensemble
===========

**A Python library for memory efficient parallelized ensemble learning**.

ML-Ensemble deploys sequential ensemble networks through a `Scikit-learn`_ API.
Ensembles can be made arbitrarily deep, by adding layers of base learners
that are fitted sequentially on previous layer's predictions. By leveraging a
network API similar to that of popular deep learning libraries like Keras_,
it is straightforward to build fast and memory efficient
multi-layered ensembles.

ML-Ensemble is looking for contributors at all levels of experience.
If you would like to get involved, reach out to the project's Github_
repository.

Core Features
-------------

Transparent Architecture API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensembles are built by adding layers to an instance object: layers in their
turn are comprised of a list of estimators. No matter how complext the
ensemble, to train it call the ``fit`` method::

    ensemble = Subsemble()

    # First layer
    ensemble.add(list_of_estimators)

    # Second layer
    ensemble.add(list_of_estimators)

    # Final meta estimator
    ensemble.add_meta(estimator)

    # Train ensemble
    ensemble.fit(X, y)

.. currentmodule:: mlens.ensemble

Memory Efficient Parallelized Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Because base learners in an ensemble are independent of each other, ensembles
benefit greatly from multiprocessing and the fundamental philosophy of
ML-Ensemble is to enable as much parallel processing as possible with minimum
impact on memory consumption. Estimation in parallel can easily lead to
unacceptable memory consumption if each sub-process requires a copy of the
training data, but ML-Ensemble avoids this issue by using memory mapping.

Training data is persisted to a memory cache that each sub-process has access
to, completely circumventing serialization of input data to be sent between
the parent process and sub-processes. Moreover, if no copying takes place
during estimation, parallel processing require no more memory than processing
single-process estimation. For more details, see :ref:`memory`.

Expect 95-97% of training time to be spent fitting the base estimators. In
general therefore, the time it takes to fit an ensemble depends
therefore on the speed of the chosen base learners and the number of CPU cores
available. Some ensembles will even scale more efficiently than their base
learners if these scale superlinearly.

Modular build of multi-layered ensembles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core unit of an ensemble is a **layer**, much as a hidden unit in a neural
network. Each layer contains an ensemble class specification and a mapping of
preprocessing pipelines to base learners to be used during fitting.

The modular build of an ensemble allows great flexibility in architecture,
both in terms of the depth of the ensemble (number of layers)
and how each layer generates predictions.

Differentiated preprocessing pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ML-Ensemble offers the possibility to specify, for each layer, a set
of preprocessing pipelines that maps to different (or the same) sets of
estimators. For instance, for one set of estimators, min-max-scaling might
be desired, while for a different set of estimators standization could be
preferred. This can easily be achieved in ML-Ensemble::

      ensemble = SuperLearner()

      preprocessing = {'pipeline-1': list_of_transformers_1,
                       'pipeline-2': list_of_transformers_2}

      estimators = {'pipeline-1': list_of_estimators_1,
                    'pipeline-2': list_of_estimators_2}

      ensemble.add(estimators, preprocessing)

Dedicated Diagnostics
^^^^^^^^^^^^^^^^^^^^^

Building complex ensembles requires an understanding of how base learners
interact. Grid searches on each estimator in isolation is unlikely to yield
superior results, not to mention being helpful in finding the right base
learners and meta estimator. ML-Ensemble comes equipped with a grid search
functionality that lets you run several estimators across any number of
preprocessing pipelines in one go. Ensemble transformers can be used to
build initial layers of ensembles as preprocessing pipelines to avoid
repeatedly fitting the same layer during model selection, which is orders of
magnitude faster that fitting an entire ensemble repeatedly just to evaluate
(say) the meta learner. Output allows easy comparison of estimator performance,
as in the example below. ::

    >>> DataFrame(evaluator.summary)
               test_score_mean  test_score_std  train_score_mean  train_score_std  fit_time_mean  fit_time_std                                             params
    class rf          0.955357        0.060950          0.972535         0.008303       0.024585      0.014300                                   {'max_depth': 5}
          svc         0.961607        0.070818          0.972535         0.008303       0.000800      0.000233                               {'C': 7.67070164682}
    proba rf          0.980357        0.046873          0.992254         0.007007       0.022789      0.003296   {'max_depth': 3, 'max_features': 0.883535082341}
          svc         0.974107        0.051901          0.969718         0.008060       0.000994      0.000367                              {'C': 0.209602254061}

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Installation

   install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guides

   getting_started
   ensemble_tutorial
   ensembles

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Details

   memory
   benchmarks
   scaling
   gotchas

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
   updates

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Package index

   source/modules

..
====

ML Ensemble is licenced under :ref:`MIT <license>` and is hosted on Github_.

.. _Github: https://github.com/flennerhag/mlens
.. _Scikit-learn: http://scikit-learn.org/stable/
.. _Keras: https://keras.io
