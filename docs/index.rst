ML-Ensemble
===========

ML-Ensemble is a Python library for **memory efficient
parallelized ensemble network learning**.

ML-Ensemble emphasizes user friendliness, and all estimators follows the
`Scikit-learn`_ API. In fact, any ML-Ensemble estimator is passes as a
proper Scikit-learn estimator and interacts seamlessly with any
Scikit-learn object. The ensemble layer API is similar to that of popular
Neural Network libraries like Keras_, and it is straightforward to build deep
ensembles of any desired level of complexity.

ML-Ensemble implements a range of ensemble techniques and the list is
constantly growing. For latest news, see :ref:`updates`.

If you are new to ML-Ensemble, check out the :ref:`getting-started`
and :ref:`ensemble-tutorial`. For more detailed examples, see the
:ref:`memory` and :ref:`scaling` section. Performance testimonials can be found
in the :ref:`here <benchmarks>`.

If you would like to get involved, don't hesitate to reach out on Github_ !

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
benefit greatly from multithreading and the fundamental philosophy of
ML-Ensemble is to enable as much parallel processing as possible with minimum
impact on memory consumption. Estimation in parallel can easily lead to
unacceptable memory consumption if each sub-process requires a copy of the
training data, but ML-Ensemble avoids this issue by using memory mapping.

Training data is persisted to a memory cache that each sub-process has access
to, allowing parallel processing to require no more memory than processing
on a single thread. For more details, see :ref:`memory`.

Expect 95-97% of training time to be spent fitting the base estimators -
*irrespective* of data size. The time it takes to fit an ensemble depends
therefore entirely on how fast the chosen base learners are,
and how many CPU cores are available.

Moreover, ensemble classes that fit estimators on subsets scale more
efficiently than the base learners when these do not scale linearly.

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

ML Ensemble implements a dedicated diagnostics and model selection suite
for intuitive and speedy ensemble evaluation. This suite is under
development, so check in frequently for new functionality.


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

   benchmarks
   memory
   scaling

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   API
   source/modules

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Additional Information

   licence

References
----------
.. [1] van der Laan, Mark J.; Polley, Eric C.; and Hubbard, Alan E.,
   "Super Learner" (July 2007). U.C. Berkeley Division of Biostatistics
   Working Paper Series. Working Paper 222.
   http://biostats.bepress.com/ucbbiostat/paper222

ML Ensemble is licenced under MIT and is hosted on Github_.

.. _Github: https://github.com/flennerhag/mlens
.. _Scikit-learn: http://scikit-learn.org/stable/
.. _Keras: https://keras.io
