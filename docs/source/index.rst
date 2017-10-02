.. Home page

:github_url: https://github.com/flennerhag/mlens


.. figure:: _static/img/logo.png
   :align: center


ML-Ensemble a Python library for memory efficient parallelized ensemble learning. In particular, ML-Ensemble is a `Scikit-learn`_ compatible library for building deep ensemble networks in just a few lines of code. :: 

  ensemble = SuperLearner().add(estimators)
  ensemble.fit(xtrain, ytrain).predict(xtest)

:ref:`Get started <getting-started>` here, or head to the :ref:`advanced tutorials <tutorials>` for an in-depth tour of ML-Ensemble's high-level features. For the computational graph API, see :ref:`mechanics walkthroughs <learner_tutorial>`. 

ML-Ensemble is easily installed via ``pip``. For further details see :ref:`install`. 

.. code-block:: shell 

  pip install -U mlens 

ML-Ensemble is open for contributions at all levels. If you would like to get involved, reach out to the project's Github_ repository. 


Core Features
-------------

.. currentmodule:: mlens.parallel

A network approach to multi-layered ensembles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensembles are built by constructing a graph of :class:`Learner` nodes that handles all computations associated with a specific estimator. Nodes are connected sequentially by a :class:`Layer` and processed at maximum prallelization by a purpose-built :class:`ParallelProcessing` manager. A high-level API provides ready-made ensemble classes that allows you to build highly optimized ensembles of almost any shape and form in just a few lines of code. Alternatively, a low-level API gives you full control of the ensemble network and the computational procedure to build virtually any type of ensemble. 

.. figure:: _static/img/network.png
   :align: center
   :scale: 60%

   The computational graph of a layer. The input :math:`X` is either the
   original data or the previous layer's output;
   :math:`\textrm{Tr}^{(j)}` represents preprocessing pipelines that transform
   the input to its associated base learners :math:`f^{(i)}`. The
   :math:`\textrm{Ft}` operation propagates specified features :math:`s` from input to
   output. Base learner predictions :math:`p^{(i)}_j` are concatenated to
   propagated features :math:`X_{:, s}` to form the output matrix :math:`P`.

Easy to use high-level API
^^^^^^^^^^^^^^^^^^^^^^^^^^

Ready-made classes allow ensembles to be built by simply calling the ``add``
method with a set of estimators to group into a layer. Ensembles are 
Scikit-learn compatible estimators–no matter how complex the
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
ML-Ensemble is designed to
maximize parallelization at minimum memory footprint and is fully thread-safe.
It can fall back on multiprocessing seamlessly and completely avoids 
overheads stemming from data sharing. An ensemble will not require more memory 
when estimated in parallel than what it consumes with sequential
processing. For more details, see :ref:`memory`.

Differentiated preprocessing pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ML-Ensemble offers the possibility to specify a set
of preprocessing pipelines that map to a specific group of estimators in a layer. 
Implementing differentiated preprocessing is straightforward and simply requires
a dictionary mapping between preprocessing cases and estimators::

      ensemble = SuperLearner()

      preprocessing = {'pipeline-1': list_of_transformers_1,
                       'pipeline-2': list_of_transformers_2}

      estimators = {'pipeline-1': list_of_estimators_1,
                    'pipeline-2': list_of_estimators_2}

      ensemble.add(estimators, preprocessing)

Dedicated Diagnostics
^^^^^^^^^^^^^^^^^^^^^

ML-Ensemble is equipped with a model selection suite that lets you compare 
several models across any number of preprocessing pipelines in one go. 
In fact, you can use an ensemble as a preprocessing input to tune higher levels
of an ensemble. Output is directly summarized in table format for easy 
comparison of performance. ::

    >>> evaluator.results
               test_score_mean  test_score_std  train_score_mean  train_score_std  fit_time_mean  fit_time_std                                             params
    class rf          0.955357        0.060950          0.972535         0.008303       0.024585      0.014300                                   {'max_depth': 5}
          svc         0.961607        0.070818          0.972535         0.008303       0.000800      0.000233                               {'C': 7.67070164682}
    proba rf          0.980357        0.046873          0.992254         0.007007       0.022789      0.003296   {'max_depth': 3, 'max_features': 0.883535082341}
          svc         0.974107        0.051901          0.969718         0.008060       0.000994      0.000367                              {'C': 0.209602254061}

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Installation

   pages/install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: High-level API 

   tutorials/start_tutorial
   tutorials/adv_tutorial
   pages/ensembles

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Mechanics 

   tutorials/learner_tutorial
   tutorials/layer_tutorial
   tutorials/parallel_tutorial
   tutorials/sequential_tutorial

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Details

   pages/memory
   pages/benchmarks
   pages/scaling
   pages/dev
   pages/gotchas

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   pages/API
   pages/mlens_configs

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Additional Information

   pages/licence
   pages/updates

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Package index

   module/modules

----

ML Ensemble is licenced under :ref:`MIT <license>` and is hosted on Github_.

.. _issue: https://github.com/flennerhag/mlens/issues
.. _Github: https://github.com/flennerhag/mlens
.. _Scikit-learn: http://scikit-learn.org/stable/
.. _Keras: https://keras.io
