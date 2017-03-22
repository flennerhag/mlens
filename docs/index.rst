ML-Ensemble
===========

ML-Ensemble is a Python library for **performant, memory efficient,
parallelized ensemble network learning**.

ML-Ensemble utilizes the `Scikit-learn`_ API and all ML-Ensemble estimators
are proper Scikit-learn estimators that can be combined with any Scikit-learn
functionality, such as grid searches or pipelines. Moreover, ML-Ensemble adopts
a network building API similar to popular Neural Network libraries like Keras_.
As such, it straightforward to 'deep' ensembles of any level of complexity.

ML-Ensemble uses **shared memory** across subprocesses to minimize footprint on
the machine's RAM, and in fact an ensemble can use *less* memory during fitting
and predicting than its some of its constituents estimators when
there are fitted on a stand-alone basis.

ML-Ensemble is highly performant, and an appropriately built ensemble will
always outperforms any of its constituent estimators. For testimonials, see
:ref:`benchmarks`. If you are new to ML-Ensemble, check out
:ref:`getting-started` and :ref:`ensemble-tutorial`.

Core Features
-------------

Memory Efficient Parallelized Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fundamental philosophy of ML-Ensemble is multithreading: using as much of a
machine's full processing power to fit as many estimators in parallel as
possible. This can easily lead to exponential use of memory and with large
ensembles, even relatively small datasets might be too large to fit.
ML-Ensemble avoids this by sharing one copy of training and test data across
all processes, and as such no unnecessary copying taking place. In fact, but
using shared memory stored on disk, an ensemble can be a good deal **more**
memory efficient than some of its constituent estimators when these are fitted
outside of the ensemble!

Not only is ML-Ensemble memory efficient, it is also fast: between 95-97% of
training time is spent fitting estimators *irrespective* of data size, and as
such the time it takes to fit an ensemble depends only on how fast and scalable
the estimators you put in are, and how many CPU cores you have available.

Multi-Layered ensembles
^^^^^^^^^^^^^^^^^^^^^^^

The core unit of an ensemble is a **layer**, much as a hidden unit in a neural
network. Each layer contains a set of preprocessing steps, estimators and
an estimation type that determines how to fit the estimators of the layer
and generate predictions.

The modular build of an ensemble allows great flexibility in building ensemble
structures, both in terms of the depth of the ensemble (number of layers)
and how each layer generates predictions. As such, it is straightforward to
build arbitrarily complex ensembles, where each layer use a different
estimation method (i.e. stacking, blending, Subsemble e.t.c.).

Differentiated preprocessing pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A key feature of ML-Ensemble is the ability to specify, for each layer, a set
of preprocessing pipelines, and map different (or the same) sets of of
estimators to each preprocessing pipeline. ::

      ensemble = SuperLearner()

      preprocessing = {'pipeline-1': list_of_transformers_1,
                       'pipeline-2': list_of_transformers_2}

      estimators = {'pipeline-1': list_of_estimators_1,
                    'pipeline-2': list_of_estimators_2}

      ensemble.add(estimators, preprocessing)

Dedicated Diagnostics
^^^^^^^^^^^^^^^^^^^^^

ML Ensemble implements a dedicated diagnostics and model selection suite
for intuitive and speedy ensemble evaluation. This is suite is under
development, so check in frequently for new functionality!


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Installation

   install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Benchmarks

   benchmarks

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
   source/modules

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Additional Information

   licence


ML Ensemble is licenced under MIT and is hosted on Github_.

.. _Github: https://github.com/flennerhag/mlens
