.. _ensemble-tutorial:

Tutorials
=========

The following tutorials highlight advanced functionality and a more detailed
view of how ensembles are built and trained.

The :ref:`proba-tutorial` shows how to build layers that output class
probabilities from each base learner, so that the next layer or meta estimator
is learning not from the predicted class of each base learner, but from each
base learner's probability distribution.

The :ref:`sequential-tutorial` shows how to build ensembles with different
layer classes. This can be a very powerful way of building scalable ensembles,
by letting initial layers use subsets of the full data, while later layers
use more compute intense estimation techniques.

The :ref:`memory-tutorial` details how users can avoid loading data into the
parent process by specifying a file path to a memmaped array or a csv file.

.. _proba-tutorial:

Probabilistic ensemble learning
-------------------------------



.. _sequential-tutorial:

General multi-layer ensemble learning
-------------------------------------


.. _memory-tutorial:

Passing file paths as data input
--------------------------------
