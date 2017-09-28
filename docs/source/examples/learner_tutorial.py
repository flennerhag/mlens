# -*- coding: utf-8 -*-
"""

.. _learner_tutorial:


.. currentmodule: mlens.parallel.learner

Learner Mechanics
=================

ML-Ensemble is designed to provide an easy user interface. But it is also designed
to be extremely flexible, all the wile providing maximum concurrency at minimal
memory consumption. The lower-level API that builds the ensemble and manages the
computations is constructed in as modular a fashion as possible.

The low-level API introduces a computational graph-like environment that you can
directly exploit to gain further control over your ensemble. In fact, building
your ensemble through the low-level API is almost as straight forward as using the
high-level API. In this tutorial, we will walk through the key core :class:`Learner` class.


The Learner API
^^^^^^^^^^^^^^^

Basics
------

When you pass an estimator to an ensemble, it gets wrapper
in a :class:`Learner` instance. This class records relevant information
about the estimator and manages the cross-validated fit. It also keeps
track of which preprocessing pipeline to use (if any). You can think of a learner as a core node in a computational graph. This node has as auxiliary nodes a blueprint estimator to use for estimation, an indexer, a pointer to a preprocessing node, and an output mapping:
"""
from mlens.utils.dummy import OLS
from mlens.parallel import Learner
from mlens.index import FoldIndex


indexer = FoldIndex(n_splits=2)            # Define a training strategy
learner = Learner(estimator=OLS(),         # Declare estimator
                  preprocess=None,         # We'll get to this
                  indexer=indexer,         # Our above instance
                  name='ols',              # Don't reuse name
                  attr='predict',          # Attribute for prediction
                  scorer=None,             # To get cv scores
                  output_columns={0: 0},   # Prediction feature index
                  verbose=True)

######################################################################
# .. currentmodule: mlens.index
#
# The ``name`` gives the learner a cache reference. When the learner is
# constructed by the high-level API , the name is guaranteed to be unique, when
# you use the low-level API this is your responsibility. The ``output_columns``
# tells the learner which column index in an output array it should populate
# when predicting: ``attr`` tells the learner which method to use.
# The output_columns can contain several entries if your indexer creates
# partitions (see :class:`SubsetIndex` and :class:`ClusteredSubsetIndex`).
#
# .. currentmodule: mlens.parallel.learner
#
# The learner doesn't do any heavy lifting itself, it manages the creation
# of auxiliary :class:`SubLearner` nodes for each fold during estimation.
# This process is dynamic: the sub-learner is a temporary instance that
# perform the estimation asked of it and caches any output. So to fit
# a learner, we first fit the indexer, then iterate through each of the
# sub-learners created for the task:

import os, tempfile
import numpy as np

X = np.arange(20).reshape(10, 2)
y = np.random.rand(10)

# Fit the indexer to data to create fold indexes
indexer.fit(X)

# Specify a cache directory
path = tempfile.TemporaryDirectory(dir=os.getcwd())

# Declare which type of job (fit, predict, transform)
for sub_learner in learner('fit', X, y):
    sub_learner('fit', path.name)

print("Cached items:\n%r" % os.listdir(path.name))

############################################################################
# Fitting the learner puts three copies of the OLS estimator in the ``path``
# directory: one for each fold and one for the full dataset.
# These are named as ``[name]__[col_id]__[fold_id]``. To load these into the
# learner, call ``collect``.

learner.collect(path.name)

############################################################################
# The main estimator, fitted on all data, gets stored into the
# ``fitted_learner`` attribute, while the others are stored in the
# ``fitted_sublearners``. These attributes are generators that will
# iterate over each fitted estimator and yield a deep copy of them.

############################################################################
# So to generate predictions, we can either use the ``fitted_sublearners``
# generator create cross-validated predictions, or ``fitted_learner``
# generator to generate predictions for the whole input set.

############################################################################
# But to generate predictions, the learner needs an output array to populate.
# In particular, the learner will populate the columns given in the
# ``output_columns`` parameter. Here, we use the ``transform`` task, which
# uses the ``fitted_sublearners`` generator to produce cross-validated
# predictions.

P = np.zeros((y.shape[0], 1))
for sub_learner in learner('transform', X, P):
    sub_learner('transform', path.name)
    print('P:')
    print(P)
    print()

############################################################################
# In the above loop, a sub-segment of ``P`` is updated by each sublearner
# spawned by the learner. To instead produce predictions for the full
# dataset using the final estimator, task the learner to ``predict``.

############################################################################
# ML-Ensemble follows the Scikit-learn API, so if you wish to update any
# hyper-parameters of the estimator, use the ``get_params`` and ``set_params``
# API:

print("Params before:")
print(learner.get_params())

learner.set_params(estimator__offset=1, indexer__n_splits=3)

print("Params after:")
print(learner.get_params())

############################################################################
#
# .. note:: Updating the indexer on one learner updates the indexer on all
#  learners that where initiated with the same instance.

############################################################################
#
# Partitioning
# ------------
#
# We can create several other types of learners by
# varying the estimation strategy. An especially interesting strategy is to
# partition the training set and create several learners fitted on a given
# partition. This will create one prediction feature per partition
# So we now need to specify in the ``output_columns`` dict which partition
# is given which column in the output array.
# Here, we fit the OLS model using two partitions and two fold CV on each
# partition. Note that by passing the output array to the sub-learner
# during fitting, we get predictions immediately.

from mlens.index import SubsetIndex

indexer = SubsetIndex(n_partitions=2, n_splits=2, X=X)
learner = Learner(estimator=OLS(),
                  preprocess=None,
                  indexer=indexer,
                  name='ols',
                  attr='predict',
                  scorer=None,
                  output_columns={0: 0, 1: 1},
                  verbose=True)

# P needs 2 cols
P = np.zeros((y.shape[0], 2))

# Pass P during 'fit' to get prediction immediately
for sub_learner in learner('fit', X, y, P):
    sub_learner.fit(path.name)
    print('P:')
    print(P)
    print()

learner.collect(path.name)

############################################################################
# Each sub-learner records fit and predict times during fitting, and if
# a scorer is passed scores the predictions as well. The learner aggregates
# this data into a ``raw_data`` list, and a tabular ``data`` attribute:

print("Data:\n %s" % learner.data)

############################################################################
#
# Preprocessing
# -------------
#
# In general, several estimators share the same preprocessing pipeline,
# so we don't want
# to pass the object itself along, or we risk conflicts. Instead,
# the learner is given a pointer to the caches preprocessing pipeline so that
# it can load when needed. To facilitate preprocessing across several learners,
# we need new type of node, the :class:``Transformer``. This class behaves
# similarly to the learner, but differs in that it doesn't output any
# predictions or transformations, but merely fits and caches the preprocessing
# pipelines. The primary reason for this design is that the transformer would
# need to a transformed copy of the input data for each fold, which would
# quickly result in massive memory consumption.

############################################################################
# So to construct a learner with preprocessing, we begin by constructing the
# transformer.

from mlens.utils.dummy import Scale
from mlens.parallel import Transformer

transformer = Transformer(pipeline=[('trans', Scale())],
                          indexer=indexer,
                          name='sc',
                          verbose=True)

############################################################################
# Now, to build the learner we now pass the ``name`` of the transformer as
# the ``preprocess`` argument to the learner. Here', we'll also include a
# scoring function.

def mse(y, p): return np.mean((y - p) ** 2)

learner = Learner(estimator=OLS(),
                  preprocess='sc',
                  indexer=indexer,
                  name='ols',
                  attr='predict',
                  scorer=mse,
                  output_columns={0: 0, 1: 1},
                  verbose=True)

###########################################################################
# To fit the learner, we must first fit the transformer. Both follow the
# same API, so we simply repeat the above step for each instance.

P = np.zeros((y.shape[0], 2))

for st in transformer('fit', X, y):
    st('fit', path.name)

for lr in learner('fit', X, y, P):
    lr('fit', path.name)

transformer.collect(path.name)
learner.collect(path.name)

############################################################################
# Note that the cache now contains the transfomers as well:

print("Cache: %r" % os.listdir(path.name))


############################################################################
#
# Estimation Data
# ---------------
#
# When fitting the learner, data is collected and stored on a case, estimator
# and partition basis. Standard data is fit time (``ft``), predict time (``pr``)
# and if applicable, test set prediction scores. Since we use cross-validated
# estimation, we get mean (``-m``) and standard deviation (``-s``) for free.

print("Data:\n%s" % learner.data)

############################################################################
# The data is stored as a ``dict`` that prints in tabular
# format for readability. You can however also pass the ``data`` attribute
# to a :class:`pandas.DataFrame` if you wish.

############################################################################
# Next we handle several learners by grouping them in a layer in the
# :ref:`layer mechanics tutorial <layer_tutorial>`.
