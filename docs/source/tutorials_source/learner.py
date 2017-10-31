# -*- coding: utf-8 -*-
"""

.. _learner_tutorial:


.. currentmodule:: mlens.parallel.learner

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
track of which preprocessing pipeline to use (if any). A learner is a parent node
in a computational sub-graph induced by the cross-validation strategy.
"""
from mlens.utils.dummy import OLS
from mlens.parallel import Learner
from mlens.index import FoldIndex


indexer = FoldIndex(folds=2)               # Define a training strategy
learner = Learner(estimator=OLS(),         # Declare estimator
                  preprocess=None,         # We'll get to this
                  indexer=indexer,         # Our above instance
                  name='ols',              # Don't reuse name
                  attr='predict',          # Attribute for prediction
                  scorer=None,             # To get cv scores
                  output_columns={0: 0},   # Prediction feature index
                  verbose=True)

######################################################################
# .. currentmodule:: mlens.index
#
# The ``name`` gives the learner a cache reference. When the learner is
# constructed by the high-level API , the name is guaranteed to be unique, but here
# you must ensure all learner names are unique. The ``output_columns``
# tells the learner which column index in an output array it should populate
# when predicting. This helps us rapidly creating prediction with several learners. When
# we have a unique prediction array use ``{0: 0}``. When the training strategy creates
# partitions, we need to map ``output_columns`` for each partition. We'll see an example of this below.
# The ``attr`` argument tells the learner which method to use.
#
# .. currentmodule:: mlens.parallel.learner
#
# The learner doesn't do any heavy lifting itself, it manages the creation a sub-graph
# of auxiliary :class:`SubLearner` nodes for each fold during estimation.
# This process is dynamic: the sub-learners are temporary instance created for each
# estimation. To fit a learner, we first fit the indexer, then iterate through each of the
# sub-learners created for the task:

import os, tempfile
import numpy as np

X = np.arange(20).reshape(10, 2)
y = np.random.rand(10)

# Fit the indexer to data to create fold indexes
indexer.fit(X)

# Specify a cache directory
path = tempfile.TemporaryDirectory(dir=os.getcwd())

# Declare which type of job (fit, predict, transform) to run on which data
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
# ``learner_`` attribute, while the others are stored in the
# ``sublearners_``. These attributes are *generators* that create
# sub-learners on-the-fly from cached fitted estimators when called upon.

############################################################################
# To generate predictions, we can either use the ``sublearners_``
# generator create cross-validated predictions, or ``learner_``
# generator to generate predictions for the whole input set.

############################################################################
# Similarly to above, we predict by specifying the job and the data to use.
# Note that now we also specify the output array to populate.
# In particular, the learner will populate the columns given in the
# ``output_columns`` parameter. Here, we use the ``transform`` task, which
# uses the ``sublearners_`` generator to produce cross-validated
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
# dataset using the estimator fitted on all training data,
# task the learner to ``predict``.

P = np.zeros((y.shape[0], 1))
for sub_learner in learner('predict', X, P):
    sub_learner('predict', path.name)
    print('P:')
    print(P)
    print()

############################################################################
# ML-Ensemble follows the Scikit-learn API, so if you wish to update any
# hyper-parameters of the estimator, use the ``get_params`` and ``set_params``
# API:

print("Params before:")
print(learner.get_params())

learner.set_params(estimator__offset=1, indexer__folds=3)

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
# partition. This will create one prediction feature per partition.
# The learner handles the computational graph for us, all we need to
# input is a mapping between partitions and output columns in the
# ``output_columns`` dict. In the following example we fit the OLS model
# using two partitions and three fold CV on each
# partition. Note that by passing the output array as an argument during ``'fit'``,
# we get predictions immediately.

from mlens.index import SubsetIndex

def mse(y, p): return np.mean((y - p) ** 2)

indexer = SubsetIndex(partitions=2, folds=2, X=X)
learner = Learner(estimator=OLS(),
                  preprocess=None,
                  indexer=indexer,
                  name='ols',
                  attr='predict',
                  scorer=mse,
                  output_columns={0: 0,    # First partition
                                  1: 1},   # Second partition
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
# this data into a ``raw_data`` attribute in the form of a list.
# More conveniently, the ``data`` attribute returns a dict with a specialized
# representation that gives a tabular output directly:
# Standard data is fit time (``ft``), predict time (``pr``).
# If a scorer was passed to the learner, cross-validated test set prediction
# scores are computed. For brevity, ``-m`` denotes the mean and ``-s``
# denotes standard deviation.

print("Data:\n%s" % learner.data)

############################################################################
#
# Preprocessing
# -------------
#
# We can easily create a preprocessing pipeline before fitting the estimator.
# In general, several estimators will share the same preprocessing pipeline,
# so we don't want to pipeline the transformations in the estimator itself–
# this will result in duplicate transformers.
# The learner accepts a ``preprocess`` argument that points it to reference in
# the estimation cache, and wil load the cached transformer for the given fold
# when running an estimation. This does mean that the input will be processed
# for each estimator and each fold, but pre-processing the data and storing the
# data does not scale as memory consumption grows exponentially.
# In contrast, running (not fitting) a transformer pipeline is often an
# efficient operation that introduce only a minor overhead on computation time.

############################################################################
# To facilitate preprocessing across several learners,
# we need new type of node, the :class:`Transformer`. This class behaves
# similarly to the learner, but differs in that it doesn't output any
# predictions or transformations, but merely fits a pipeline and caches it
# for the learner to load when needed. To construct a learner with
# a preprocessing pipeline, we begin by constructing the
# transformer.

from mlens.utils.dummy import Scale
from mlens.parallel import Transformer

transformer = Transformer(pipeline=[('trans', Scale())],  # the pipeline should be a list of transformers
                          indexer=indexer,
                          name='sc',
                          verbose=True)

############################################################################
# Now, to build the learner we now pass the ``name`` of the transformer as
# the ``preprocess`` argument to the learner.

learner = Learner(estimator=OLS(),
                  preprocess='sc',
                  indexer=indexer,
                  name='ols',
                  attr='predict',
                  scorer=mse,
                  output_columns={0: 0, 1: 1},
                  verbose=True)

###########################################################################
# We now repeat the above process to fit the learner, starting with fitting
# the transformer. Both follow the same API.

P = np.zeros((y.shape[0], 2))

for st in transformer('fit', X, y):
    st('fit', path.name)

for lr in learner('fit', X, y, P):
    lr('fit', path.name)

transformer.collect(path.name)
learner.collect(path.name)

############################################################################
# Note that the cache now contains the transformers as well:

print("Cache: %r" % os.listdir(path.name))

############################################################################
# Data is collected on a partition basis:
print("Data:\n%s" % learner.data)

############################################################################
#
# Parallel estimation
# -------------------
#
# Since the learner and transformer class do not perform estimations themselves,
# we are free to modify the estimation behavior. For instance, to parallelize
# estimation with several learners, we don't want a nested loop over each learner,
# but instead flatten the for loops for maximal concurrency.
# This is the topic of our next walkthrough, here we show how to parallelize
# estimation with a single learner. Using the integrated :mod:`joblib` package, we can fit a
# learner in parallel as follow:
from mlens.externals.joblib import Parallel, delayed
from numpy.testing import assert_array_equal

P_t = np.zeros((y.shape[0], 2))

# Since ML-Ensemble is thread-safe, we use threading as P_t is not memmapped.
with Parallel(backend='threading', n_jobs=-1) as parallel:
    parallel(delayed(sublearner, check_pickle=False)('fit', path.name)
             for sublearner in learner('fit', X, y, P_t)
             )

assert_array_equal(P, P_t)

############################################################################
# Joblib is built on top of the :mod:`multiprocessing` package, and we
# can similarly directly use the ``Pool().map()`` API to achieve the same
# result:

# The dummy module wraps the threading package in the multiprocessing API
from multiprocessing.dummy import Pool

def run(tup): tup[0](tup[1], tup[2])

P_p = np.zeros((y.shape[0], 2))
Pool(4).map(run, [(sublearner, 'fit', path.name)
                  for sublearner in learner('fit', X, y, P_p)])

assert_array_equal(P, P_t)

############################################################################
# Next we handle several learners by grouping them in a layer in the
# :ref:`layer mechanics tutorial <layer_tutorial>`.
