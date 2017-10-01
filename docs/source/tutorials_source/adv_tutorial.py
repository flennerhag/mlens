"""
.. _ensemble-tutorial:

Advanced High-level API Tutorials
=================================

The following tutorials highlight advanced functionality and provide in-depth
material on ensemble APIs.

===============================  ==============================================
Tutorial                         Content
===============================  ==============================================
:ref:`propa-tutorial`            Propagate feature input features through layers
\                                to allow several layers to see the same input.
:ref:`proba-tutorial`            Build layers that output class probabilities from each base
\                                learner so that the next layer or meta estimator learns
\                                from probability distributions.
:ref:`subsemble-tutorial`        Learn homogenous partitions of feature space
\                                that maximize base learner's performance on each partition.
:ref:`sequential-tutorial`       How to build ensembles with different layer classes
:ref:`memory-tutorial`           Avoid loading data into the parent process by specifying a
\                                file path to a memmaped array or a csv file.
:ref:`model-selection-tutorial`  Build transformers that replicate layers in ensembles for
\                                model selection of higher-order layers and / or meta learners.
===============================  ==============================================

We use the same preliminary settings as in the
:ref:`getting started <getting-started>` section.
"""
import numpy as np
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

seed = 2017
np.random.seed(seed)

data = load_iris()
idx = np.random.permutation(150)
X = data.data[idx]
y = data.target[idx]

##############################################################################
# .. _propa-tutorial:
#
# Propagating input features
# --------------------------
#
# When stacking several layers of base learners, the variance of the input
# will typically get smaller as learners get better and better at predicting
# the output and the remaining errors become increasingly difficult to correct
# for. This multicolinearity can significantly limit the ability of the
# ensemble to improve upon the best score of the subsequent layer as there is too
# little variation in predictions for the ensemble to learn useful combinations.
# One way to increase this variation is to propagate features from the original
# input and / or earlier layers. To achieve this in ML-Ensemble, we use the
# ``propagate_features`` attribute. To see how this works, let's compare
# a three-layer ensemble with and without feature propagation.
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_ensemble(incl_meta, propagate_features=None):
    """Return an ensemble."""
    if propagate_features:
        n = len(propagate_features)
        propagate_features_1 = propagate_features
        propagate_features_2 = [i for i in range(n)]
    else:
        propagate_features_1 = propagate_features_2 = None

    estimators = [RandomForestClassifier(random_state=seed), SVC()]

    ensemble = SuperLearner()
    ensemble.add(estimators, propagate_features=propagate_features_1)
    ensemble.add(estimators, propagate_features=propagate_features_2)

    if incl_meta:
        ensemble.add_meta(LogisticRegression())
    return ensemble

##############################################################################
# Without feature propagation, the meta learner will learn from the predictions
# of the penultimate layers:
base = build_ensemble(False)
base.fit(X, y)
pred = base.predict(X)[:5]
print("Input to meta learner :\n %r" % pred)

##############################################################################
# When we propagate features, some (or all) of the input seen by one layer is
# passed along to the next layer. For instance, we can propagate some or all of
# the input array through our two intermediate layers to the meta learner input
# of the meta learner:
base = build_ensemble(False, [1, 3])
base.fit(X, y)
pred = base.predict(X)[:5]
print("Input to meta learner :\n %r" % pred)

##############################################################################
# In this scenario, the meta learner will see noth the predictions made by the
# penultimate layer, as well as the second and fourth feature of the original
# input. By propagating
# features, the issue of multicolinearity in deep ensembles can be mitigated.
# In particular, it can give the meta learner greater opportunity to identify
# neighborhoods in the original feature space where base learners struggle. We
# can get an idea of how feature propagation works with our toy example. First,
# we need a simple ensemble evaluation routine. In our case, propagating the
# original features through two layers of the same
# library of base learners gives a dramatic increase in performance on the test
# set:

def evaluate_ensemble(propagate_features):
    """Wrapper for ensemble evaluation."""
    ens = build_ensemble(True, propagate_features)
    ens.fit(X[:75], y[:75])
    pred = ens.predict(X[75:])
    return accuracy_score(pred, y[75:])


score_no_prep = evaluate_ensemble(None)
score_prep = evaluate_ensemble([0, 1, 2, 3])
print("Test set score no feature propagation  : %.3f" % score_no_prep)
print("Test set score with feature propagation: %.3f" % score_prep)

##############################################################################

# .. py:currentmodule:: mlens.preprocessing
#
# By combining feature propagation with the :class:`Subset` transformer, you can
# propagate the feature through several layers without any of the base estimators
# in those layers seeing the propagated features. This can be desirable if you
# want to propagate the input features to the meta learner without intermediate
# base learners always having access to the original input data. In this case,
# we specify propagation as above, but add a preprocessing pipeline to
# intermediate layers:

from mlens.preprocessing import Subset

estimators = [RandomForestClassifier(random_state=seed), SVC()]
ensemble = SuperLearner()

# Initial layer, propagate as before
ensemble.add(estimators, propagate_features=[0, 1])

# Intermediate layer, keep propagating, but add a preprocessing
# pipeline that selects a subset of the input
ensemble.add(estimators,
             preprocessing=[Subset([2, 3])],
             propagate_features=[0, 1])

##############################################################################
# In the above example, the two first features of the original input data
# will be propagated through both layers, but the second layer will not be
# trained on it. Instead, it will only see the predictions made by the base
# learners in the first layer.

ensemble.fit(X, y)
n = ensemble.layer_2.estimators_[0][1].feature_importances_.shape
m = ensemble.predict(X).shape[1]
print("Num features seen by estimators in intermediate layer: %i" % n)
print("Num features in the output array of the intermediate layer: %i" % m)

##############################################################################
# .. _proba-tutorial:
#
# Probabilistic ensemble learning
# -------------------------------
#
# When the target to predict is a class label, it can often be beneficial to
# let higher-order layers or the meta learner learn from *class probabilities*,
# as opposed to the predicted class. Scikit-learn classifiers can return a
# matrix that, for each observation in the test set, gives the probability that
# the observation belongs to the a given class. While we are ultimately
# interested in class membership, this information is much richer that just
# feeding the predicted class to the meta learner. In essence, using class
# probabilities allow the meta learner to weigh in not just the predicted
# class label (the highest probability), but also with what confidence each
# estimator makes the prediction, and how estimators consider the alternative.
# First, let us set a benchmark ensemble performance when learning is by
# predicted class membership.

from mlens.ensemble import BlendEnsemble
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def build_ensemble(proba, **kwargs):
    """Return an ensemble."""
    estimators = [RandomForestClassifier(random_state=seed),
                  SVC(probability=proba)]

    ensemble = BlendEnsemble(**kwargs)
    ensemble.add(estimators, proba=proba)   # Specify 'proba' here
    ensemble.add_meta(LogisticRegression())

    return ensemble

##############################################################################
# As in the :ref:`ensemble guide <ensemble-guide>`, we fit on the first half,
# and test on the remainder.

ensemble = build_ensemble(proba=False)
ensemble.fit(X[:75], y[:75])
preds = ensemble.predict(X[75:])
print("Accuracy:\n%r" % accuracy_score(preds, y[75:]))

##############################################################################
# Now, to enable probabilistic learning, we set ``proba=True`` in the ``add``
# method for all layers except the final meta learner layer.

ensemble = build_ensemble(proba=True)
ensemble.fit(X[:75], y[:75])
preds = ensemble.predict(X[75:])
print("\nAccuracy:\n%r" % accuracy_score(preds, y[75:]))

##############################################################################
# In this case, using probabilities has a drastic effect on predictive
# performance, increasing some 40 percentage points. For an applied example
# see the ensemble used to beat the Scikit-learn :ref:`mnist` benchmark.
#
# .. _subsemble-tutorial:
#
# Advanced Subsemble techniques
# -----------------------------
#
# .. currentmodule:: mlens.ensemble
#
# Subsembles leverages the idea that neighborhoods of feature space have a
# specific local structure. When we fit an estimator across all feature space,
# it is very hard to capture several such local properties. Subsembles partition
# the feature space and fits each base learner to each partitions, thereby
# allow base learners to optimize locally. Instead, the task of generalizing
# across neighborhoods is left to the meta learner. This strategy can be very
# powerful when the local structure first needs to be extracted, before an
# estimator can learn to generalize. Suppose you want to learn the probability
# distribution of some variable :math:`y`. Often, the true distribution is
# multi-modal, which is an extremely hard problem. In fact, most
# machine learning algorithms, especially with convex optimization objectives, are
# ill equipped to solve this problem. Subsembles can overcome this issue allowing
# base estimators to fit one mode of the distribution at a time, which yields a
# better representation of the distribution and greatly facilitates the learning
# problem of the meta learner.
#
# .. py:currentmodule:: mlens.ensemble
#
# By default, the :class:`Subsemble` class partitioning the dataset randomly.
# Note however that partitions are created on the data "as is", so if the ordering
# of observations is not random, neither will the partitioning be. For this
# reason, it is recommended to shuffle the data (e.g. via the ``shuffle``
# option at initialization). To build a subsemble with random partitions, the
# only parameter needed is the number of ``partitions`` when instantiating
# the :class:`Subsemble`.

from mlens.ensemble import Subsemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def build_subsemble():
    """Build a subsemble with random partitions"""
    sub = Subsemble(partitions=3, folds=2)
    sub.add([SVC(), LogisticRegression()])
    return sub

sub= build_subsemble()
sub.fit(X, y)
s = sub.predict(X[:10]).shape[1]
print("No. prediction features: %i " % s)

##############################################################################
# During training, the base learners are copied to each partition,
# so the output of each layer gets multiplied by the number of partitions. In this
# case, we have 2 base learners for 3 partitions, giving 6 prediction features.

##############################################################################
# By creating partitions, subsembles :ref:`scale significantly better <bench>`
# than the
# :class:`SuperLearner`, but in contrast to :class:`BlendEnsemble`,
# the full training data is leveraged during training. But randomly partitioning
# the data does however not exploit the full advantage of locality, since it is
# only by luck that we happen to create such partitions. A better way is to
# *learn* how to best partition the data. We can either use
# unsupervised algorithms to generate clusters, or supervised estimators and
# create partitions based on their predictions. In ML-Ensemble, this is
# achieved by passing an estimator as ``partition_estimator``. This estimator
# can differ between layers.

##############################################################################
# .. currentmodule:: mlens.index
#
# Very few limitation are imposed on the estimator: you can specify whether
# you want to fit it before generating partitions, whether to use
# labels in the partitioning, and what method to call to generate the
# partitions. See :class:`ClusteredSubsetIndex` for the full documentation.
# This level of generality does impose some
# responsibility on the user. In particular, it is up to the user to ensure that
# sensible partitions are created. Problems to watch out for is too small
# partitions (too many clusters, too uneven cluster sizes) and clusters with too
# little variation: for instance with only a single class label in the entire
# partition, base learners have nothing to learn. Let's see how to do this in
# practice. For instance, we can use an unsupervised K-Means
# clustering estimator to partition the data, like so:

from sklearn.cluster import KMeans

def build_clustered_subsemble(estimator):
    """Build a subsemble with random partitions"""
    sub = Subsemble(partitions=2,
                    partition_estimator=estimator,
                    folds=2, verbose=2)

    sub.add([SVC(), LogisticRegression()])
    sub.add_meta(SVC())
    return sub

sub = build_clustered_subsemble(KMeans(2))
sub.fit(X[:, [0, 1]], y)

##############################################################################
# The Iris dataset can actually separate the classes perfectly with a KMeans
# estimator which leads to zero label variation in each partition. For that
# reason the above code fits the KMeans estimator on only the first two
# columns. But this approach is nota very good way of doing it since we loose
# the rest of the data when fitting the estimators too. Instead, we could
# customize the
# partitioning estimator to make the subset selection itself. For instance,
# we can use Scikit-learn's :class:`sklearn.pipeline.Pipeline`
# class to put a dimensionality reduction transformer before the partitioning
# estimator, such as a :class:`sklearn.decomposition.PCA`, or the
# :class:`mlens.preprocessing.Subset` transformer to drop some features before
# estimation. We then use this pipeline as a our partition estimator and fit
# the subsemble on all features.

from mlens.preprocessing import Subset
from sklearn.pipeline import make_pipeline

# This partition estimator is equivalent to the one used above
pe = make_pipeline(Subset([0, 1]), KMeans(2))
sub = build_clustered_subsemble(pe)

sub.fit(X, y)

##############################################################################
# In general, you may need to wrap an estimator around a custom class to modify
# it's output to generate good partitions. For instance, in regression problems,
# the output of a supervised estimator needs to be binarized to give a discrete
# number of partitions. Here's minimalist way of wrapping a Scikit-learn
# estimator:

from sklearn.linear_model import LinearRegression

class MyClass(LinearRegression):

    def __init__(self, **kwargs):
        super(MyClass, self).__init__(**kwargs)

    def fit(self, X, y):
        """Fit estimator."""
        super(MyClass, self).fit(X, y)
        return self

    def predict(self, X):
        """Generate partition"""
        p = super(MyClass, self).predict(X)
        return 1 * (p > p.mean())

##############################################################################
# Finally, to summarize the functionality in one example,
# let's implement a simple (but rather useless) partition estimator that splits
# the data in half based on the sum of the features.

class SimplePartitioner():

    def __init__(self):
        pass

    def our_custom_function(self, X, y=None):
        """Split the data in half based on the sum of features"""
        # Labels should be numerical
        return 1 * (X.sum(axis=1) > X.sum(axis=1).mean())

# Note that the number of partitions the estimator creates *must* match the
# ``partitions`` argument passed to the Subsemble.
# The ``folds`` option is completely independent.

sub = Subsemble(partitions=2, folds=3, verbose=1)
sub.add([SVC(), LogisticRegression()],
        partition_estimator=SimplePartitioner(),
        fit_estimator=False,
        attr="our_custom_function")

sub.fit(X, y)

##############################################################################
# A final word of caution. When implementing custom estimators from scratch, some
# care needs to be taken if you plan on copying the Subsemble. It is advised that
# the estimator inherits the :class:`sklearn.base.BaseEstimator` class to
# provide a Scikit-learn compatible interface. For further information,
# see the :ref:`API` documentation of the :class:`Subsemble`
# and :class:`mlens.base.indexer.ClusteredSubsetIndex`.
#
# For an example of using clustered subsemble, see the subsemble
# used to beat the Scikit-learn :ref:`mnist` benchmark.
#
# .. _sequential-tutorial:
#
# General multi-layer ensemble learning
# -------------------------------------
#
# .. currentmodule:: mlens.ensemble
#
# The modular ``add`` API of ML-Ensembles allow users to build arbitrarily
# deep ensembles. If you would like to alternate between the *type* of each layer
# the :class:`SequentialEnsemble` class can be used to specify what type of
# layer (i.e. stacked, blended, subsamle-style) to add. This can be particularly
# powerful if facing a large dataset, as the first layer can use a fast approach
# such as blending, while subsequent layers fitted on the remaining data can
# use more computationally intensive approaches. The type of layer, along with
# any parameter settings pertaining to that layer, are specified in the
# ``add`` method.

from mlens.ensemble import SequentialEnsemble

ensemble = SequentialEnsemble()

# The initial layer is a blended layer, same as a layer in the BlendEnsemble
ensemble.add('blend',
             [SVC(), RandomForestClassifier(random_state=seed)])

# The second layer is a stacked layer, same as a layer of the SuperLearner
ensemble.add('stack', [SVC(), RandomForestClassifier(random_state=seed)])

# The third layer is a subsembled layer, same as a layer of the Subsemble
ensemble.add('subsemble', [SVC(), RandomForestClassifier(random_state=seed)])

# The meta estimator is added as in any other ensemble
ensemble.add_meta(SVC())

##############################################################################
# The below table maps the types of layers available in the :class:`SequentialEnsemble` with the corresponding ensemble.
#
# ===================  ============================
# front-end parameter  SequentialEnsemble parameter
# ===================  ============================
# 'SuperLearner'       'stack'
# 'BlendEnsemble'      'blend'
# 'Subsemble'          'subsemble'
# ===================  ============================
#
# Once instantiated, the :class:`SequentialEnsemble`` behaves as expect:

preds = ensemble.fit(X[:75], y[:75]).predict(X[75:])
accuracy_score(preds, y[75:])

##############################################################################
# In this case, the multi-layer :class:`SequentialEnsemble` with an initial
# blended layer and second stacked layer achieves similar performance as the
# :class:`BlendEnsemble` with probabilistic learning. Note that we could have
# made any of the layers probabilistic by setting ``Proba=True``.
#
# .. _memory-tutorial:
#
# Passing file paths as data input
# --------------------------------
#
# With large datasets, it can be expensive to load the full data into memory as
# a numpy array. Since ML-Ensemle uses a memmaped cache, the need to keep the
# full array in memory can be entirely circumvented by passing a file path as
# entry to ``X`` and ``y``. There are two important things to note when doing
# this.

##############################################################################
# First, ML-Ensemble delpoys Scikit-learn's array checks, and passing a
# string will cause an error. To avoid this, the ensemble must be initialized
# with ``array_check=0``, in which case there will be no checks on the array.
# The user should make certain that the the data is approprate for esitmation,
# by converting missing values and infinites to numerical representation,
# ensuring that all features are numerical, and remove any headers,
# index columns and footers.
#
# Second, ML-Ensemble expects the file to be either a ``csv``,
# an ``npy`` or ``mmap`` file and will treat these differently.
#
#     - If a path to a ``csv`` file is passed, the ensemble will first **load**
#       the file into memory, then dump it into the cache, before discarding the
#       file from memory by replacing it with a pointer to the memmaped file.
#       The loading module used for the ``csv``
#       file is the :func:`numpy.loadtxt` function.
#
#     - If a path to a ``npy`` file is passed, a memmaped pointer to it will be
#       loaded.
#
#     - If a path to a ``mmap`` file is passed, it will be used as the memmaped
#       input array for estimation.
#
import os
import gc
import tempfile

# We create a temporary folder in the current working directory
temp = tempfile.TemporaryDirectory(dir=os.getcwd())

# Dump the X and y array in the temporary directory, here as csv files
fx = os.path.join(temp.name, 'X.csv')
fy = os.path.join(temp.name, 'y.csv')

np.savetxt(fx, X)
np.savetxt(fy, y)

# We can now fit any ensemble simply by passing the file pointers ``fx`` and
# ``fy``. Remember to set ``array_check=0``.
ensemble = build_ensemble(False, array_check=0)
ensemble.fit(fx, fy)
preds = ensemble.predict(fx)
print(preds[:10])

##############################################################################
# If you are following the examples on your machine,
# don't forget to remove the temporary directory.

try:
    temp.cleanup()
    del temp
except OSError:
    # This can fail on Windows
    pass

##############################################################################
# .. _model-selection-tutorial:
#
# Ensemble model selection
# ------------------------
#
# Ensembles benefit from a diversity of base learners, but often it is not clear
# how to parametrize the base learners. In fact, combining base learners with
# lower predictive power can often yield a superior ensemble. This hinges on the
# errors made by the base learners being relatively uncorrelated, thus allowing
# a meta estimator to learn how to overcome each model's weakness. But with
# highly correlated errors, there is little for the ensemble to learn from.

##############################################################################
# To fully exploit the learning capacity in an ensemble, it is beneficial to
# conduct careful hyper parameter tuning, treating the base learner's parameters
# as the parameters of the ensemble. By far the most critical part of the
# ensemble is the meta learner, but selecting an appropriate meta learner can be
# an ardous task if the entire ensemble has to be evaluated each time.
#
# .. py:currentmodule:: mlens.preprocessing
#
# The :class:`EnsembleTransformer` can be leveraged to treat the initial
# layers of the ensemble as preprocessing. Thus, a copy of the transformer is
# fitted once on each fold, and any model selection will use these pre-fits to
# convert raw input to prediction matrices that corresponds to the output of the
# specified ensemble.
#
# .. py:currentmodule:: mlens.ensemble
#
# The transformer follows the same API as the :class:`SequentialEnsemble`, but
# does not implement a meta estimator and has a transform method that recovers
# the prediction matrix from the ``fit`` call. In the following example,
# we run model selection on the meta learner of a blend ensemble, and try
# two configurations of the blend ensemble: learning from class predictions or
# from probability distributions over classes.

from mlens.model_selection import Evaluator
from mlens.metrics import make_scorer
from scipy.stats import uniform, randint

# Set up two competing ensemble bases as preprocessing transformers:
# one stacked ensemble base with proba and one without
base_learners = [RandomForestClassifier(random_state=seed),
                 SVC(probability=True)]

proba_transformer = SuperLearner(random_state=seed).add(base_learners, proba=True)
class_transformer = SuperLearner(random_state=seed).add(base_learners, proba=False)

# Set up a preprocessing mapping
# Each pipeline in this map is fitted once on each fold before
# evaluating candidate meta learners.
preprocessing = {'proba': [('layer-1', proba_transformer)],
                 'class': [('layer-1', class_transformer)]}

# Set up candidate meta learners
# We can specify a dictionary if we wish to try different candidates on
# different cases, or a list if all estimators should be run on all
# preprocessing pipelines (as in this example)
meta_learners = [SVC(), ('rf', RandomForestClassifier(random_state=seed))]

# Set parameter mapping
# Here, we differentiate distributions between cases for the random forest
params = {'svc': {'C': uniform(0, 10)},
          ('class', 'rf'): {'max_depth': randint(2, 10)},
          ('proba', 'rf'): {'max_depth': randint(2, 10),
                            'max_features': uniform(0.5, 0.5)}
          }

scorer = make_scorer(accuracy_score)
evaluator = Evaluator(scorer=scorer, random_state=seed, cv=2)

evaluator.fit(X, y, meta_learners, params, preprocessing=preprocessing, n_iter=5)

##############################################################################
# We can now compare the performance of the best fit for each candidate
# meta learner.

print("Results:\n%s" % evaluator.results)
