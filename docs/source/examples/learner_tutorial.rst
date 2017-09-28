

.. _sphx_glr_examples_learner_tutorial.py:



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



.. code-block:: python

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







.. currentmodule: mlens.index

The ``name`` gives the learner a cache reference. When the learner is
constructed by the high-level API , the name is guaranteed to be unique, when
you use the low-level API this is your responsibility. The ``output_columns``
tells the learner which column index in an output array it should populate
when predicting: ``attr`` tells the learner which method to use.
The output_columns can contain several entries if your indexer creates
partitions (see :class:`SubsetIndex` and :class:`ClusteredSubsetIndex`).

.. currentmodule: mlens.parallel.learner

The learner doesn't do any heavy lifting itself, it manages the creation
of auxiliary :class:`SubLearner` nodes for each fold during estimation.
This process is dynamic: the sub-learner is a temporary instance that
perform the estimation asked of it and caches any output. So to fit
a learner, we first fit the indexer, then iterate through each of the
sub-learners created for the task:



.. code-block:: python


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





.. rst-class:: sphx-glr-script-out

 Out::

    ols__0__0 done | 00:00:00
    ols__0__1 done | 00:00:00
    ols__0__2 done | 00:00:00
    Cached items:
    ['ols__0__0.pkl', 'ols__0__1.pkl', 'ols__0__2.pkl']


Fitting the learner puts three copies of the OLS estimator in the ``path``
directory: one for each fold and one for the full dataset.
These are named as ``[name]__[col_id]__[fold_id]``. To load these into the
learner, call ``collect``.



.. code-block:: python


    learner.collect(path.name)







The main estimator, fitted on all data, gets stored into the
``fitted_learner`` attribute, while the others are stored in the
``fitted_sublearners``. These attributes are generators that will
iterate over each fitted estimator and yield a deep copy of them.


So to generate predictions, we can either use the ``fitted_sublearners``
generator create cross-validated predictions, or ``fitted_learner``
generator to generate predictions for the whole input set.


But to generate predictions, the learner needs an output array to populate.
In particular, the learner will populate the columns given in the
``output_columns`` parameter. Here, we use the ``transform`` task, which
uses the ``fitted_sublearners`` generator to produce cross-validated
predictions.



.. code-block:: python


    P = np.zeros((y.shape[0], 1))
    for sub_learner in learner('transform', X, P):
        sub_learner('transform', path.name)
        print('P:')
        print(P)
        print()





.. rst-class:: sphx-glr-script-out

 Out::

    ols__0__1 done | 00:00:00
    P:
    [[-0.17859605]
     [-0.08281824]
     [ 0.01295956]
     [ 0.10873736]
     [ 0.20451517]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]
     [ 0.        ]]

    ols__0__2 done | 00:00:00
    P:
    [[-0.17859605]
     [-0.08281824]
     [ 0.01295956]
     [ 0.10873736]
     [ 0.20451517]
     [ 1.02793004]
     [ 1.12144248]
     [ 1.21495492]
     [ 1.30846736]
     [ 1.4019798 ]]


In the above loop, a sub-segment of ``P`` is updated by each sublearner
spawned by the learner. To instead produce predictions for the full
dataset using the final estimator, task the learner to ``predict``.


ML-Ensemble follows the Scikit-learn API, so if you wish to update any
hyper-parameters of the estimator, use the ``get_params`` and ``set_params``
API:



.. code-block:: python


    print("Params before:")
    print(learner.get_params())

    learner.set_params(estimator__offset=1, indexer__n_splits=3)

    print("Params after:")
    print(learner.get_params())





.. rst-class:: sphx-glr-script-out

 Out::

    Params before:
    {'attr': 'predict', 'estimator__offset': 0, 'estimator': OLS(offset=0), 'indexer__X': None, 'indexer__n_splits': 2, 'indexer__raise_on_exception': True, 'indexer': FoldIndex(X=None, n_splits=2, raise_on_exception=True), 'name': 'ols', 'output_columns': {0: 0}, 'preprocess': None, 'raise_on_exception': True, 'scorer': None, 'verbose': True}
    Params after:
    {'attr': 'predict', 'estimator__offset': 1, 'estimator': OLS(offset=1), 'indexer__X': None, 'indexer__n_splits': 3, 'indexer__raise_on_exception': True, 'indexer': FoldIndex(X=None, n_splits=3, raise_on_exception=True), 'name': 'ols', 'output_columns': {0: 0}, 'preprocess': None, 'raise_on_exception': True, 'scorer': None, 'verbose': True}


.. note:: Updating the indexer on one learner updates the indexer on all
 learners that where initiated with the same instance.


Partitioning
------------

We can create several other types of learners by
varying the estimation strategy. An especially interesting strategy is to
partition the training set and create several learners fitted on a given
partition. This will create one prediction feature per partition
So we now need to specify in the ``output_columns`` dict which partition
is given which column in the output array.
Here, we fit the OLS model using two partitions and two fold CV on each
partition. Note that by passing the output array to the sub-learner
during fitting, we get predictions immediately.



.. code-block:: python


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





.. rst-class:: sphx-glr-script-out

 Out::

    ols__0__0 done | 00:00:00
    P:
    [[ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]]

    ols__1__0 done | 00:00:00
    P:
    [[ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]]

    ols__0__1 done | 00:00:00
    P:
    [[ 0.59681212  0.        ]
     [ 0.65795198  0.        ]
     [ 0.71909183  0.        ]
     [ 0.          0.        ]
     [ 0.          0.        ]
     [ 0.9025114   0.        ]
     [ 0.96365125  0.        ]
     [ 1.02479111  0.        ]
     [ 0.          0.        ]
     [ 0.          0.        ]]

    ols__0__2 done | 00:00:00
    P:
    [[ 0.59681212  0.        ]
     [ 0.65795198  0.        ]
     [ 0.71909183  0.        ]
     [ 1.29263015  0.        ]
     [ 1.58638516  0.        ]
     [ 0.9025114   0.        ]
     [ 0.96365125  0.        ]
     [ 1.02479111  0.        ]
     [ 2.76140519  0.        ]
     [ 3.0551602   0.        ]]

    ols__1__1 done | 00:00:00
    P:
    [[ 0.59681212 -3.37784119]
     [ 0.65795198 -2.89985018]
     [ 0.71909183 -2.42185916]
     [ 1.29263015  0.        ]
     [ 1.58638516  0.        ]
     [ 0.9025114  -0.98788612]
     [ 0.96365125 -0.5098951 ]
     [ 1.02479111 -0.03190409]
     [ 2.76140519  0.        ]
     [ 3.0551602   0.        ]]

    ols__1__2 done | 00:00:00
    P:
    [[ 0.59681212 -3.37784119]
     [ 0.65795198 -2.89985018]
     [ 0.71909183 -2.42185916]
     [ 1.29263015  0.73410616]
     [ 1.58638516  0.61041277]
     [ 0.9025114  -0.98788612]
     [ 0.96365125 -0.5098951 ]
     [ 1.02479111 -0.03190409]
     [ 2.76140519  0.11563924]
     [ 3.0551602  -0.00805415]]


Each sub-learner records fit and predict times during fitting, and if
a scorer is passed scores the predictions as well. The learner aggregates
this data into a ``raw_data`` list, and a tabular ``data`` attribute:



.. code-block:: python


    print("Data:\n %s" % learner.data)





.. rst-class:: sphx-glr-script-out

 Out::

    Data:
               ft-m  ft-s  pt-m  pt-s
      ols  0   0.0   0.0   0.0   0.0
      ols  1   0.0   0.0   0.0   0.0


Preprocessing
-------------

In general, several estimators share the same preprocessing pipeline,
so we don't want
to pass the object itself along, or we risk conflicts. Instead,
the learner is given a pointer to the caches preprocessing pipeline so that
it can load when needed. To facilitate preprocessing across several learners,
we need new type of node, the :class:``Transformer``. This class behaves
similarly to the learner, but differs in that it doesn't output any
predictions or transformations, but merely fits and caches the preprocessing
pipelines. The primary reason for this design is that the transformer would
need to a transformed copy of the input data for each fold, which would
quickly result in massive memory consumption.


So to construct a learner with preprocessing, we begin by constructing the
transformer.



.. code-block:: python


    from mlens.utils.dummy import Scale
    from mlens.parallel import Transformer

    transformer = Transformer(pipeline=[('trans', Scale())],
                              indexer=indexer,
                              name='sc',
                              verbose=True)







Now, to build the learner we now pass the ``name`` of the transformer as
the ``preprocess`` argument to the learner. Here', we'll also include a
scoring function.



.. code-block:: python


    def mse(y, p): return np.mean((y - p) ** 2)

    learner = Learner(estimator=OLS(),
                      preprocess='sc',
                      indexer=indexer,
                      name='ols',
                      attr='predict',
                      scorer=mse,
                      output_columns={0: 0, 1: 1},
                      verbose=True)







To fit the learner, we must first fit the transformer. Both follow the
same API, so we simply repeat the above step for each instance.



.. code-block:: python


    P = np.zeros((y.shape[0], 2))

    for st in transformer('fit', X, y):
        st('fit', path.name)

    for lr in learner('fit', X, y, P):
        lr('fit', path.name)

    transformer.collect(path.name)
    learner.collect(path.name)





.. rst-class:: sphx-glr-script-out

 Out::

    sc__0__0 done | 00:00:00
    sc__1__0 done | 00:00:00
    sc__0__1 done | 00:00:00
    sc__0__2 done | 00:00:00
    sc__1__1 done | 00:00:00
    sc__1__2 done | 00:00:00
    sc__ols__0__0 done | 00:00:00
    sc__ols__1__0 done | 00:00:00
    sc__ols__0__1 done | 00:00:00
    sc__ols__0__2 done | 00:00:00
    sc__ols__1__1 done | 00:00:00
    sc__ols__1__2 done | 00:00:00


Note that the cache now contains the transfomers as well:



.. code-block:: python


    print("Cache: %r" % os.listdir(path.name))






.. rst-class:: sphx-glr-script-out

 Out::

    Cache: ['ols__0__0.pkl', 'ols__0__1.pkl', 'ols__0__2.pkl', 'ols__1__0.pkl', 'ols__1__1.pkl', 'ols__1__2.pkl', 'sc__0__0.pkl', 'sc__0__1.pkl', 'sc__0__2.pkl', 'sc__1__0.pkl', 'sc__1__1.pkl', 'sc__1__2.pkl', 'sc__ols__0__0.pkl', 'sc__ols__0__1.pkl', 'sc__ols__0__2.pkl', 'sc__ols__1__0.pkl', 'sc__ols__1__1.pkl', 'sc__ols__1__2.pkl']


Estimation Data
---------------

When fitting the learner, data is collected and stored on a case, estimator
and partition basis. Standard data is fit time (``ft``), predict time (``pr``)
and if applicable, test set prediction scores. Since we use cross-validated
estimation, we get mean (``-m``) and standard deviation (``-s``) for free.



.. code-block:: python


    print("Data:\n%s" % learner.data)





.. rst-class:: sphx-glr-script-out

 Out::

    Data:
                  score-m  score-s  ft-m  ft-s  pt-m  pt-s
      sc  ols  0      0.8     0.37   0.0   0.0   0.0   0.0
      sc  ols  1     5.61     4.94   0.0   0.0   0.0   0.0


The data is stored as a ``dict`` that prints in tabular
format for readability. You can however also pass the ``data`` attribute
to a :class:`pandas.DataFrame` if you wish.


Next we handle several learners by grouping them in a layer in the
:ref:`layer mechanics tutorial <layer_tutorial>`.


**Total running time of the script:** ( 0 minutes  0.031 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: learner_tutorial.py <learner_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: learner_tutorial.ipynb <learner_tutorial.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
