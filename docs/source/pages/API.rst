.. Main API page

.. _API:

API
====

The Front-end API collects documentation on ensemble classes, preprocessing transformers, model selection modules and visualization features. The Low-level API presents the core estimation and ensemble management classes. 

Front-end API
-------------

Ensemble estimators
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mlens.ensemble

.. autosummary::

   SuperLearner
   Subsemble
   BlendEnsemble
   SequentialEnsemble

Model Selection
^^^^^^^^^^^^^^^

.. currentmodule:: mlens.model_selection

.. autosummary::

   Evaluator
   EnsembleTransformer

Preprocessing
^^^^^^^^^^^^^

.. currentmodule:: mlens.preprocessing

.. autosummary::

   Shift 
   Subset

Visualization
^^^^^^^^^^^^^

.. currentmodule:: mlens.visualization

.. autosummary::

   corrmat
   clustered_corrmap
   corr_X_y
   pca_plot
   pca_comp_plot
   exp_var_plot


.. _low-level-api:

Low-level API
-------------

.. _estimation-api:

Parallel estimation API 
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mlens.parallel

.. autosummary::

   Learner
   Transformer
   ParallelProcessing
   Layer


Ensemble Base classes
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mlens.ensemble

.. autosummary::

   Sequential
   BaseEnsemble


.. _indexer-api:

Indexers
^^^^^^^^

.. currentmodule:: mlens.index

.. autosummary::

    BlendIndex
    FoldIndex
    FullIndex
    SubsetIndex
    ClusteredSubsetIndex


.. _Scikit-learn: http://scikit-learn.org/stable/
