.. role:: hidden
    :class: hidden-section

.. automodule:: mlens.parallel
.. currentmodule:: mlens.parallel

mlens.parallel
==============

Graph Nodes
-----------

.. currentmodule:: mlens.parallel.layer

:hidden:`Layer`
^^^^^^^^^^^^^^^

.. autoclass:: Layer 
    :members:
    :show-inheritance:

.. currentmodule:: mlens.parallel.learner

:hidden:`Learner`
^^^^^^^^^^^^^^^^^

.. autoclass:: Learner 
    :members:
    :show-inheritance:

:hidden:`Transformer`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Transformer 
    :members:
    :show-inheritance:

:hidden:`EvalLearner`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EvalLearner 
    :members:
    :show-inheritance:

:hidden:`EvalTransformer`
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EvalTransformer 
    :members:
    :show-inheritance:

:hidden:`BaseNode`
^^^^^^^^^^^^^^^^^^

.. autoclass:: BaseNode 
    :members:
    :show-inheritance:

:hidden:`SubLearner`
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SubLearner 
    :members:
    :show-inheritance:

:hidden:`SubTransformer`
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SubTransformer 
    :members:
    :show-inheritance:

:hidden:`EvalSubTransformer`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EvalSubLearner 
    :members:
    :show-inheritance:

:hidden:`IndexedEstimator`
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: IndexedEstimator 
    :members:
    :show-inheritance:

:hidden:`Cache`
^^^^^^^^^^^^^^^

.. autoclass:: Cache 
    :members:
    :show-inheritance:

Handles
-------

.. currentmodule:: mlens.parallel.handles

:hidden:`Group`
^^^^^^^^^^^^^^^

.. autoclass:: Group 
    :members:
    :show-inheritance:

:hidden:`Pipeline`
^^^^^^^^^^^^^^^^^^

.. autoclass:: Pipeline 
    :members:
    :show-inheritance:

:hidden:`make_group`
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: make_group

Wrappers
--------

.. currentmodule:: mlens.parallel.wrapper

:hidden:`EstimatorMixin`
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EstimatorMixin 
    :members:
    :show-inheritance:

:hidden:`run`
^^^^^^^^^^^^^

.. autofunction:: run 

Backend
-------

.. currentmodule:: mlens.parallel.backend

:hidden:`BaseProcessor`
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BaseProcessor 
    :members:
    :show-inheritance:

:hidden:`ParallelProcessing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ParallelProcessing 
    :members:
    :show-inheritance:

:hidden:`ParallelEvaluation`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ParallelEvaluation 
    :members:
    :show-inheritance:

:hidden:`Job`
^^^^^^^^^^^^^

.. autoclass:: Job 
    :members:
    :show-inheritance:

:hidden:`dump_array`
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dump_array 

Base classes
------------

Schedulers for global setups:

================   ================================   =======================
Order              Setup types                        Function calls 
================   ================================   =======================
0. Base setups     Independent of other features      IndexMixin._setup_0_index

1. Global setups   Reserved for aggregating classes   BaseStacker._setup_1_global

2. General local   Setups Dependents on 0             ProbaMixin.__setup_2_multiplier

3. Conditional     Setups Dependents on 0, 2          OutputMixin.__setup_3__output_columns
================   ================================   =======================


Note that base classes and setup schedulers are experimental 
and may change without a deprecation cycle.

.. currentmodule:: mlens.parallel.base

:hidden:`BaseBackend`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BaseBackend 
    :members:
    :private-members:
    :special-members:
    :show-inheritance:

:hidden:`BaseParallel`
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BaseParallel
    :members:
    :private-members:
    :special-members:
    :show-inheritance:

:hidden:`BaseEstimator`
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BaseEstimator 
    :members:
    :private-members:
    :special-members:
    :show-inheritance:

:hidden:`BaseStacker`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BaseStacker 
    :members:
    :private-members:
    :special-members:
    :show-inheritance:

Mixins
------

:hidden:`ParamMixin`
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ParamMixin 
    :members:
    :private-members:
    :special-members:
    :show-inheritance:


:hidden:`IndexMixin`
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: IndexMixin 
    :members:
    :private-members:
    :special-members:
    :show-inheritance:

:hidden:`OutputMixin`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: OutputMixin 
    :members:
    :private-members:
    :special-members:
    :show-inheritance:

:hidden:`ProbaMixin`
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ProbaMixin 
    :members:
    :private-members:
    :special-members:
    :show-inheritance:
