.. Known issues

Troubleshooting
===============

Known potential issues. `Raise an issue`_ if your problem is not addressed here.

.. _third-party-issues:

Bad interaction with third-party packages
-----------------------------------------

Due to how `Python forks the main process when running multiprocessing`_, 
workers can receive corrupted thread states prompting them to acquiring more threads than are available, 
with the resulting of a deadlock. Due to this limitation and the additional overhead of multiprocessing, 
ML-Ensemble seeks to be thread safe and run on multithreading which entirely circumvents the issue.

If in spite of this race conditions or crashes occurs, raise an issue at the Github repository.
In the meantime, there are a few things to try that might alleviate the problem:

    #. ensure that all estimators in the ensemble or the evaluator has ``n_jobs`` or ``nthread`` equal to ``1``,
    #. try changing the ``backend`` parameter to either ``threading`` or ``multiprocessing``, 
    #. if using ``multiprocessing``, try varying the start method (see :ref:`configs`).
          
For more information on this in the multiprocessing case see the `Scikit-learn FAQ`_.

Array copying during fitting
----------------------------

When the number of folds is greater than 2, it is not possible to slice the
full data in such a way as to return a view_ of that array (i.e. without
copying any data). Hence for fold numbers larger than 2, each worker 
will in fact trigger a copy of the training data. If you experience memory-bound
issues, please consider using fewer folds during fitting. For further information on
avoiding copying data during estimation, see :ref:`memory`.

.. _GIL: https://wiki.python.org/moin/GlobalInterpreterLock
.. _view: http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
.. _Python forks the main process when running multiprocessing: https://wiki.python.org/moin/ParallelProcessing
.. _Scikit-learn FAQ: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
.. _issue tracker: https://github.com/flennerhag/mlens/issues
.. _Raise an issue: https://github.com/flennerhag/mlens/issues
