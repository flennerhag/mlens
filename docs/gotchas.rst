.. Known issues

Troubleshooting
===============

Here we collect a set of subtle potential issues and limitations that may
explain odd behavior that you have encountered. Feel free to reach out if your
problem is not addressed here.

.. _third-party-issues:

Bad interaction with third-party packages
-----------------------------------------

Parallel processing with generic Python objects is a difficult task, and while
ML-Ensemble is routinely tested to function seamlessly with Scikit-learn, other machine
learning libraries can cause bad behaviour during parallel estimations. This
is unfortunately a fundamental problem rooted in how `Python runs processes in parallel`_,
and in particular that Python is not thread-safe. ML-Ensemble is by configured
to avoid such issues to the greatest extent possible, but issues can occur.

In particular, ensemble can run either on multiprocessing or multithreading.
For standard Scikit-learn use cases, the GIL_ can be released and
multithreading used. This will speed up estimation and consume less memory.
However, Python is not inherently thread-safe, so this strategy is not stable.
For this reason, the safest choice to avoid corrupting the estimation process
is to use multiprocessing instead. This requires creating sub-process to run
each job, and so increases additional overhead both in terms of job management
and sharing memory. As of this writing, the default setting in ML-Ensemble is
'multiprocessing', but you can change this variable globally: see :ref:`configs`.

In Python 3.4+, ML-Ensemble defaults to ``'forkserver'`` on unix systems
and ``'spawn'`` on Windows for generating sub-processes. These require more
overhead than the default ``'fork'`` method, but avoids corrupting the thread
state and as such is much more stable against third-party conflict. These
conflicts are caused by each worker thinking they have more threads available
than they actually do, leading to deadlocks and race conditions. For more
information on this issue see the `Scikit-learn FAQ`_.

Array copying during fitting
----------------------------

When the number of folds is greater than 2, it is not possible to slice the
full data in such a way as to return a view_ of that array (i.e. without
copying any data). Hence for fold numbers larger than 2, each subprocess
will in fact trigger a copy of the training data (which can be from 67% to
99% of the full data size). A possible alleviation to this problem is to
memmap the required slices before estimation to avoid creating new copies in
each subprocess. However this will induce the equivalent of several copies of
the underlying data to be persisted to disk and may instead lead to the issue
remaining as a disk-bound issue. Since elementary diagnostics suggest that for
data sizes where memory becomes a constraining factor, increasing the number
of folds beyond 2 does not significantly impact performance and at this time
of writing this is the suggested approach. For further information on
avoiding copying data during estimation, see :ref:`memory`.

File permissions on Windows
---------------------------

During ensemble estimation, ML-Ensemble will create a temporary directory and
populate it with training data and predictions, along with pickled estimators
and transformers. Each subprocess is given an container object that points to
the objects in the directory, and once the estimation is done the temporary
directory is cleaned and removed. The native python execution of the
termination typically fails due to how Windows gives read and write permission
between processes. To overcome this, ML-Ensemble runs an explicit shell command
(``rmdir -s -q dir``) that forcibly removes the cache. Current testing on
development machines indicates this exception handling is successful and
Windows users should not expect any issues. If however you do notice
memory performance issues, create an issue at the `issue tracker`_.

.. _GIL: https://wiki.python.org/moin/GlobalInterpreterLock
.. _view: http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
.. _Python runs processes in parallel: https://wiki.python.org/moin/ParallelProcessing
.. _Scikit-learn FAQ: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
.. _issue tracker: https://github.com/flennerhag/mlens/issues
