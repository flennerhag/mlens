Known limitations
=================

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


Third-party multiprocessed objects
----------------------------------

ML-Ensemble runs by default on multi-threading. This requires releasing the
GIL_, which can cause race conditions. In standard uses cases, releasing the
GIL is harmless since input data is shared in read-only mode and output arrays
are partitioned. If you experience issues with multithreading, you can try
switching to multiprocessing either by the ``backend`` argument or by changing
the global default (``mlens.config.BACKEND``). Estimation is then parallelized
on processes instead of threads, and thus keeps the GIL in place. Multiprocessing however
is not without its issues and can interact badly with third-party classes that
are also multiprocessed, which can lead to deadlocks. This issue is due to a
limitation of how `Python runs processes in parallel`_ and is an issue beyond
the scope of ML-Ensemble.

If you experience issues on both multithreading and multiprocessing, the simplest
solution  is to turn off parallelism by setting ``n_jobs`` to ``1``. Start by
switching off parallelism in the learners of the ensemble as this will not
impact the training speed of the ensemble, and only switch off paralllism in the
ensemble as a last resort.

In Python 3.4+, it is possible to spawn a ``forkprocess`` backend within the
native python ``multiprocessing`` library. To do this, set the multiprocessing
start method to ``forkserver`` as below. ::

    import multiprocessing

    # You can put imports and functions/class definitions other than mlens here

    if __name__ == '__main__':

        multiprocessing.set_start_method('forkserver')

        # Import mlens here

        # Your execution here

Note that this solution is currently experimental.
Further information can be found here_.

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
.. _here: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
.. _issue tracker: https://github.com/flennerhag/mlens/issues
