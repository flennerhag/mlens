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

This issue arises when a multit-hreaded python object is nested within another
multi-threaded object. Examples include when a base estimator in an ensemble
is multi-threaded or when a multi-threaded estimator (such as an ensemble) is
passed to a grid search object that is run in parallel. This issue is due to a
limitation of how `Python runs processes in parallel`_, and as such beyond the
scope of ML-Ensemble to fix. In short, when Python forks the main process, the
sub-processes will act as if they have the entire machine's threads at their
disposal when in fact only the main thread has been forked. Hence, every
sub-procsses will send jobs to all threads, causing a grid lock where every
thread is waiting on each other. Your main process (where you executed the
code) will not fail, but simple stall. You can detect this issue if your
activity monitor shows not Python process with significant CPU usage (< 5%).
Your script is executing, but not doing anything.

The simplest solution to avoid this is to turn off multi-threading in the
sub-processes, for instance by setting ``n_jobs`` (or ``nthread``) to ``1``.
Since an ensemble or grid search runs processes in parallel as is, this will
not cause a significant performance drop.

In Python 3.4+, it is possible to circumvent this issue by using the
``forkprocess`` backend within the native python ``multiprocessing`` library.
To do this, set the multiprocessing start method to ``forkserver`` as below. ::

    import multiprocessing

    # Imports and functions/class definitions

    if __name__ == '__main__':

        multiprocessing.set_start_method('forkserver')

        # Your execution here

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

.. _view: http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
.. _Python runs processes in parallel: https://wiki.python.org/moin/ParallelProcessing
.. _here: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
.. _issue tracker: https://github.com/flennerhag/mlens/issues
