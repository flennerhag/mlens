.. configurations



.. _configs:

Global configurations
=====================

ML-Ensemble allows a set of low-level global configurations to tailor the
behavior of classes during estimation. Every variable is accessible through
``mlens.config``. Alternatively, all variables can be set as global
environmental variables, where the exported variable name is
``MLENS_[VARNAME]``.

* ``mlens.config.BACKEND``
    configures the global default backend during parallelized estimation.
    Default is ``'threading'``. Options are ``'multiprocessing'`` and
    ``'forkserver'``. See joblib_ for further information. Alter with the
    ``set_backend`` function.

* ``mlens.config.DTYPE``
    determines the default dtype of numpy arrays created during estimation; in
    particular, the prediction matrices of each intermediate layer. Default is
    :obj:`numpy.float32`. Alter with the ``set_backend`` function.

* ``mlens.config.TMPDIR``
    The directory where temporary folders are created during estimation.
    Default uses the tempfile_ function ``gettempdir()``. Alter with the
    ``set_backend`` function.

* ``mlens.config.START_METHOD``
    The method used by the job manager to generate a new job. ML-Ensemble
    defaults to ``forkserver``on Unix with Python 3.4+, and ``spawn`` on
    windows. For older Python versions, the default is ``fork``. This method
    has the least overhead, but it can cause issues with third-party software.
    See :ref:`third-party-issues` for details. Set this variable with the
    ``set_start_method`` function.

.. _joblib: https://pythonhosted.org/joblib/parallel.html
.. _tempfile: https://docs.python.org/3/library/tempfile.html
