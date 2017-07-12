.. configurations



.. _configs:

Global configurations
=====================

ML-Ensemble allows a set of low-level global configurations to tailor the
behavior of classes during estimation. Every variable is accessible through
``mlens.config``.

* ``mlens.config.BACKEND``
    configures the global default backend during parallelized estimation.
    Default is ``'threading'``. Options are ``'multiprocessing'`` and
    ``'forkserver'``. See joblib_ for further information.

* ``mlens.config.DTYPE``
    determines the default dtype of numpy arrays created during estimation; in
    particular, the prediction matrices of each intermediate layer. Default is
    :obj:`numpy.float32`.

* ``mlens.config.TMPDIR``
    The directory where temporary folders are created during estimation.
    Default uses the tempfile_ function ``gettempdir()``.

.. _joblib: https://pythonhosted.org/joblib/parallel.html
.. _tempfile: https://docs.python.org/3/library/tempfile.html
