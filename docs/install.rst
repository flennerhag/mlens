Install
=======

ML-Ensemble is available through PyPi. For latest stable version, install
``mlens`` through ``pip``. ::

    pip install mlens


Bleeding edge
^^^^^^^^^^^^^

ML-Ensemble is rapidly developing. To keep up with the latest stable
development version, install the ``master`` branch of the Github repository. ::

    git clone https://flennerhag/mlens.git; cd mlens;
    python install setup.py

To update, first pull the latest changes, and re-install::

    git pull; python install setup.py

To avoid explicitly re-installing, one can install the repository with
symlinks (Mac, Linux) enabled through ``pip``::

    git clone https://flennerhag/mlens.git; cd mlens;
    pip install -e .

This requires only pulling the latest changes to update the library.

Developer
^^^^^^^^^

For the latest in-development version, install the ``dev`` branch of the
``mlens`` repository. It is advised to check Travis build history
first to ensure the current version does not contain apparent errors.

Dependencies
^^^^^^^^^^^^

To run ``mlens`` the following dependencies are required:

============  =======  ======================
Package       Version   Module
============  =======  ======================
scipy         >= 0.17  All
numpy         >= 1.11  All
joblib        >= 0.9   All
scikit-learn  >= 0.18  All
pandas        >= 0.17  All
matplotlib    >= 1.5   Only for visualization
seaborn       >= 0.7   Only for visualization
============  =======  ======================

Test Build
==========

To test the installation, run ::

    cd mlens;
    python check_build.py

Note that this requires the Nose_ unit testing suite: if not found, the test
script will automatically try to install it using
``pip install nose nose-exclude``. The expected output should look like::

    >>> python check_build.py
    Setting up tests... Ready.
    Checking build... Build ok.

If the build fails, a log file will be created named ``check_build_log.txt``
that contains the traceback for the failed test for debugging.

.. _Nose: http://pypi.org/nose
