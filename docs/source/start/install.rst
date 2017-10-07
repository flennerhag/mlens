
.. Install instructions

.. _install:

Install
=======

ML-Ensemble is available through PyPi: 

.. code-block:: bash

    pip install -U mlens

Bleeding edge
^^^^^^^^^^^^^

For the bleeding edge version install via Github_. You can optionally checkout pending pull requests
before running the install command.

.. code-block:: bash

    git clone https://github.com/flennerhag/mlens.git; cd mlens;
    python install setup.py

Dependencies
^^^^^^^^^^^^

To install :mod:`mlens` the following dependencies are required:

============  =======  ======================
Package       Version   Module
============  =======  ======================
scipy         >= 0.17  All
numpy         >= 1.11  All
============  =======  ======================

Additionally, to use the :mod:`visualization` module, the following
libraries are necessary:

============  =======
Package       Version
============  =======
matplotlib    >= 1.5
seaborn       >= 0.7
============  =======

If you want to run examples, you also need:

============  =======
Package       Version
============  =======
Scikit-learn  >= 0.17
============  =======

Test build
==========

To test the installation, run:

.. code-block:: bash

    cd mlens;
    python check_build.py

Note that this requires the Nose_ unit testing suite: if not found, the test
script will automatically try to install it using
``pip install nose-exclude``. The expected output should look like:

.. code-block:: bash

    >>> python check_build.py
    Setting up tests... Ready.
    Checking build... Build ok.

If the build fails, a log file will be created named ``check_build_log.txt``
that contains the traceback for the failed test for debugging.

.. _Nose: http://nose.readthedocs.io/en/latest/
.. _Github: https://github.com/flennerhag/mlens
