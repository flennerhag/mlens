.. Documentation on ensemble scaling

.. _scaling:


.. currentmodule:: mlens.ensemble

Scale benchmarks
================

Single process vs multi-process
-------------------------------

This benchmark compares the performance of the :class:`SuperLearner` and the
:class:`BlendEnsemble` when fitted with a single process and when fitted with
four cores. The ensembles have four SVR base estimators and an SVR as final
meta learner. Hence, while the single-processed ensembles need to fit 5
SVR models consecutively, the multiprocessed ensembles need only the time
equivalent to fit 2 consecutively. As the figure below shows, there are clear
benefits to multi-processing.

.. image:: img/scale_benchmark.png
   :align: center


To replicate the benchmark, in the ``mlens`` benchmark folder, execute::

    >>> python scale_comp.py

    ML-ENSEMBLE

    Threading performance test for data set dimensioned up to (10000, 50)
    Available CPUs: 4

    Ensemble architecture
    Num layers: 2
    Fit per base layer estimator: 2 + 1
    layer-1 | Estimators: ['svr-1', 'svr-2', 'svr-3', 'svr-4'].
    layer-2 | Meta Estimator: svr

    FIT TIMES
    samples
       1000 SuperLearner (1) :   0.88s | BlendEnsemble (1) :   0.35s |
       1000 SuperLearner (4) :   0.71s | BlendEnsemble (4) :   0.41s |

       2000 SuperLearner (1) :   2.82s | BlendEnsemble (1) :   0.76s |
       2000 SuperLearner (4) :   1.51s | BlendEnsemble (4) :   0.59s |

       3000 SuperLearner (1) :   6.04s | BlendEnsemble (1) :   1.56s |
       3000 SuperLearner (4) :   2.96s | BlendEnsemble (4) :   0.90s |

       4000 SuperLearner (1) :  10.94s | BlendEnsemble (1) :   2.79s |
       4000 SuperLearner (4) :   7.92s | BlendEnsemble (4) :   1.53s |

       5000 SuperLearner (1) :  18.45s | BlendEnsemble (1) :   4.58s |
       5000 SuperLearner (4) :   8.52s | BlendEnsemble (4) :   2.26s |

       6000 SuperLearner (1) :  27.48s | BlendEnsemble (1) :   7.24s |
       6000 SuperLearner (4) :  15.06s | BlendEnsemble (4) :   3.41s |

       7000 SuperLearner (1) :  38.73s | BlendEnsemble (1) :   8.62s |
       7000 SuperLearner (4) :  18.21s | BlendEnsemble (4) :   4.41s |

       8000 SuperLearner (1) :  52.08s | BlendEnsemble (1) :  12.10s |
       8000 SuperLearner (4) :  23.43s | BlendEnsemble (4) :   4.95s |

       9000 SuperLearner (1) :  61.70s | BlendEnsemble (1) :  14.58s |
       9000 SuperLearner (4) :  28.55s | BlendEnsemble (4) :   8.45s |

      10000 SuperLearner (1) :  75.76s | BlendEnsemble (1) :  18.72s |
      10000 SuperLearner (4) :  32.71s | BlendEnsemble (4) :   7.52s |

    Benchmark done | 00:09:00
