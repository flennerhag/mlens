# -*- coding: utf-8 -*-
"""

.. _parallel_tutorial:


.. currentmodule: mlens.parallel.learner

Parallel Mechanics
==================

ML-Ensemble is designed to provide an easy user interface. But it is also designed
to be extremely flexible, all the wile providing maximum concurrency at minimal
memory consumption. The lower-level API that builds the ensemble and manages the
computations is constructed in as modular a fashion as possible.

The low-level API introduces a computational graph-like environment that you can
directly exploit to gain further control over your ensemble. In fact, building
your ensemble through the low-level API is almost as straight forward as using the
high-level API. In this tutorial, we will walk through the key core :class:`Learner` class.


"""
from mlens.parallel import ParallelProcessing
