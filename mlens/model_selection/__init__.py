"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Model selection suite for ensemble learning.

Implements computational graph framework for maximally parallelized
cross-validation of an arbitrary set of estimators over an arbitrary set
of preprocessing pipelines. Model selection suite features batch randomized
grid search and batch benchmarking. Ensembles can be treated as preprocessing
pipelines for next-layer model selection.
"""

from .model_selection import BaseEval, Evaluator, Benchmark, benchmark
from .ensemble_transformer import EnsembleTransformer


__all__ = ['BaseEval', 'Evaluator',
           'EnsembleTransformer', 'Benchmark', 'benchmark']
