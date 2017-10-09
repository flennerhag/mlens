"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT
"""
from .manager import ParallelProcessing, ParallelEvaluation
from .learner import Learner, EvalLearner, Transformer
from .layer import Layer

__all__ = ['ParallelProcessing',
           'ParallelEvaluation',
           'Layer',
           'Learner',
           'EvalLearner',
           'Transformer',
           ]
