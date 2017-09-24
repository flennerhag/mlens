"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT
"""
from .layer import Layer
from .evaluation import Evaluation
from .learner import Learner, EvalLearner, Transformer
from .manager import ParallelProcessing, ParallelEvaluation

__all__ = ['ParallelProcessing',
           'ParallelEvaluation',
           'Evaluation',
           'Layer',
           'Learner',
           'EvalLearner',
           'Transformer',
           ]
