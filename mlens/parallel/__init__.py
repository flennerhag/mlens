"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT
"""
from .backend import ParallelProcessing, ParallelEvaluation
from .learner import Learner, EvalLearner, Transformer, EvalTransformer
from .layer import Layer
from .handles import Group, make_group, Pipeline
from .wrapper import run, get_backend

__all__ = ['ParallelProcessing',
           'ParallelEvaluation',
           'Layer',
           'Group',
           'Pipeline'
           'Learner',
           'Transformer',
           'EvalLearner',
           'EvalTransformer',
           'make_group',
           'run',
           'get_backend',
           ]
