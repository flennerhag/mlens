"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Computational graph module for memory-neutral parallel processing of
deep general-purpose ensembles.

Implements backend graph managers, base classes for interacting with graph
managers, and job managers for preprocessing pipelines and estimators, as well
as handles for multiple instances and wrappers for standard parallel job calls.
"""
from .backend import ParallelProcessing, ParallelEvaluation, Job, dump_array
from .learner import Learner, EvalLearner, Transformer, EvalTransformer
from .layer import Layer
from .handles import Group, make_group, Pipeline
from .wrapper import run, get_backend

__all__ = ['ParallelProcessing',
           'ParallelEvaluation',
           'Job',
           'Layer',
           'Group',
           'Pipeline',
           'Learner',
           'Transformer',
           'EvalLearner',
           'EvalTransformer',
           'make_group',
           'run',
           'get_backend',
           'dump_array'
           ]
