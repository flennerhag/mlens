"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

ML-Ensemble, a Python library for memory efficient parallelized ensemble
learning.
"""
# Initialize configurations
import mlens.config
from mlens.config import clear_cache

__version__ = "0.2.0.dev0"

__all__ = ['index',
           'utils',
           'metrics',
           'parallel',
           'ensemble',
           'visualization',
           'preprocessing',
           'model_selection',
           'externals',
           ]
