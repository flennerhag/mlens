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

__version__ = "0.1.6"

__all__ = ['base',
           'utils',
           'metrics',
           'parallel',
           'ensemble',
           'externals',
           'visualization',
           'preprocessing',
           'model_selection',
          ]
