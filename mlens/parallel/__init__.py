"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT
"""

from .stacking import Stacker
from .base import ParallelProcessing

__all__ = ['ParallelProcessing',
           'Stacker']
