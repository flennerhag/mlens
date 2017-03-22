"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT
"""

from .stack import Stacker
from .blend import Blender
from .manager import ParallelProcessing

__all__ = ['ParallelProcessing',
           'Stacker']
