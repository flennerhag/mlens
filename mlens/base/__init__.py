"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base modules
"""

from .id_train import IdTrain
from .indexer import FullIndex, BlendIndex


INDEXERS = {'stack': FullIndex,
            'blend': BlendIndex}


__all__ = ['IdTrain', 'BlendIndex', 'FullIndex']
