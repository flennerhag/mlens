"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base modules
"""

from .id_train import IdTrain
from .indexer import FoldIndex, BlendIndex, SubsetIndex, FullIndex


INDEXERS = {'stack': FoldIndex,
            'blend': BlendIndex,
            'subset': SubsetIndex,
            'full': FullIndex}


__all__ = ['IdTrain', 'BlendIndex', 'FoldIndex', 'SubsetIndex',
           'FullIndex']
