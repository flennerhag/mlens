"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Base modules
"""

from .base import FullIndex
from .fold import FoldIndex
from .blend import BlendIndex
from .subsemble import SubsetIndex, ClusteredSubsetIndex


INDEXERS = {'stack': FoldIndex,
            'blend': BlendIndex,
            'subsemble': SubsetIndex,
            'clusteredsubsemble': ClusteredSubsetIndex,
            'full': FullIndex
            }


__all__ = ['BlendIndex',
           'FoldIndex',
           'SubsetIndex',
           'FullIndex',
           'ClusteredSubsetIndex'
           ]
