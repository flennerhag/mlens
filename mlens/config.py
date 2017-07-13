"""
Global configurations.
"""

import tempfile
from numpy import float32

BACKEND = 'threading'
DTYPE = float32
TMPDIR = tempfile.gettempdir()
