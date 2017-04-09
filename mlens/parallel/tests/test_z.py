"""

Close cache.
"""

import os
import shutil
import warnings


def test_cache():
    """[Parallel] close cache."""
    try:
        shutil.rmtree(os.path.join(os.getcwd(), 'tmp'))
    except:
        warnings.warn("Failed to remove temporary cache.")
