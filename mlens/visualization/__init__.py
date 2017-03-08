"""ML-ENSEMBLE

author: Sebastian Flennerhag
"""

try:
    from .correlations import corrmat, clustered_corrmap, corr_X_y
    from .var_analysis import IO_plot_comp, IO_plot, exp_var_plot
except ImportError:
    pass

__all__ = ['corrmat', 'clustered_corrmap', 'corr_X_y',
           'IO_plot_comp', 'IO_plot', 'exp_var_plot']
