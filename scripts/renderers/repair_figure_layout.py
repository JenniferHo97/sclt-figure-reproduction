"""Compatibility entry point for the equal-height Figure 6 renderer.

The default call requires strictly complete FT results. Explicit layout proofs
must pass require_complete=False and allow_incomplete=True and remain isolated.
"""
from repair_figure_equal_layout import draw_figure, load_plot_inputs

__all__ = ['draw_figure', 'load_plot_inputs']
