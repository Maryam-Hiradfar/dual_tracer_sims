"""PowerPoint-friendly Matplotlib defaults for PET/TCN evaluation plots.

Apply once before creating plots. Only visual settings are changed;
numerical data, predictions, evaluation metrics and model stay untouched.
"""
from __future__ import annotations

import matplotlib as mpl


def use_slide_plot_style():
    """Readable at PowerPoint presentation size, including exported PDF."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "font.weight": "bold",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.6,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 2.6,
        "lines.markersize": 7,
        "legend.fontsize": 13,
        "figure.titlesize": 19,
        "figure.titleweight": "bold",
        "grid.linewidth": 0.9,
        "savefig.dpi": 300,
    })
