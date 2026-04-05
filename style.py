"""
style.py — Shared Plotting Style for EE1204 Simulation Assignment
=================================================================

This module sets consistent matplotlib aesthetics (fonts, sizes, colours,
colour-maps) so that every figure across all three questions has a uniform
professional appearance.

Usage
-----
    import style          # simply importing executes rcParams update

    # Then use the colour constants anywhere:
    ax.plot(x, y, color=style.PLATE_POS)

Author : Aniket (EE25BTECH11007)
Course : EE1204 — Engineering Electromagnetics, IIT Hyderabad
Date   : March 2026
"""

import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
#  Global matplotlib style — applied on import
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    # --- Resolution ---
    'figure.dpi':        220,         # screen preview resolution
    'savefig.dpi':       220,         # saved-file resolution

    # --- Typography ---
    'font.family':       'serif',     # serif fonts (Computer Modern in LaTeX)
    'font.size':         11,          # base font size (pt)
    'axes.titlesize':    13,
    'axes.labelsize':    11.5,
    'legend.fontsize':   9,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,

    # --- Lines & grids ---
    'axes.linewidth':    0.8,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.5,
    'lines.linewidth':   1.4,

    # --- Background ---
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',

    # --- Saving ---
    'savefig.bbox':      'tight',     # crop whitespace around figure
    'savefig.pad_inches': 0.15,
})

# ──────────────────────────────────────────────────────────────
#  Colour palette — used throughout all question scripts
# ──────────────────────────────────────────────────────────────
PLATE_POS   = '#2166ac'     # blue  — positive plate / conductor
PLATE_NEG   = '#b2182b'     # red   — negative plate / point charge
ACCENT      = '#1b9e77'     # teal  — dielectric interface / needle
ACCENT2     = '#d95f02'     # orange — secondary accent
NEUTRAL     = '#636363'     # grey  — grid lines, annotations
BG_LIGHT    = '#f7f7f7'     # off-white background

# ──────────────────────────────────────────────────────────────
#  Colour maps — chosen for specific physical quantities
# ──────────────────────────────────────────────────────────────
CMAP_POT    = 'coolwarm'    # potential (diverging: blue–white–red)
CMAP_FIELD  = 'inferno'     # field magnitude (sequential, perceptual)
CMAP_HEAT   = 'magma'       # heat maps (sequential, dark background)
CMAP_SEQ    = 'viridis'     # generic sequential data
