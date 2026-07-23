"""Shared figure style (dataviz reference palette, used unchanged).

Palette is the validated reference instance (categorical slot order is the CVD-safety
mechanism; documented worst adjacent CVD dE = 24.2 in light mode). We do not substitute a
brand palette, so the documented validation applies as-is.

Rules enforced here:
  * color by JOB: categorical = identity, sequential = magnitude, diverging = polarity
  * signed quantities (influence) use the diverging blue<->red pair with a GRAY midpoint
  * NEVER a dual-axis chart -- two measures of different scale get two panels
  * recessive grid/axes, thin marks, text in ink tokens (never in a series color)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# categorical slots (light mode), in the fixed validated order
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE = CAT

# chrome & ink
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

# sequential (single hue, light->dark) and diverging (blue <-> red, gray midpoint)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_br", ["#184f95", "#86b6ef", "#f0efec",
                                                   "#e88b8a", "#b52d2c"])

INSIDER, OUTSIDER = BLUE, ORANGE          # 2-category identity used across figures


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 160,
        "font.size": 9,
    })


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
