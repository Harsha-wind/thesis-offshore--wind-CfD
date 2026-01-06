import matplotlib.pyplot as plt

def set_style(body_pt=10, theme="latex_like"):
    """
    Apply a consistent style for plots.
    body_pt : int
        Match with LaTeX body text size (10, 11, or 12).
    theme : str
        "latex_like" = Computer/Latin Modern look
        "times_like" = Times Roman look
    """
    if theme == "latex_like":
        font_family = "serif"
        font_serif = ["CMU Serif", "Latin Modern Roman", 
                      "Computer Modern Roman", "DejaVu Serif"]
    else:  # "times_like"
        font_family = "serif"
        font_serif = ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"]

    plt.rcParams.update({
        # Figure & lines (but NOT colors)
        "figure.figsize": (12, 5),
        "lines.linewidth": 2,
        "lines.markersize": 6,

        # Fonts & sizes
        "font.size": body_pt,
        "font.family": font_family,
        "font.serif": font_serif,
        "axes.titlesize": body_pt,
        "axes.labelsize": body_pt,
        "legend.fontsize": max(body_pt-1, 8),
        "xtick.labelsize": max(body_pt-1, 8),
        "ytick.labelsize": max(body_pt-1, 8),

        # Grid & save
        
        "savefig.dpi": 300,
    })
