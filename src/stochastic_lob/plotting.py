"""Figure style shared by all experiments: inward ticks on four sides, print-size type."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = Path(__file__).with_name("journal.mplstyle")
SINGLE = (3.5, 2.625)   # one journal column (in)
WIDE = (7.2, 2.8)       # full page width (in)


def use_style():
    plt.style.use(STYLE)


def panel_label(ax, text):
    """Bold panel tag such as (a), placed just above the upper-left corner of the axes."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def style_3d(ax, elev=24.0, azim=-58.0, labelsize=7.5):
    """Light panes, faint grid and print-size tick labels for a 3D axes."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        try:
            axis._axinfo["grid"].update(color="0.85", linewidth=0.4)
            axis._axinfo["tick"].update(inward_factor=0.4, outward_factor=0.0)
        except (AttributeError, KeyError):
            pass
    ax.tick_params(labelsize=labelsize, pad=0.5)
    ax.view_init(elev=elev, azim=azim)
    return ax


def save(fig, stem, outdir, width_in=None):
    """Save as PNG at 300 dpi (the only figure format kept in the repository).

    If the environment variable FIGCHECK names a folder holding check_figure.py
    (a journal-figure audit script), the figure is audited before saving.
    """
    import os
    import sys

    check_dir = os.environ.get("FIGCHECK")
    if check_dir:
        sys.path.insert(0, check_dir)
        from check_figure import audit  # type: ignore

        print(f"--- audit {stem}")
        audit(fig, target_width_in=width_in or fig.get_figwidth())
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=300)
