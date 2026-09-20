"""Shared typography and colours for the paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

FIGWIDTH = 6.5  # inches, = \linewidth

FONTSIZE = 9
LEGENDSIZE = 7.5
PANEL_LABEL_SIZE = 10

# Method colours.
COLORS = {
    "base": "navy",
    "lightning": "crimson",
    "graw": "darkgreen",
    "ecfp": "darkorange",
    "chemeleon": "indigo",

    "chemeleon_graw": "mediumvioletred",
    "ecfp_graw": "saddlebrown",
    "drugclip_zeroshot": "#607d8b",
    "drugclip_headft": "#008b8b",
}

# Label-budget ramp: darker = more labels.
N_COLORS = {40: "#f4a0a8", 100: "#d94a5c", 300: "crimson"}


def panel_label(ax, letter: str, x: float = None) -> None:
    """Draw a bold 'a)' / 'b)' / 'c)' panel label at the top-left of ax."""
    ax.set_title(letter, loc="left", fontsize=PANEL_LABEL_SIZE, fontweight="bold")


def savefig_both(fig, out_path: Path) -> None:
    """Save both .png (dpi=600) and vector .pdf at the same print dimensions."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    pdf = out_path.with_suffix(".pdf")
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    print(f"Wrote {png} and {pdf}")
