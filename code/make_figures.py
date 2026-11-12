"""Figures for the paper (print, light background).

Palette: first three categorical slots of the validated reference palette
(blue #2a78d6, orange #eb6834, aqua #1baf7a), each with its own marker and line
style, since aqua sits below 3:1 contrast. Floors and reference lines in neutral grey.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
STYLE = {"border": (BLUE, "o", "-"), "stealthy": (ORANGE, "s", "-"), "random": (AQUA, "^", "--")}


def fig_distortion():
    e3 = json.load(open(os.path.join(RES, "e3.json")))
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.85), sharey=True)
    for ax, info in zip(axes, e3["info"]):
        ds = info["dataset"]
        for strat, (col, mk, ls) in STYLE.items():
            rows = sorted([r for r in e3["rows"] if r["dataset"] == ds and r["strategy"] == strat], key=lambda r: r["g"])
            g = [r["g"] for r in rows]
            k = [r["distortion"] for r in rows]
            ax.plot(g, k, ls, color=col, lw=1.1, marker=mk, ms=2.8, markevery=max(1, len(g) // 8), label=strat)
        rows = sorted([r for r in e3["rows"] if r["dataset"] == ds and r["strategy"] == "stealthy"], key=lambda r: r["g"])
        ax.plot([r["g"] for r in rows], [r["lower_bound"] for r in rows], ":", color=MUTED, lw=1.2,
                label="floor (Thm. 3), stealthy")
        ax.axvline(info["g_res"], color=GRID, lw=1.0, zorder=0)
        ax.text(info["g_res"] * 1.06, 1100, r"$g_{\mathrm{res}}$", color=MUTED, fontsize=6, ha="left", va="top")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(ds, color=INK)
        ax.set_xlabel("targets moved $g$")
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_ylim(0.9, 1500)
    axes[0].set_ylabel(r"distortion $\kappa(d)$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "distortion.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "distortion.png"), dpi=200, bbox_inches="tight")
    print("wrote distortion")


def fig_detection():
    p = os.path.join(RES, "e4.json")
    e4 = json.load(open(p))
    rows = e4["rows"]
    fams = ["weighted Euclidean", "Mahalanobis", "Manhattan", "Chebyshev", "cosine", "NCA"]
    atk = [("relocation-stealthy-g1", "stealthy, $g=1$"), ("relocation-stealthy-g10", "stealthy, $g=10$"),
           ("relocation-random-g1", "random, $g=1$"), ("relocation-random-g10", "random, $g=10$"),
           ("two-level (random labels)", "two-level")]
    labels = fams + [a[1] for a in atk]
    data = [[np.exp(r["distort"]) for r in rows if r["variant"] == f] for f in fams] + \
           [[np.exp(r["distort"]) for r in rows if r["variant"] == a[0]] for a in atk]
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        col = MUTED if i < len(fams) else ORANGE
        y = len(data) - 1 - i
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(vals, y + jitter, s=5, color=col, alpha=0.55, linewidths=0)
        ax.plot([np.median(vals)] * 2, [y - 0.3, y + 0.3], color=INK, lw=1.2)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels[::-1])
    ax.axhline(len(atk) - 0.5, color=GRID, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"distortion $\kappa(d)$ relative to standardised Euclidean distance")
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.text(0.99, 0.98, "legitimate metrics (grey)\nattacks (orange)", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "detection.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "detection.png"), dpi=200, bbox_inches="tight")
    print("wrote detection")


if __name__ == "__main__":
    fig_distortion()
    if os.path.exists(os.path.join(RES, "e4.json")) and "auc" in json.load(open(os.path.join(RES, "e4.json"))):
        fig_detection()
